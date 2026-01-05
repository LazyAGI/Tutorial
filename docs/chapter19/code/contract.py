import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn

# --- 网络结构 ---
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- PPO 算法核心 ---
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, 
                 gamma, lmbda, epochs, eps, device, mode="GAE"):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.mode = mode # "GAE", "MC", or "TD"

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, rewards, values, next_values, dones):
        rewards = rewards.cpu().flatten().numpy()
        values = values.cpu().flatten().detach().numpy()
        next_values = next_values.cpu().flatten().detach().numpy()
        dones = dones.cpu().flatten().numpy()

        if self.mode == "GAE":
            td_delta = rewards + self.gamma * next_values * (1 - dones) - values
            advantage = 0.0
            advantage_list = []
            for delta in reversed(td_delta):
                advantage = self.gamma * self.lmbda * advantage + delta
                advantage_list.append(advantage)
            advantage_list.reverse()
            return torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)

        elif self.mode == "MC":
            # 计算折扣回报 G_t
            returns = []
            G = 0
            for r, d in zip(reversed(rewards), reversed(dones)):
                G = r + self.gamma * G * (1 - d)
                returns.append(G)
            returns.reverse()
            # A = G_t - V(s)
            adv = np.array(returns) - values
            return torch.tensor(adv, dtype=torch.float).to(self.device)

        elif self.mode == "TD":
            # A = r + gamma*V(s') - V(s)
            adv = rewards + self.gamma * next_values * (1 - dones) - values
            return torch.tensor(adv, dtype=torch.float).to(self.device)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict["states"]), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict["actions"]), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict["rewards"]), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict["next_states"]), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict["dones"]), dtype=torch.float).view(-1, 1).to(self.device)

        # 准备优势函数计算所需的值
        values = self.critic(states)
        next_values = self.critic(next_states)
        
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantage = self.compute_advantage(rewards, values, next_values, dones).view(-1, 1)
        
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = ratio.clamp(1 - self.eps, 1 + self.eps) * advantage
            
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

# --- 训练函数 ---
def train_ppo(mode):
    print(f"\n开始训练模式: {mode}")
    actor_lr, critic_lr = 1e-3, 1e-2
    num_episodes = 200 
    hidden_dim, gamma, lmbda, epochs, eps = 64, 0.98, 0.95, 10, 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("CartPole-v1")
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, 
                gamma, lmbda, epochs, eps, device, mode=mode)

    return_list = []
    for i in range(10):
        with tqdm(total=num_episodes//10, desc=f"{mode} Iter {i}") as pbar:
            for episode in range(num_episodes//10):
                episode_return = 0
                transition_dict = {"states":[], "actions":[], "rewards":[], "next_states":[], "dones":[]}
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["rewards"].append(reward)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                pbar.update(1)
    env.close()
    return return_list

# --- 运行对比实验 ---
if __name__ == "__main__":
    modes = ["GAE", "MC", "TD"]
    all_results = {}

    for m in modes:
        all_results[m] = train_ppo(m)

    # 绘图对比
    plt.figure(figsize=(10, 6))
    for m in modes:
        # 使用滑动平均使曲线平滑
        data = np.array(all_results[m])
        smooth_data = [np.mean(data[max(0, i-10):i+1]) for i in range(len(data))]
        plt.plot(smooth_data, label=f"Advantage: {m}")

    plt.xlabel("Episodes")
    plt.ylabel("Filtered Returns")
    plt.title("Comparison of Advantage Methods in PPO")
    plt.legend()
    plt.grid(True)
    plt.show()