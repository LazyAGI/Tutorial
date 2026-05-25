import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

model_name = 'shibing624/text2vec-base-chinese'
try:
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    print('模型加载成功！')
except Exception:
    print('请检查网络或模型路径')

chosen_rewards = torch.tensor([1.5, 2.3], requires_grad=True)
rejected_rewards = torch.tensor([0.2, -0.5], requires_grad=True)

loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

print(f'Chosen 分数: {chosen_rewards.data}')
print(f'Rejected 分数: {rejected_rewards.data}')
print(f'计算得到的 Loss: {loss.item():.4f}')

loss.backward()
print('反向传播成功，梯度已更新。')
