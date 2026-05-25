import os

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from trl import GRPOTrainer, GRPOConfig  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from datasets import load_dataset  # noqa: E402
import torch  # noqa: E402
import re  # noqa: E402

# === 配置路径与模型 ===
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'  # 演示建议用小模型
OUTPUT_DIR = './qwen-grpo-output'


def load_model_and_tokenizer(model_name):
    '''加载模型和分词器，如果失败则提示用户配置'''
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(
            f'\n{"="*60}\n'
            f'模型加载失败: {model_name}\n'
            f'错误信息: {e}\n'
            f'{"="*60}\n'
            f'请检查以下配置:\n'
            f'1. 设置 HuggingFace Token: export HF_TOKEN=your_token\n'
            f'2. 或修改 MODEL_NAME 为本地模型路径\n'
            f'3. 或配置镜像: export HF_ENDPOINT=https://hf-mirror.com\n'
            f'{"="*60}\n'
        ) from e


# 加载模型和 tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)


# === 1. 加载数据 ===
def get_gsm8k_dataset():
    '''加载 GSM8K 数据集用于 GRPO 训练'''
    dataset = load_dataset('gsm8k', 'main', split='train[:30%]')

    # 数据预处理 - 新版 TRL 支持标准格式和对话格式
    # 标准格式: 包含 'prompt' 列的文本
    dataset = dataset.map(
        lambda x: {'prompt': x['question']},
        remove_columns=['question', 'answer']
    )
    return dataset


# 加载数据集
dataset = get_gsm8k_dataset()
print(f'Dataset size: {len(dataset)}')
print(f'Sample prompt: {dataset[0]["prompt"]}')


# === 2. 定义奖励函数 (新版 TRL API) ===
# 注意: 新版 TRL 的奖励函数必须接受 **kwargs 参数
# 可用参数包括: prompts, completions, completion_ids, trainer_state 等

def format_reward_func(completions, **kwargs):
    '''格式奖励：鼓励模型生成数学推理格式'''
    rewards = []
    for completion in completions:
        # 检查是否包含推理步骤和最终答案
        if 'reasoning' in completion.lower() and 'answer' in completion.lower():
            rewards.append(1.0)  # 格式正确给奖励
        elif any(keyword in completion.lower() for keyword in
                 ['step', 'calculate', 'therefore']):
            rewards.append(0.5)  # 包含推理关键词给部分奖励
        else:
            rewards.append(0.0)  # 无格式扣分
    return rewards


def soft_length_reward_func(completions, **kwargs):
    '''长度奖励：鼓励简洁但不奖励过短回答'''
    rewards = []
    for completion in completions:
        length = len(completion.split())
        if 10 <= length <= 100:  # 合适长度范围
            rewards.append(1.0)
        elif length < 10:  # 过短
            rewards.append(0.1)
        else:  # 过长
            rewards.append(0.5)
    return rewards


def math_accuracy_reward_func(completions, prompts, **kwargs):
    '''数学准确性奖励：检查答案是否正确'''
    rewards = []

    for completion, _prompt in zip(completions, prompts):
        try:
            # 提取最终答案（假设格式为 #### 数字）
            answer_match = re.search(r'####\s*(\d+(?:\.\d+)?)', completion)
            if answer_match:
                # predicted_answer 提取但不使用，用于未来扩展
                _ = float(answer_match.group(1))  # noqa: F841
                # 这里可以添加更复杂的数学验证逻辑
                # 暂时给提取到答案的样本奖励
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


# === 3. 配置 GRPO 参数 ===
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    # 【关键】：每条 Prompt 生成 8 个样本做组内对比
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=512,
    num_train_epochs=1,
    bf16=True,
    report_to='none',
    logging_steps=1,
    save_steps=10,
    save_total_limit=2,
    # 新版 TRL 默认 beta=0.0 (不使用 KL 散度项)
    # 如需启用 KL 散度，取消下面注释:
    # beta=0.01,
)

# === 4. 初始化 GRPOTrainer (新版 TRL API) ===
# 新版 TRL 支持直接传入模型名称字符串，会自动加载模型
# 也可以继续传入已加载的 model 对象（推荐，可以自定义加载参数）
trainer = GRPOTrainer(
    model=model,  # 传入已加载的模型对象
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        format_reward_func,           # 格式奖励
        soft_length_reward_func,      # 长度奖励
        math_accuracy_reward_func     # 准确性奖励
    ],
    # 传入自定义 tokenizer
    processing_class=tokenizer,
)

# === 5. 执行训练 ===
print('>>> 开始 GRPO 强化学习训练...')
print(f'训练数据集大小: {len(dataset)}')
print(f'每轮生成样本数: {training_args.num_generations}')
total_steps = (
    len(dataset) * training_args.num_train_epochs
    // (training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps)
)
print(f'总训练步数: {total_steps}')

trainer.train()

# === 6. 保存结果 ===
trainer.save_model(OUTPUT_DIR)
print(f'模型已保存到: {OUTPUT_DIR}')


# === 7. 推理测试 ===
print('\n=== 推理测试 ===')

def generate_response(prompt, max_length=512):
    '''生成推理响应'''
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# 测试几个数学问题
test_questions = [
    'Natalia sold clips to 48 of her friends in April, and then she sold '
    'half as many clips in May. How many clips did Natalia sell altogether '
    'in April and May?',
    'A bakery has 30 cupcakes and 24 cookies. If they sell 12 cupcakes and '
    '8 cookies, how many items do they have left?',
]

for i, question in enumerate(test_questions, 1):
    print(f'\n测试问题 {i}: {question}')
    response = generate_response(
        f'Solve this math problem step by step:\n\n{question}\n\nReasoning:'
    )
    print(f'模型回答: {response}')
    print('-' * 50)
