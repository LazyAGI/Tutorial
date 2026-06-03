import os

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from trl import RewardTrainer, RewardConfig  # noqa: E402

# === 配置路径与模型 ===
MODEL_NAME = 'gpt2'
OUTPUT_DIR = './reward_model_output'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 直接从 HuggingFace 下载数据集
dataset = load_dataset('Anthropic/hh-rlhf', split='train')

print(f'DEBUG: Dataset columns are: {dataset.column_names}')
print(f'DEBUG: First example: {dataset[0]}')

# === 2. 预处理 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples):
    new_examples = {
        'input_ids_chosen': [],
        'attention_mask_chosen': [],
        'input_ids_rejected': [],
        'attention_mask_rejected': [],
    }

    for chosen, rejected in zip(examples['chosen'], examples['rejected']):
        tokenized_chosen = tokenizer(chosen, truncation=True, max_length=512)
        tokenized_rejected = tokenizer(rejected, truncation=True, max_length=512)

        new_examples['input_ids_chosen'].append(tokenized_chosen['input_ids'])
        new_examples['attention_mask_chosen'].append(
            tokenized_chosen['attention_mask']
        )
        new_examples['input_ids_rejected'].append(tokenized_rejected['input_ids'])
        new_examples['attention_mask_rejected'].append(
            tokenized_rejected['attention_mask']
        )

    return new_examples


tokenized_dataset = dataset.map(
    preprocess_function, batched=True, num_proc=2
)

# === 3. 加载奖励模型 ===
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    pad_token_id=tokenizer.eos_token_id
).to(device)

# === 4. 训练参数设置 ===
training_args = RewardConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-6,
    logging_steps=100,
    save_strategy='steps',
    save_steps=100,
    max_length=512,
    report_to='none',
)
original_compute_loss = RewardTrainer.compute_loss


def patched_compute_loss(
    self, model, inputs, return_outputs=False, num_items_in_batch=None
):
    return original_compute_loss(self, model, inputs, return_outputs)


RewardTrainer.compute_loss = patched_compute_loss


# === 5. 初始化 Trainer 并开始训练 ===
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print('>>> Starting training with local dataset...')
trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f'>>> Model saved to {OUTPUT_DIR}')

# === 6. 推理测试 ===
print('-' * 30)
print('Testing the trained Reward Model...')


def get_score(text):
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.item()


# 你可以根据你的数据内容修改测试用例
good_conversation = (
    'Human: 如何缓解焦虑？\nAssistant: '
    '你可以尝试深呼吸、冥想或者去散步。如果情况严重，建议咨询心理医生。'
)
bad_conversation = (
    'Human: 如何缓解焦虑？\nAssistant: 焦虑个屁，忍着。'
)

score_good = get_score(good_conversation)
score_bad = get_score(bad_conversation)

print(f'Good Response Score: {score_good:.4f}')
print(f'Bad Response Score:  {score_bad:.4f}')

if score_good > score_bad:
    print('✅ Success: RM 成功识别出了更好的回答！')
    print(f'Gap: {score_good - score_bad:.4f}')
else:
    print('❌ Failed: 模型得分倒挂，请检查数据质量或增加训练量。')
