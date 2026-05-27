#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

import lazyllm
from lazyllm import finetune, launchers
from lazyllm.components.formatter import (
    JsonFormatter,
    encode_query_with_filepaths
)
from lazyllm.tools.data.pipelines.img_pipelines import build_img2qa_pipeline

random.seed(42)

# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'vqa_rad_simple'
IMAGE_DIR = OUTPUT_DIR / 'images'

TRAIN_JSON = OUTPUT_DIR / 'train.json'
TEST_JSON = OUTPUT_DIR / 'test.json'

SFT_RESULT_JSON = BASE_DIR / 'sft_result.json'
INFER_RESULT_JSON = BASE_DIR / 'infer_result.json'

SFT_SCORE_JSON = BASE_DIR / 'sft_score.json'
INFER_SCORE_JSON = BASE_DIR / 'infer_score.json'

os.makedirs(IMAGE_DIR, exist_ok=True)


# =========================
# Step 1: 下载 + 转换数据
# =========================
def prepare_dataset():
    print('📥 Loading dataset...')
    dataset = load_dataset('flaviagiammarino/vqa-rad')

    def process_split(split_name, save_path):
        data_out = []
        split = dataset[split_name]

        print(f'Processing {split_name}...')

        for i, sample in enumerate(tqdm(split)):
            img = sample['image'].convert('RGB')
            question = sample['question']
            answer = sample['answer']

            img_path = IMAGE_DIR / f'{split_name}_{i}.jpg'
            img.save(img_path)

            data_out.append({
                'instruction': question,
                'input': str(img_path),
                'output': answer
            })

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_out, f, ensure_ascii=False, indent=2)

    process_split('train', TRAIN_JSON)
    process_split('test', TEST_JSON)


# =========================
# Step 2: img2qa 推理增强
# =========================
def run_img2qa():
    print('🧠 Generating reasoning data...')

    gen_prompt = (
        '''Role: You are an expert radiologist and medical educator.
        Task: Given a medical image, a specific question (Context),
        and its ground-truth answer (Reference),
        your goal is to reverse-engineer the diagnostic reasoning path.

        Step-by-Step Logic:
        1. Anatomical Localization: Specify the key anatomical structures and markers relevant to the question.
        2. Radiographic Evidence: Pinpoint the specific visual characteristics
        (e.g., intensity, margins, enhancement, mass effect) that serve as evidence.
        3. Clinical Synthesis: Bridge the gap between observations and the Reference answer.

        Constraint:
        - The 'answer' field must be a cohesive, professional paragraph.
        - Format: Reasoning: [Your step-by-step analysis]. Therefore, the final answer is [Reference].
        - Output strictly in JSON format: {"query": "...", "answer": "..."}'''
    )

    model = lazyllm.TrainableModule('Qwen2.5-VL-32B-Instruct').deploy_method(
        lazyllm.deploy.vllm,
        openai_api=True
    )

    ppl = build_img2qa_pipeline(
        model=model,
        gen_prompt=gen_prompt,
        image_key='input',
        context_key='instruction',
        img_resize=True,
        size=(336, 336)
    )

    with open(TRAIN_JSON) as f:
        train_data = json.load(f)

    train_data = ppl(train_data)

    with open(TRAIN_JSON, 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    model.stop()


# =========================
# Step 3: 构建 SFT 模型
# =========================
def build_sft_model(model_path):
    return (
        lazyllm.TrainableModule(model_path, type='vlm', target_path=BASE_DIR)
        .mode('finetune')
        .trainset(str(TRAIN_JSON))
        .finetune_method(
            (
                finetune.llamafactory,
                {
                    'learning_rate': 1e-5,
                    'cutoff_len': 1024,
                    'max_samples': 10000,
                    'preprocessing_num_workers': 1,
                    'val_size': 0.1,
                    'per_device_train_batch_size': 12,
                    'num_train_epochs': 3.0,
                    'gradient_accumulation_steps': 10,
                    'overwrite_cache': False,
                    'launcher': launchers.empty(ngpus=1),
                }
            )
        )
        .prompt(
            dict(
                system='You are a medical assistant, answer the question, '
                       'answer no if you are not sure about the result.',
                drop_builtin_system=True
            )
        )
    )


# =========================
# Step 4: 推理模型
# =========================
def build_infer_model(model_path):
    return lazyllm.TrainableModule(
        model_path,
        type='vlm'
    ).prompt(
        dict(system='', drop_builtin_system=True)
    )


# =========================
# Step 5: 打分
# =========================
def score_with_model(data, scorer, save_path):
    out = []

    for x in data:
        prompt = (
            f"Q: {x['instruction']}\n"  # noqa: Q000
            f"GT: {x['output']}\n"  # noqa: Q000
            f"Pred: {x['prediction']}"  # noqa: Q000
        )

        try:
            r = scorer(prompt)
            score = int(r.get('score', 0))
        except Exception:
            score = 0

        x['score'] = score
        out.append(x)

    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)

    return out


# =========================
# Step 6: 分析
# =========================
def analyze(data, name):
    scores = [x['score'] for x in data]
    acc = sum(scores) / len(scores) if scores else 0

    counter = Counter(scores)

    print(f'\n📊 {name}')
    print(f'Accuracy: {acc:.4f}')
    print('Distribution:')
    for k in sorted(counter):
        print(f'{k}: {counter[k]}')


# =========================
# 主流程
# =========================
def main():

    # Step 1: 数据
    prepare_dataset()

    # Step 2: reasoning 增强
    run_img2qa()

    # 读取测试集
    with open(TEST_JSON) as f:
        test_data = json.load(f)

    eval_prompts = [
        encode_query_with_filepaths(x['instruction'], files=[x['input']])
        for x in test_data
    ]

    # =========================
    # Step 3: SFT
    # =========================
    sft_model = build_sft_model('Qwen2.5-VL-3B-Instruct')

    sft_model.evalset(eval_prompts)
    sft_model.update()

    sft_output = [
        {
            'instruction': item['instruction'],
            'input': item['input'],
            'output': item['output'],
            'prediction': pred
        }
        for item, pred in zip(test_data, sft_model.eval_result)
    ]

    with open(SFT_RESULT_JSON, 'w') as f:
        json.dump(sft_output, f, indent=2)

    sft_model.stop()

    # =========================
    # Step 4: 推理
    # =========================
    infer_model = build_infer_model('Qwen2.5-VL-3B-Instruct')

    infer_model.start()
    infer_model.evalset(eval_prompts)
    infer_model.eval()

    infer_output = [
        {
            'instruction': item['instruction'],
            'input': item['input'],
            'output': item['output'],
            'prediction': pred
        }
        for item, pred in zip(test_data, infer_model.eval_result)
    ]

    with open(INFER_RESULT_JSON, 'w') as f:
        json.dump(infer_output, f, indent=2)

    infer_model.stop()

    # =========================
    # Step 5: 打分（只启动一次）
    # =========================
    scorer = (
        lazyllm.TrainableModule('qwen3-14b').deploy_method(
            lazyllm.deploy.vllm,
            openai_api=True
        )
        .prompt(
            '''
Task: Accuracy Check.
Judge if the 'Prediction' logically supports.
1 = Correct & Consistent
0 = Incorrect or Irrelevant
Only output JSON: {'score': 0 or 1}
'''
        )
        .formatter(JsonFormatter())
        .start()
    )

    sft_scored = score_with_model(sft_output, scorer, SFT_SCORE_JSON)
    infer_scored = score_with_model(infer_output, scorer, INFER_SCORE_JSON)

    scorer.stop()

    # =========================
    # Step 6: 分析
    # =========================
    analyze(sft_scored, 'SFT')
    analyze(infer_scored, 'Infer')


if __name__ == '__main__':
    main()
