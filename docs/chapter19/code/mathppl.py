#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path
import math_verify
from datasets import load_dataset

import lazyllm
from lazyllm import deploy, finetune
from lazyllm.tools.data.pipelines.math_pipelines import build_math_cot_pipeline

random.seed(42)

SYS_PROMPT = '''
Solve the math question;
Cover the final answer with \\boxed{}
'''

COT_USER_PROMPT = '''
Generate the answer step by step.
And cover your final answer with \\boxed{}
'''

# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_JSONL = BASE_DIR / 'gsm8k_train.jsonl'
TEST_JSONL = BASE_DIR / 'gsm8k_test.jsonl'
TRAIN_JSON = BASE_DIR / 'train.json'

SFT_RESULT_JSON = BASE_DIR / 'sft_result.json'
INFER_RESULT_JSON = BASE_DIR / 'infer_result.json'

SFT_SCORE_JSON = BASE_DIR / 'sft_score.json'
INFER_SCORE_JSON = BASE_DIR / 'infer_score.json'


# =========================
# 通用 IO
# =========================
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            item = dict(row)
            item['reference'] = row['answer'].split('##')[-1].strip()
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# =========================
# Step 1: 下载 + 切分数据
# =========================
def prepare_dataset():
    print('Loading GSM8K dataset...')
    dataset = load_dataset('openai/gsm8k', 'main')

    data = dataset['train'].select(range(5000))
    train = data.select(range(4000))
    test = data.select(range(4000, 5000))

    save_jsonl(train, TRAIN_JSONL)
    save_jsonl(test, TEST_JSONL)


# =========================
# Step 2: CoT 增强训练集
# =========================
def build_teacher_model():
    return lazyllm.TrainableModule(
        'qwen2.5-32b-instruct'
    ).deploy_method(lazyllm.deploy.vllm)


def run_math_cot():
    print('Generating math CoT training data...')
    data = load_json(TRAIN_JSONL)

    teacher_model = build_teacher_model()
    pipeline = build_math_cot_pipeline(
        question_key='question',
        reference_key='reference',
        answer_key='answer',
        extracted_key='math_answer',
        verify_key='is_equal',
        cot_user_prompt=COT_USER_PROMPT,
        model=teacher_model,
        num_samples=3,
    )

    train_data = pipeline(data)
    write_json(train_data, TRAIN_JSON)


# =========================
# Step 3: 构建 SFT 模型
# =========================
def build_sft_model(model_path):
    return (
        lazyllm.TrainableModule(model_path, target_path=BASE_DIR)
        .mode('finetune')
        .trainset(str(TRAIN_JSON))
        .finetune_method(
            (
                finetune.llamafactory,
                {
                    'learning_rate': 1e-4,
                    'cutoff_len': 512,
                    'max_samples': 10000,
                    'val_size': 0.1,
                    'per_device_train_batch_size': 24,
                    'num_train_epochs': 2.0,
                },
            )
        )
        .prompt(
            dict(
                system=SYS_PROMPT,
                drop_builtin_system=True,
            )
        )
        .deploy_method(deploy.Vllm)
    )


# =========================
# Step 4: 构建推理模型
# =========================
def build_infer_model(model_path):
    return (
        lazyllm.TrainableModule(model_path)
        .prompt(
            dict(
                system=SYS_PROMPT,
                drop_builtin_system=True,
            )
        )
        .deploy_method(deploy.Vllm)
    )


# =========================
# Step 5: 评估
# =========================
def extract_boxed(text):
    match = re.search(r'\\boxed\{(.*?)\}', text or '')
    if match:
        return match.group(1).strip()
    return None


def extract_gt(text):
    return (text or '').split('#')[-1].strip()


def score_predictions(data, save_path):
    total = 0
    no_boxed = 0
    parsed = 0
    correct = 0
    output = []

    for item in data:
        prediction = item.get('prediction', '')
        answer = item.get('output', '')

        total += 1
        pred = extract_boxed(prediction)
        gt = extract_gt(answer)

        scored = dict(item)
        scored['parsed_prediction'] = pred
        scored['reference_answer'] = gt

        if pred is None or gt is None:
            no_boxed += 1
            scored['score'] = 0
            output.append(scored)
            continue

        parsed += 1
        if str(pred).strip() == str(gt).strip():
            scored['score'] = 1
        else:
            parsed_real = math_verify.parse(str(gt))
            parsed_llm = math_verify.parse(str(pred))
            mv = math_verify.verify(parsed_real, parsed_llm)
            scored['score'] = 1 if mv else 0

        correct += scored['score']
        output.append(scored)

    write_json(output, save_path)
    return {
        'data': output,
        'total': total,
        'no_boxed': no_boxed,
        'parsed': parsed,
        'correct': correct,
        'acc_total': correct / total if total else 0,
        'acc_parsed': correct / parsed if parsed else 0,
    }


def analyze(result, name):
    print(f'\n====== {name} ======')
    print('Total:', result['total'])
    print('No Boxed:', result['no_boxed'])
    print('Parsed:', result['parsed'])
    print('Correct:', result['correct'])
    print('Accuracy (overall):', result['acc_total'])
    print('Accuracy (parsed):', result['acc_parsed'])


# =========================
# 主流程
# =========================
def main():
    prepare_dataset()
    run_math_cot()

    test_data = load_json(TEST_JSONL)
    eval_prompts = [item['question'] for item in test_data]

    # =========================
    # Step 3: SFT
    # =========================
    sft_model = build_sft_model('qwen1.5-0.5b-chat')
    sft_model.evalset(eval_prompts)
    sft_model.update()

    sft_output = []
    for item, pred in zip(test_data, sft_model.eval_result):
        sft_output.append(
            {
                'instruction': item['question'],
                'output': item['answer'],
                'prediction': pred,
            }
        )

    write_json(sft_output, SFT_RESULT_JSON)
    sft_model.stop()

    # =========================
    # Step 4: Infer
    # =========================
    infer_model = build_infer_model('qwen1.5-0.5b-chat')
    infer_model.start()
    infer_model.evalset(eval_prompts)
    infer_model.eval()

    infer_output = []
    for item, pred in zip(test_data, infer_model.eval_result):
        infer_output.append(
            {
                'instruction': item['question'],
                'output': item['answer'],
                'prediction': pred,
            }
        )

    write_json(infer_output, INFER_RESULT_JSON)
    infer_model.stop()

    # =========================
    # Step 5: 评估
    # =========================
    sft_output = load_json(SFT_RESULT_JSON)
    infer_output = load_json(INFER_RESULT_JSON)
    sft_result = score_predictions(sft_output, SFT_SCORE_JSON)
    infer_result = score_predictions(infer_output, INFER_SCORE_JSON)

    analyze(sft_result, 'SFT')
    analyze(infer_result, 'Infer')


if __name__ == '__main__':
    main()
