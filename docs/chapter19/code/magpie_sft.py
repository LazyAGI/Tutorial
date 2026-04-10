#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path

import regex
from datasets import load_dataset

import lazyllm
from lazyllm import deploy, finetune

random.seed(42)

SYS_PROMPT = '''
You are a careful reasoning assistant.

Solve the problem step by step.

Requirements:
- Show concise but complete reasoning.
- Put the final answer in the format: \\boxed{ANSWER}
- The final line should contain only the boxed answer.
'''.strip()

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_JSONL = BASE_DIR / 'magpie_reasoning_3k.jsonl'
TRAIN_JSON = BASE_DIR / 'train.json'
TEST_JSON = BASE_DIR / 'test.json'
SFT_RESULT_JSON = BASE_DIR / 'sft_result.json'
INFER_RESULT_JSON = BASE_DIR / 'infer_result.json'

DATASET_NAME = (
    'Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B'
)
DATASET_SPLIT = 'train[:3000]'
TRAIN_SIZE = 2000
TEST_SIZE = 1000

BASELINE_MODEL = 'qwen2.5-0.5b-instruct'

LONG_COT_FINETUNE_ARGS = {
    'learning_rate': 8e-5,
    'cutoff_len': 2048,
    'max_samples': 3000,
    'val_size': 0.1,
    'per_device_train_batch_size': 8,
    'num_train_epochs': 3.0,
}

LONG_COT_DEPLOY_ARGS = {
    'max_model_len': 4096,
    'max_num_batched_tokens': 32768,
}


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


def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def prepare_dataset():
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    ds.to_json(str(RAW_DATA_JSONL), force_ascii=False)
    print(f'下载完成，已保存 {len(ds)} 条数据到 {RAW_DATA_JSONL}')


def transform_item(item):
    return {
        'instruction': item.get('instruction', ''),
        'input': '',
        'output': item.get('response', ''),
    }


def split_dataset():
    data = load_json(RAW_DATA_JSONL)
    print(f'总条数: {len(data)}')

    train_data = [transform_item(item) for item in data[:TRAIN_SIZE]]
    test_data = [transform_item(item) for item in data[-TEST_SIZE:]]

    write_json(train_data, TRAIN_JSON)
    write_json(test_data, TEST_JSON)
    print(
        f'训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条'
    )


def build_sft_model(model_path):
    return (
        lazyllm.TrainableModule(model_path, target_path=BASE_DIR)
        .mode('finetune')
        .trainset(str(TRAIN_JSON))
        .finetune_method((finetune.llamafactory, LONG_COT_FINETUNE_ARGS))
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method(deploy.Vllm, **LONG_COT_DEPLOY_ARGS)
    )


def build_infer_model(model_path):
    return (
        lazyllm.TrainableModule(model_path)
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method(deploy.Vllm, **LONG_COT_DEPLOY_ARGS)
    )


def extract_box_value(text):
    if not isinstance(text, str):
        return None
    pattern = (
        r'\\boxed\{(?P<content>(?:[^{}]+|\{(?&content)\})*)\}'
    )
    matches = regex.findall(pattern, text)
    return matches[-1].strip() if matches else None


def has_box(text):
    return isinstance(text, str) and bool(re.search(r'\\boxed\{', text))


def is_repeat(text, n=6, min_loops=3):
    if not isinstance(text, str):
        return False
    tokens = text.split()
    if len(tokens) < n * min_loops:
        return False

    for i in range(len(tokens) - n * min_loops + 1):
        base = tokens[i:i + n]
        if all(tokens[i + k * n:i + (k + 1) * n] == base
               for k in range(1, min_loops)):
            return True
    return False


def calc_metrics(result_data):
    total = len(result_data)
    correct = 0
    no_box = 0
    repeat_count = 0

    for item in result_data:
        prediction = item.get('prediction')
        output = item.get('output')

        if not has_box(prediction):
            no_box += 1
        if extract_box_value(prediction) == extract_box_value(output):
            correct += 1
        if is_repeat(prediction):
            repeat_count += 1

    def percent(value):
        return f'{(value / total * 100):.2f}%' if total else '0.00%'

    return {
        'total_rows': total,
        'acc': percent(correct),
        'correct_count': correct,
        'no_box_count': no_box,
        'no_box_rate': percent(no_box),
        'repeat_count': repeat_count,
        'repeat_rate': percent(repeat_count),
    }


def print_metrics(name, metrics):
    print(f'===== {name} =====')
    for key, value in metrics.items():
        print(f'{key}: {value}')


def main():
    prepare_dataset()
    split_dataset()

    test_data = load_json(TEST_JSON)
    eval_prompts = [item['instruction'] for item in test_data]

    sft_model = build_sft_model(BASELINE_MODEL)
    sft_model.evalset(eval_prompts)
    sft_model.update()

    sft_output = [
        {
            'instruction': item['instruction'],
            'output': item['output'],
            'prediction': pred,
        }
        for item, pred in zip(test_data, sft_model.eval_result)
    ]
    write_json(sft_output, SFT_RESULT_JSON)
    sft_model.stop()

    infer_model = build_infer_model(BASELINE_MODEL)
    infer_model.start()
    infer_model.evalset(eval_prompts)
    infer_model.eval()

    infer_output = [
        {
            'instruction': item['instruction'],
            'output': item['output'],
            'prediction': pred,
        }
        for item, pred in zip(test_data, infer_model.eval_result)
    ]
    write_json(infer_output, INFER_RESULT_JSON)
    infer_model.stop()

    print_metrics('Before SFT', calc_metrics(infer_output))
    print_metrics('After SFT', calc_metrics(sft_output))


if __name__ == '__main__':
    main()
