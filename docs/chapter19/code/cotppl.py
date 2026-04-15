#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path
from datasets import load_dataset

import lazyllm
from lazyllm import deploy, finetune, launchers
from lazyllm.tools.data.pipelines.cot_pipelines import build_cot_pipeline

random.seed(42)

# =========================
# Prompt（BBH版本）
# =========================
SYS_PROMPT = '''
You are a careful reasoning assistant.

Solve the following multiple-choice question step by step.

Requirements:
- Explain your reasoning clearly.
- Only one option is correct.
- Do not repeat all the options again.
- At the end output the answer strictly in the format:

#### <LETTER>

Where <LETTER> is one of A, B, C, D, E, or F.
'''

USER_PROMPT = SYS_PROMPT + '\nQuestion:\n'

# =========================
# 路径
# =========================
BASE_DIR = Path(__file__).resolve().parent

BBH_JSONL = BASE_DIR / 'bbh.jsonl'
TRAIN_JSON = BASE_DIR / 'train.json'
TEST_JSON = BASE_DIR / 'test.json'

SFT_RESULT_JSON = BASE_DIR / 'sft_result.json'
INFER_RESULT_JSON = BASE_DIR / 'infer_result.json'


# =========================
# IO
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


# =========================
# Step 1: 下载 BBH
# =========================
def prepare_dataset():
    tasks = [
        'boolean_expressions',
        'causal_judgement',
        'date_understanding',
        'disambiguation_qa',
        'formal_fallacies',
        'geometric_shapes',
        'logical_deduction_five_objects',
        'logical_deduction_three_objects',
        'multistep_arithmetic_two',
        'object_counting',
        'reasoning_about_colored_objects',
        'temporal_sequences',
        'tracking_shuffled_objects_three_objects',
        'web_of_lies',
        'word_sorting',
    ]
    print('下载 lukaemon/bbh 数据集 ing...')

    with open(BBH_JSONL, 'w', encoding='utf-8') as f:
        for task in tasks:
            dataset = load_dataset('lukaemon/bbh', task)
            for item in dataset['test']:
                if item['target'] in ['(A)',
                                      '(B)',
                                      '(C)',
                                      '(D)',
                                      '(E)',
                                      '(F)']:
                    row = {
                        'task': task,
                        'question': item['input'],
                        'reference': item['target'][1],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print('BBH 下载成功！')


# =========================
# Step 2: PPL生成CoT数据
# =========================
def run_cot_pipeline():
    data = load_json(BBH_JSONL)

    teacher = lazyllm.TrainableModule('qwen2.5-32b-instruct').deploy_method(
        lazyllm.deploy.vllm, max_model_len=1024, max_num_batched_tokens=32768
    )

    ppl = build_cot_pipeline(
        input_key='question',
        reference_key='reference',
        cot_key='cot_answer',
        extracted_key='llm_extracted',
        verify_key='is_equal',
        model=teacher,
        use_self_consistency=True,
        num_samples=3,
        user_prompt=USER_PROMPT,
        enable_verify=True,
        hash_answer=True,
        boxed_answer=False,
    )

    result = ppl(data)

    split_idx = int(len(result) * 0.8)
    train_data = [
        {'instruction': x['instruction'], 'output': x['output']}
        for x in result[:split_idx]
    ]
    test_data = [
        {'instruction': x['instruction'], 'output': x['output']}
        for x in result[split_idx:]
    ]

    write_json(train_data, TRAIN_JSON)
    write_json(test_data, TEST_JSON)

    print('CoT + split done')


# =========================
# Step 3: SFT
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
                    'cutoff_len': 1024,
                    'max_samples': 10000,
                    'val_size': 0.1,
                    'per_device_train_batch_size': 24,
                    'num_train_epochs': 3.0,
                    'launcher': launchers.empty(ngpus=1),
                },
            )
        )
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method(
            deploy.Vllm,
            max_model_len=1024,
            max_num_batched_tokens=98304,
        )
    )


# =========================
# Step 4: Infer
# =========================
def build_infer_model(model_path):
    return (
        lazyllm.TrainableModule(model_path)
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method(
            deploy.Vllm,
            max_model_len=1024,
            max_num_batched_tokens=98304,
        )
    )


# =========================
# Step 5: 评分
# =========================
def score_detailed(path, data):
    stats = {
        'total': 0,
        'standard_format_count': 0,
        'standard_correct': 0,
        'wrong_format_count': 0,
        'wrong_format_correct': 0,
        'parse_fail': 0,
    }

    def extract_hash(text):
        if not text or '#' not in text:
            return None
        ans = text.strip().split('#')[-1].strip()
        if '(' in ans and ')' in ans:
            ans = ans[ans.find('(') + 1: ans.find(')')]
        return ans

    def extract_last_option(text):
        if not text:
            return None
        matches = re.findall(r'\b([A-F])\b|\(([A-F])\)', text)
        if matches:
            last_match = matches[-1]
            return last_match[0] if last_match[0] else last_match[1]
        return None

    def get_answer(text):
        ans = extract_hash(text)
        if ans:
            return ans, 'standard'
        ans = extract_last_option(text)
        if ans:
            return ans, 'wrong_format'
        return None, 'fail'

    for obj in data:
        pred_ans, pred_type = get_answer(obj.get('prediction'))
        gt_ans, _ = get_answer(obj.get('output'))

        stats['total'] += 1
        if pred_type == 'standard':
            stats['standard_format_count'] += 1
            if pred_ans == gt_ans:
                stats['standard_correct'] += 1
        elif pred_type == 'wrong_format':
            stats['wrong_format_count'] += 1
            if pred_ans == gt_ans:
                stats['wrong_format_correct'] += 1
        else:
            stats['parse_fail'] += 1

    total_correct = stats['standard_correct'] + stats['wrong_format_correct']

    print(f'File: {Path(path).name}')
    print(f"  Total Samples: {stats['total']}")
    print(f'  Total Samples: {stats["total"]}')
    print('-' * 25)
    print('  [标准格式 (#)]')
    print(f'    数量: {stats["standard_format_count"]}')
    print(f'    其中正确: {stats["standard_correct"]}')
    print('  [错误答案格式 (Regex)]')
    print(f'    数量: {stats["wrong_format_count"]}')
    print(f'    其中正确: {stats["wrong_format_correct"]}')
    print(f'  [无法解析]: {stats["parse_fail"]}')
    print('-' * 25)
    print(f'  总正确数 (All): {total_correct}')
    acc = total_correct / stats['total'] if stats['total'] else 0
    print(f'  总正确率 (Acc): {acc:.4%}')
    print('=' * 40)


# =========================
# 主流程
# =========================
def main():
    prepare_dataset()
    run_cot_pipeline()

    test_data = load_json(TEST_JSON)
    eval_prompts = [x['instruction'] for x in test_data]

    # ===== SFT =====
    sft_model = build_sft_model('qwen2.5-0.5b-instruct')
    sft_model.evalset(eval_prompts)
    sft_model.update()

    sft_output = []
    for item, pred in zip(test_data, sft_model.eval_result):
        sft_output.append(
            {
                'instruction': item['instruction'],
                'output': item['output'],
                'prediction': pred,
            }
        )

    write_json(sft_output, SFT_RESULT_JSON)
    sft_model.stop()

    # ===== Infer =====
    infer_model = build_infer_model('qwen2.5-0.5b-instruct')
    infer_model.start()
    infer_model.evalset(eval_prompts)
    infer_model.eval()

    infer_output = []
    for item, pred in zip(test_data, infer_model.eval_result):
        infer_output.append(
            {
                'instruction': item['instruction'],
                'output': item['output'],
                'prediction': pred,
            }
        )

    write_json(infer_output, INFER_RESULT_JSON)
    infer_model.stop()

    # ===== Score =====
    score_detailed(SFT_RESULT_JSON, sft_output)
    score_detailed(INFER_RESULT_JSON, infer_output)


if __name__ == '__main__':
    main()
