#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import random
from collections import Counter
from huggingface_hub import snapshot_download  # NID002

import lazyllm
from lazyllm import finetune, deploy
from lazyllm.components.formatter import JsonFormatter
from lazyllm.tools.data.pipelines.pdf_pipelines import build_pdf2qa_pipeline

random.seed(32)

# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / 'pdfs'
IMAGE_OUTPUT_FOLDER = BASE_DIR / 'pdfs/images'
TRAIN_JSON = BASE_DIR / 'train.json'
TEST_JSON = BASE_DIR / 'test.json'
SFT_RESULT_JSON = BASE_DIR / 'sft_result.json'
SFT_SCORE_JSON = BASE_DIR / 'sft_score_result.json'
INFER_RESULT_JSON = BASE_DIR / 'infer_result.json'
INFER_SCORE_JSON = BASE_DIR / 'infer_score_result.json'

snapshot_download(
    repo_id='scui382/pdf-ppl',
    repo_type='dataset',
    local_dir=BASE_DIR,  # 下载到本地的路径
    allow_patterns='*.pdf'  # 只下载 PDF 文件（可选）
)


# =========================
# Step 1: PDF → QA 数据
# =========================
def generate_qa_from_pdfs(model):
    data = [{'pdf_path': str(p)} for p in PDF_FOLDER.glob('*.pdf')]
    print(data)
    print(f'找到 {len(data)} 个 PDF 文件')

    generator_prompt = '''
你是一个用于构建训练数据的助手，需要基于给定的图像或文本内容，生成一个高质量的中文问答对（QA），用于监督微调（SFT）。

【任务要求】
仅基于输入内容提取关键信息，生成1个问答对
问题应关注概念、原理、方法或作用，不要关注具体数值或细节数据
问题必须清晰、独立、自然，不能依赖“上下文”才能理解
答案应准确、完整，直接回答问题

【严格约束】
必须使用中文输出
不要出现以下表达：
“根据上下文”
“根据给定内容”
“文中提到”
“作者认为”
“本段/本章”
问题中不能包含模糊指代（如“该方法”“这个模型”），必须明确具体对象
不要提问精确数字（如年份、数量、比例等）

【输出格式要求】
只能输出 JSON，不能包含任何额外说明或解释
'''

    ppl = build_pdf2qa_pipeline(
        model=model,
        mineru_api='http://10.119.30.80:20234',
        image_output_folder=IMAGE_OUTPUT_FOLDER,
        chunk_key='chunk',
        image_key='image_path',
        qa_user_prompt=generator_prompt,
        max_chunk_chars=1000,
        chat_format=False,
        context_key='chunk'
    )

    results = ppl(data)

    # 打乱并划分训练/测试
    random.shuffle(results)
    split_idx = int(len(results) * 0.2)
    test_data = results[:split_idx]
    train_data = results[split_idx:]

    # 写出 JSON
    with open(TRAIN_JSON, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(TEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(
        f'生成数据完成: Total={len(results)}, '
        f'Train={len(train_data)}, Test={len(test_data)}'
    )
    return train_data, test_data


# =========================
# Step 2: 构建 SFT / 推理 模型
# =========================
def build_sft_model(model_path, train_set_path):
    model = (
        lazyllm.TrainableModule(model_path, target_path=BASE_DIR)
        .mode('finetune')
        .trainset(str(train_set_path))
        .finetune_method(
            (
                finetune.llamafactory,
                {
                    'learning_rate': 5e-5,
                    'cutoff_len': 1024,
                    'max_samples': 10000,
                    'val_size': 0.1,
                    'per_device_train_batch_size': 4,
                    'gradient_accumulation_steps': 4,
                    'num_train_epochs': 1.0,
                },
            )
        )
        .prompt(dict(system='', drop_builtin_system=True))
        .deploy_method(deploy.Vllm)
    )
    return model


def build_infer_model(model_path):
    model = lazyllm.TrainableModule(model_path).prompt(
        dict(system='', drop_builtin_system=True)
    )
    return model


# =========================
# Step 3: 打分
# =========================
def score_predictions_with_model(data, scorer, result_file):
    '''
    使用已部署的打分模型对数据评分
    '''
    scored = []
    for item in data:
        instruction = item.get('instruction', '')
        answer = item.get('output', '')
        prediction = item.get('prediction', '')
        input_text = item.get('input', '')
        prompt = f'''
问题：
{instruction}

上下文:

{input_text}

标准答案：
{answer}

模型预测：
{prediction}
'''
        result = scorer(prompt)
        try:
            score = float(result.get('score', 0))
        except (TypeError, ValueError, AttributeError):
            score = 0.0
        scored.append({
            'instruction': instruction,
            'output': answer,
            'prediction': prediction,
            'score': score,
            'input': input_text
        })

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f'打分完成: {result_file}')
    return scored


# =========================
# Step 4: 分析统计
# =========================
def analyze_scores(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scores = [item.get('score', 0) for item in data]
    total = len(scores)
    avg = sum(scores) / total if total else 0
    counter = Counter(scores)
    print(f'\n📊 File: {path}')
    print(f'Total: {total}')
    print(f'Average score: {avg:.4f}')
    print('Distribution:')
    for k in sorted(counter.keys()):
        print(f'  {k}: {counter[k] / total:.2%}')


# =========================
# 主流程：自动执行全部
# =========================
def main():
    model_path = 'Qwen2.5-VL-32B-Instruct'
    sft_base_model = 'qwen1.5-0.5b-chat'
    score_model = 'qwen2.5-14b-instruct'

    # =========================
    # Step 0: 部署生成模型（用于 PDF → QA）
    # =========================
    generate_model = lazyllm.TrainableModule(model_path)

    # Step 1: PDF → QA → train/test JSON
    train_data, test_data = generate_qa_from_pdfs(generate_model)
    generate_model.stop()  # 如果只想生成一次，可以停掉

    # =========================
    # Step 2: SFT 微调 + 保存预测
    # =========================
    sft_model = build_sft_model(sft_base_model, TRAIN_JSON)
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    eval_prompts = [
        item['instruction'] + '上下文：{}'.format(item['input'])
        for item in test_data
    ]
    sft_model.evalset(eval_prompts)
    sft_model.update()

    sft_output = []
    for item, pred in zip(test_data, sft_model.eval_result):
        sft_output.append({
            'instruction': item['instruction'],
            'output': item['output'],
            'prediction': pred,
            'input': item.get('input', '')
        })
    with open(SFT_RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(sft_output, f, ensure_ascii=False, indent=2)
    sft_model.stop()

    # =========================
    # Step 3: 推理模型预测
    # =========================
    infer_model = build_infer_model(sft_base_model)
    infer_model.start()
    infer_model.evalset(eval_prompts)
    infer_model.eval()

    infer_output = []
    for item, pred in zip(test_data, infer_model.eval_result):
        infer_output.append({
            'instruction': item['instruction'],
            'output': item['output'],
            'prediction': pred,
            'input': item.get('input', '')
        })
    with open(INFER_RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(infer_output, f, ensure_ascii=False, indent=2)
    infer_model.stop()

    # =========================
    # Step 4: 部署打分模型（只部署一次）
    # =========================
    scorer_model = (
        lazyllm.TrainableModule(score_model)
        .prompt('''
请判断“模型预测”和“标准答案”是否相符。

评分标准：
1 = 相符
0.5 = 部分正确
0 = 完全错误

输出格式必须严格为 JSON：
{
    'score': 0 或 0.5 或 1
}

不要输出解释。
不要输出额外内容。
''')
        .formatter(JsonFormatter())
        .start()
    )

    # Step 4a: 打分 SFT
    score_predictions_with_model(sft_output, scorer_model, SFT_SCORE_JSON)

    # Step 4b: 打分推理
    score_predictions_with_model(infer_output, scorer_model, INFER_SCORE_JSON)

    # 停止打分模型
    scorer_model.stop()

    # =========================
    # Step 5: 分析统计
    # =========================
    analyze_scores(SFT_SCORE_JSON)
    analyze_scores(INFER_SCORE_JSON)


if __name__ == '__main__':
    main()
