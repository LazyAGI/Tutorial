#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

import lazyllm
from lazyllm import deploy, finetune

random.seed(42)

SYS_PROMPT = """
You are a careful math reasoning assistant.

Solve the question step by step.
Put the final answer inside \\boxed{}.
""".strip()

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_JSONL = BASE_DIR / "gsm8k_raw.jsonl"
TRAIN_JSON = BASE_DIR / "train.json"
TEST_JSON = BASE_DIR / "test.json"
SFT_RESULT_JSON = BASE_DIR / "sft_result.json"
INFER_RESULT_JSON = BASE_DIR / "infer_result.json"

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
TRAIN_SIZE = 5000
TEST_SIZE = 1000

TEACHER_MODEL = "qwen2.5-0.5b-instruct"

SFT_ARGS = {
    "learning_rate": 1e-4,
    "cutoff_len": 2048,
    "max_samples": TRAIN_SIZE,
    "val_size": 0.1,
    "per_device_train_batch_size": 8,
    "num_train_epochs": 2.0,
}

DEPLOY_ARGS = {
    "max_model_len": 4096,
    "max_num_batched_tokens": 32768,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_boxed_answer(text):
    if not isinstance(text, str):
        return None

    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return normalize_answer(matches[-1])
    return None


def extract_gsm8k_answer(answer_text):
    if not isinstance(answer_text, str):
        return None
    if "####" not in answer_text:
        return None
    answer = answer_text.split("####")[-1].strip()
    return normalize_answer(answer)


def normalize_answer(text):
    if text is None:
        return None
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_output(raw_answer):
    final_answer = extract_gsm8k_answer(raw_answer)
    if not final_answer:
        return None
    reasoning = raw_answer.split("####")[0].strip()
    return f"{reasoning}\n\\boxed{{{final_answer}}}"


def prepare_dataset():
    train_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)
    test_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT)

    raw_records = []
    train_data = []
    test_data = []

    for item in train_ds:
        output = format_output(item.get("answer", ""))
        answer = extract_gsm8k_answer(item.get("answer", ""))
        if not output or not answer:
            continue
        record = {
            "instruction": item.get("question", "").strip(),
            "input": "",
            "output": output,
            "answer": answer,
        }
        raw_records.append(record)
        train_data.append(record)
        if len(train_data) >= TRAIN_SIZE:
            break

    for item in test_ds:
        output = format_output(item.get("answer", ""))
        answer = extract_gsm8k_answer(item.get("answer", ""))
        if not output or not answer:
            continue
        test_data.append(
            {
                "instruction": item.get("question", "").strip(),
                "input": "",
                "output": output,
                "answer": answer,
            }
        )
        if len(test_data) >= TEST_SIZE:
            break

    write_jsonl(raw_records, RAW_DATA_JSONL)
    write_json(train_data, TRAIN_JSON)
    write_json(test_data, TEST_JSON)
    print(f"已保存训练集 {len(train_data)} 条到 {TRAIN_JSON}")
    print(f"已保存测试集 {len(test_data)} 条到 {TEST_JSON}")


def build_sft_model(model_path):
    return (
        lazyllm.TrainableModule(model_path, target_path=BASE_DIR)
        .mode("finetune")
        .trainset(str(TRAIN_JSON))
        .finetune_method((finetune.llamafactory, SFT_ARGS))
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method(deploy.Vllm, **DEPLOY_ARGS)
    )


def build_infer_model(model_path):
    return (
        lazyllm.TrainableModule(model_path)
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method(deploy.Vllm, **DEPLOY_ARGS)
    )


def extract_prediction_answer(text):
    return extract_boxed_answer(text)


def has_boxed_answer(text):
    return extract_boxed_answer(text) is not None


def is_repeat(text, n=6, min_loops=3):
    if not isinstance(text, str):
        return False
    tokens = text.split()
    if len(tokens) < n * min_loops:
        return False

    for i in range(len(tokens) - n * min_loops + 1):
        base = tokens[i:i + n]
        if all(tokens[i + k * n:i + (k + 1) * n] == base for k in range(1, min_loops)):
            return True
    return False


def calc_metrics(result_data):
    total = len(result_data)
    boxed_hit = 0
    boxed_correct = 0
    correct = 0
    repeat_count = 0

    for item in result_data:
        prediction = item.get("prediction")
        answer = item.get("answer")
        pred_answer = extract_prediction_answer(prediction)

        if has_boxed_answer(prediction):
            boxed_hit += 1
        if pred_answer == normalize_answer(answer):
            boxed_correct += 1
            correct += 1
        if is_repeat(prediction):
            repeat_count += 1

    def percent(value):
        return f"{(value / total * 100):.2f}%" if total else "0.00%"

    return {
        "total_rows": total,
        "boxed_hit_count": boxed_hit,
        "boxed_hit_rate": percent(boxed_hit),
        "boxed_correct_count": boxed_correct,
        "boxed_correct_rate": percent(boxed_correct),
        "acc": percent(correct),
        "correct_count": correct,
        "correct_rate": percent(correct),
        "repeat_count": repeat_count,
        "repeat_rate": percent(repeat_count),
    }


def print_metrics(name, metrics):
    print(f"===== {name} =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main():
    prepare_dataset()

    test_data = load_json(TEST_JSON)
    eval_prompts = [item["instruction"] for item in test_data]

    sft_model = build_sft_model(TEACHER_MODEL)
    sft_model.evalset(eval_prompts)
    sft_model.update()

    sft_output = [
        {
            "instruction": item["instruction"],
            "output": item["output"],
            "answer": item["answer"],
            "prediction": pred,
        }
        for item, pred in zip(test_data, sft_model.eval_result)
    ]
    write_json(sft_output, SFT_RESULT_JSON)
    sft_model.stop()

    infer_model = build_infer_model(TEACHER_MODEL)
    infer_model.start()
    infer_model.evalset(eval_prompts)
    infer_model.eval()

    infer_output = [
        {
            "instruction": item["instruction"],
            "output": item["output"],
            "answer": item["answer"],
            "prediction": pred,
        }
        for item, pred in zip(test_data, infer_model.eval_result)
    ]
    write_json(infer_output, INFER_RESULT_JSON)
    infer_model.stop()

    print_metrics("Before SFT", calc_metrics(infer_output))
    print_metrics("After SFT", calc_metrics(sft_output))


if __name__ == "__main__":
    main()
