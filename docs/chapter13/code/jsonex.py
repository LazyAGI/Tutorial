import json
import re
import ast
from pathlib import Path

import lazyllm
from lazyllm import finetune, deploy
from datasets import load_dataset

base_dir = Path(__file__).parent
# ======================
# 全局配置
# ======================
SYS_PROMPT = """
You are a helpful assistant that outputs **only a JSON dictionary**.
Follow these rules strictly:

1. The dictionary must have exactly two keys:
   - "name": a string
   - "arguments": a dictionary or string
2. Do NOT include any extra text
"""

MODEL_NAME = "qwen2.5-0.5B-instruct"

pattern = re.compile(r"\{.*\}", re.S)


# ======================
# 工具函数
# ======================
def extract_obj(text):
    match = pattern.search(text)
    return match.group() if match else None


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ======================
# 数据准备
# ======================
def prepare_data(base_dir):
    print("📥 Loading dataset...")

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    items = []

    for i, item in enumerate(ds):
        if i >= 5000:
            break

        answers = json.loads(item["answers"]) if isinstance(item["answers"], str) else item["answers"]
        output = answers[0] if answers else ""

        if isinstance(output, str):
            output = output.replace("\n", " ")

        output = json.dumps(output, ensure_ascii=False)

        items.append({
            "instruction": item["query"],
            "output": output
        })

    train_path = base_dir / "train.json"
    test_path = base_dir / "test.json"

    print("💾 Saving dataset...")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(items[:4000], f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(items[4000:5000], f, ensure_ascii=False, indent=2)

    return train_path, test_path


# ======================
# 推理
# ======================
def run_inference(model_path, eval_prompts, output_path):
    print("🚀 Running base inference...")

    model = (
        lazyllm.TrainableModule(model_path)
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method((deploy.Vllm, {"max_num_seqs": 128}))
    )

    model.evalset(eval_prompts)
    model.start()
    model.eval()

    write_jsonl([{"result": r} for r in model.eval_result], output_path)
    model.stop()
    return output_path


# ======================
# SFT
# ======================
def run_sft(model_path, train_path, eval_prompts, output_path):
    print("🔥 Running SFT...")

    model = (
        lazyllm.TrainableModule(model_path, target_path=base_dir)
        .mode("finetune")
        .trainset(str(train_path))
        .finetune_method((finetune.llamafactory, {
            "learning_rate": 1e-4,
            "cutoff_len": 1024,
            "max_samples": 5000,
            "val_size": 0.1,
            "per_device_train_batch_size": 24,
            "num_train_epochs": 3.0,
        }))
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method((deploy.Vllm, {"max_num_seqs": 128}))
    )

    model.evalset(eval_prompts)
    model.update()

    write_jsonl([{"result": r} for r in model.eval_result], output_path)
    model.stop()
    return output_path


# ======================
# 评测
# ======================
def evaluate(file_path):
    total = 0
    json_ok = 0
    dict_ok = 0
    bracket_ok = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            data = json.loads(line)
            result = data.get("result", "")

            if "{" in result and "}" in result:
                bracket_ok += 1
                result = extract_obj(result)

            try:
                json.loads(result)
                json_ok += 1
            except:
                try:
                    ast.literal_eval(result)
                    dict_ok += 1
                except:
                    pass

    print(f"\n📊 Evaluation: {file_path}")
    print(f"Total: {total}")
    print(f"Has {{}}: {bracket_ok}")
    print(f"JSON OK: {json_ok}")
    print(f"Dict OK: {dict_ok}")


# ======================
# 主流程
# ======================
def main():

    print(f"📁 Working dir: {base_dir}")

    # 1️⃣ 数据准备
    train_path, test_path = prepare_data(base_dir)

    # 2️⃣ 构造 eval prompts
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    eval_prompts = [item["instruction"] for item in test_data]

    # 3️⃣ base inference
    base_out = base_dir / "base.jsonl"
    run_inference(MODEL_NAME, eval_prompts, base_out)

    # 4️⃣ SFT
    sft_out = base_dir / "sft.jsonl"
    run_sft(MODEL_NAME, train_path, eval_prompts, sft_out)

    # 5️⃣ 评测
    evaluate(base_out)
    evaluate(sft_out)


if __name__ == "__main__":
    main()