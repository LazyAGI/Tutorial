#!/usr/bin/env bash
set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-LLaMA-Factory}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-${LLAMAFACTORY_DIR}/data/dataset_info.json}"
DATASET_NAME="${DATASET_NAME:-ultrafeedback_h4_ppo}"
HF_DATASET_REPO="${HF_DATASET_REPO:-HuggingFaceH4/ultrafeedback_binarized}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-train_gen}"

# Llama 3.2 是 gated 模型。
# 如果你遇到 401/403 或 access denied，需要先在 HF 页面同意许可证：
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"

# 这里只是示例写法。实际 PPO 训练更推荐指向你已经训练好的奖励模型。
RM_MODEL_PATH="${RM_MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"

python - <<'PY'
import json
import os
from pathlib import Path

dataset_info_path = Path(os.environ["DATASET_INFO_PATH"])
dataset_info_path.parent.mkdir(parents=True, exist_ok=True)

if dataset_info_path.exists():
    data = json.loads(dataset_info_path.read_text(encoding="utf-8"))
else:
    data = {}

data[os.environ["DATASET_NAME"]] = {
    "hf_hub_url": os.environ["HF_DATASET_REPO"],
    "split": os.environ["HF_DATASET_SPLIT"],
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages"
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
    }
}

dataset_info_path.write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Registered dataset '{os.environ['DATASET_NAME']}' in {dataset_info_path}")
PY

cd "$LLAMAFACTORY_DIR"

llamafactory-cli train \
    --stage ppo \
    --do_train \
    --model_name_or_path "$ACTOR_MODEL_PATH" \
    --trust_remote_code True \
    --reward_model "$RM_MODEL_PATH" \
    --reward_model_type full \
    --reward_model_quantization_bit 4 \
    --dataset "$DATASET_NAME" \
    --overwrite_output_dir True \
    --max_samples 3000 \
    --template llama3 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir saves/Llama-3.2-1B-Instruct/ppo \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --learning_rate 1e-6 \
    --num_train_epochs 1.0 \
    --bf16 True \
    --cutoff_len 1024 \
    --top_k 0 \
    --top_p 0.9 \
    --ppo_score_norm True \
    --ppo_whiten_rewards True \
    --logging_steps 10 \
    --save_steps 20 \
    --save_total_limit 3 \
    --plot_loss True \
    --report_to none
