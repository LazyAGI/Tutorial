#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$BASE_DIR/data"
MODEL_DIR="$BASE_DIR/models"
OUTPUT_DIR="$BASE_DIR/output"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$DATA_DIR" "$MODEL_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# 日志文件
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log_error() {
    log "[ERROR] $1"
}

log_info() {
    log "[INFO] $1"
}

log_step() {
    log "[STEP] $1"
}

safe_exit() {
    local code="$1"
    if [ "$code" -ne 0 ]; then
        log_error "脚本异常退出，退出码: $code"
    else
        log_info "脚本正常完成"
    fi
    exit "$code"
}

LAZYLLM_PATH="/path/to/your/lazyllm"
PIPELINE_MODEL="/path/to/pipeline/model"
SFT_MODEL="/path/to/sft/base/model"

if [ ! -d "$LAZYLLM_PATH" ]; then
    log_error "LAZYLLM_PATH 不存在: $LAZYLLM_PATH"
    log "请修改脚本中的 LAZYLLM_PATH 配置"
    safe_exit 1
fi

log "=========================================="
log "一键代码SFT训练脚本"
log "=========================================="
log ""
log_info "配置信息:"
log "  - 基础目录: $BASE_DIR"
log "  - 数据目录: $DATA_DIR"
log "  - 模型目录: $MODEL_DIR"
log "  - 输出目录: $OUTPUT_DIR"
log "  - 日志文件: $LOG_FILE"
log ""

# ============ 步骤1: 下载数据 ============
log_step "[1/5] 下载 tiny-codes 数据集..."

python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import os
from datasets import load_dataset

DATA_DIR = "$DATA_DIR"
train_path = os.path.join(DATA_DIR, "train_python.json")
eval_path = os.path.join(DATA_DIR, "eval_python.json")

if os.path.exists(train_path) and os.path.exists(eval_path):
    print("  数据已存在，跳过下载")
    exit(0)

print("  正在从 Hugging Face 加载数据集...")
ds = load_dataset('nampdn-ai/tiny-codes', split='train', streaming=True)
python_data = []
target_count = 6000

for entry in ds:
    if entry.get("programming_language", "").lower() == "python":
        prompt = entry.get("prompt", "").strip()
        response = entry.get("response", "").strip()
        if len(prompt) > 5 and len(response) > 10:
            python_data.append({"instruction": prompt, "input": "", "output": response})
        if len(python_data) % 500 == 0:
            print(f"    已收集 {len(python_data)} 条...")
        if len(python_data) >= target_count:
            break

with open(train_path, 'w') as f:
    json.dump(python_data[:5000], f, indent=2)
with open(eval_path, 'w') as f:
    json.dump(python_data[5000:6000], f, indent=2)

print(f"  训练集: {len(python_data[:5000])} 条")
print(f"  验证集: {len(python_data[5000:6000])} 条")
EOF

if [ $? -ne 0 ]; then
    log_error "步骤1失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤2: 数据处理Pipeline ============
log_step "[2/5] 运行数据增强 pipeline..."

python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys

LAZYLLM_PATH = "$LAZYLLM_PATH"
sys.path.insert(0, LAZYLLM_PATH)
sys.path.insert(0, os.path.dirname(LAZYLLM_PATH))

import lazyllm
from lazyllm.tools.data.pipelines.codegen_pipelines import build_codegen_pipeline

DATA_DIR = "$DATA_DIR"
input_file = os.path.join(DATA_DIR, "train_python.json")
output_file = os.path.join(DATA_DIR, "codegen.json")

if os.path.exists(output_file):
    print("  Pipeline 输出已存在，跳过处理")
    exit(0)

with open(input_file, 'r') as f:
    data = json.load(f)

print(f"  加载数据: {len(data)} 条")

pipeline_model_path = "$PIPELINE_MODEL"
model = lazyllm.TrainableModule(pipeline_model_path)
model.start()

ppl = build_codegen_pipeline(model=model, input_key='messages', min_score=8, max_score=10)

formatted_data = []
for item in data[:100]:
    messages = [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": item['instruction']}
    ]
    result = ppl([{"messages": messages, "metadata": {}}])
    if result:
        formatted_data.append(result[0])

with open(output_file, 'w') as f:
    json.dump(formatted_data, f, indent=2)

model.stop()
print(f"  生成数据: {len(formatted_data)} 条")
EOF

if [ $? -ne 0 ]; then
    log_error "步骤2失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤3: SFT训练 ============
log_step "[3/5] 开始 SFT 训练..."

python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys

local_path = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if local_path not in sys.path:
    sys.path.insert(0, local_path)

import lazyllm
from lazyllm import finetune, deploy, launchers

DATA_DIR = "$DATA_DIR"
MODEL_DIR = "$MODEL_DIR"
train_file = os.path.join(DATA_DIR, "codegen.json")
checkpoint_dir = os.path.join(MODEL_DIR, "checkpoint")

if os.path.exists(checkpoint_dir):
    print("  模型已存在，跳过训练")
    exit(0)

sft_model_path = "$SFT_MODEL"
model = lazyllm.TrainableModule(sft_model_path, target_path=checkpoint_dir)\
    .mode('finetune')\
    .trainset(train_file)\
    .finetune_method((finetune.llamafactory, {
        'learning_rate': 1e-4,
        'cutoff_len': 4096,
        'max_samples': 5000,
        'val_size': 0.02,
        'optim': 'adamw_torch_fused',
        'bf16': True,
        'fp16': False,
        'per_device_train_batch_size': 8,
        'gradient_accumulation_steps': 4,
        'num_train_epochs': 2.0,
        'template': 'qwen',
        'stage': 'sft',
        'save_steps': 100,
        'save_total_limit': 2,
        'launcher': launchers.sco(ngpus=1, partition='a800'),
    }))

model.update()
print(f"  模型保存: {checkpoint_dir}")
EOF

if [ $? -ne 0 ]; then
    log_error "步骤3失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤4: 评测集推理 ============
log_step "[4/5] 运行评测集推理..."

python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys

local_path = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if local_path not in sys.path:
    sys.path.insert(0, local_path)

import lazyllm
from lazyllm import deploy

DATA_DIR = "$DATA_DIR"
OUTPUT_DIR = "$OUTPUT_DIR"
MODEL_DIR = "$MODEL_DIR"

eval_file = os.path.join(DATA_DIR, "eval_python.json")
inference_output = os.path.join(OUTPUT_DIR, "inference_results.json")
model_path = os.path.join(MODEL_DIR, "checkpoint")

if os.path.exists(inference_output):
    print("  推理结果已存在，跳过推理")
    exit(0)

# 加载评测数据
print("  加载评测数据...")
with open(eval_file, 'r') as f:
    eval_data = json.load(f)
print(f"  评测样本: {len(eval_data)} 条")

# 加载训练好的模型
print("  加载训练好的模型...")
model = lazyllm.TrainableModule(model_path).deploy_method(deploy.vllm)
model.start()

# 构建prompt并推理
SYS_PROMPT = "You are an expert Python programmer. Write clean, correct Python code to solve the given problem."

print("  开始推理...")
results = []
for i, item in enumerate(eval_data):
    prompt = item.get('instruction', '')
    reference = item.get('output', '')

    full_prompt = f"{SYS_PROMPT}\n\n### Problem:\n{prompt}\n\n### Solution:\n"
    response = model(full_prompt)

    results.append({
        'id': i,
        'prompt': prompt,
        'reference': reference,
        'prediction': response
    })

    if (i + 1) % 10 == 0:
        print(f"    已处理: {i+1}/{len(eval_data)}")

# 保存推理结果
with open(inference_output, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"  推理完成: {inference_output}")
model.stop()
EOF

if [ $? -ne 0 ]; then
    log_error "步骤4失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤5: 评估 ============
log_step "[5/5] 运行代码评估..."

python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import re
import ast
import os
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor

OUTPUT_DIR = "$OUTPUT_DIR"

inference_file = os.path.join(OUTPUT_DIR, "inference_results.json")
report_path = os.path.join(OUTPUT_DIR, "evaluation_report.csv")

if os.path.exists(report_path):
    print("  评估报告已存在，跳过评估")
    exit(0)

def extract_code(text):
    pattern = r'\`\`\`python\s+(.*?)\s+\`\`\`'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern = r'\`\`\`\s+(.*?)\s+\`\`\`'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.strip().startswith('def ') or text.strip().startswith('import '):
        return text.strip()
    return None

def check_syntax(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def run_in_docker(code, case_id):
    file_name = f'tmp_run_{case_id}.py'
    lines = code.split('\n')
    indented = '\n'.join('        ' + line for line in lines)
    mock_wrapper = f"""
import unittest.mock
import sys
def mock_input(prompt=''):
    return '25'
with unittest.mock.patch('builtins.input', side_effect=mock_input):
    try:
{indented}
    except Exception as e:
        print(f'RUNTIME_ERROR: {{e}}', file=sys.stderr)
"""
    final_code = mock_wrapper if 'input(' in code else code
    with open(file_name, 'w') as f:
        f.write('import sys\n' + final_code)

    try:
        result = subprocess.run([
            'docker', 'run', '--rm', '--network', 'none', '--memory', '128m',
            '-v', f'{os.path.abspath(file_name)}:/app/test.py',
            'python-sandbox', 'python', '/app/test.py'
        ], capture_output=True, text=True, timeout=10)
        os.remove(file_name)
        return 'Pass' if result.returncode == 0 else 'Fail', result.stdout, result.stderr
    except Exception as e:
        if os.path.exists(file_name):
            os.remove(file_name)
        return 'Error', '', str(e)

def evaluate_case(item):
    idx = item.get('id', 0)
    code = extract_code(item.get('prediction', ''))
    if not code:
        return {'id': idx, 'status': 'NoCode'}
    valid, err = check_syntax(code)
    if not valid:
        return {'id': idx, 'status': 'SyntaxError', 'error': err}
    status, stdout, stderr = run_in_docker(code, idx)
    return {'id': idx, 'status': status, 'stdout': stdout[:200], 'stderr': stderr[:200]}

# 加载推理结果
with open(inference_file, 'r') as f:
    inference_data = json.load(f)

print(f"  评估 {len(inference_data)} 条推理结果...")
results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(evaluate_case, inference_data))

with open(report_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'status', 'stdout', 'stderr', 'error'])
    writer.writeheader()
    writer.writerows(results)

summary = {}
for r in results:
    summary[r['status']] = summary.get(r['status'], 0) + 1
print(f"  评估结果: {summary}")
EOF

if [ $? -ne 0 ]; then
    log_error "步骤5失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 完成 ============
log ""
log "=========================================="
log "全部完成!"
log "=========================================="
log ""
log_info "结果汇总:"
log "  数据目录: $DATA_DIR"
log "  模型目录: $MODEL_DIR/checkpoint"
log "  推理结果: $OUTPUT_DIR/inference_results.json"
log "  评估报告: $OUTPUT_DIR/evaluation_report.csv"
log "  日志文件: $LOG_FILE"
log ""

# 显示评估结果摘要
if [ -f "$OUTPUT_DIR/evaluation_report.csv" ]; then
    log_info "评估统计:"
    python3 << RESULT 2>&1 | tee -a "$LOG_FILE"
import csv
from collections import Counter

with open("$OUTPUT_DIR/evaluation_report.csv", 'r') as f:
    reader = csv.DictReader(f)
    statuses = [row['status'] for row in reader]
    summary = Counter(statuses)
    total = len(statuses)
    pass_count = summary.get('Pass', 0)
    print(f"  - 总样本: {total}")
    print(f"  - 通过: {pass_count} ({pass_count/total*100:.1f}%)")
    print(f"  - 失败: {summary.get('Fail', 0)}")
    print(f"  - 语法错误: {summary.get('SyntaxError', 0)}")
    print(f"  - 无代码: {summary.get('NoCode', 0)}")
    print(f"  - 错误: {summary.get('Error', 0)}")
RESULT
fi

log ""
log "=========================================="

safe_exit 0
