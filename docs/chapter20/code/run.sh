#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$BASE_DIR/data"
MODEL_DIR="$BASE_DIR/models"
OUTPUT_DIR="$BASE_DIR/output"
mkdir -p "$DATA_DIR" "$MODEL_DIR" "$OUTPUT_DIR"

LAZYLLM_PATH="/path/to/your/lazyllm"
PIPELINE_MODEL="/path/to/pipeline/model"
SFT_MODEL="/path/to/sft/base/model"

if [ ! -d "$LAZYLLM_PATH" ]; then
    echo "错误: 请修改脚本中的 LAZYLLM_PATH 配置"
    echo "当前路径: $LAZYLLM_PATH"
    exit 1
fi

echo "=========================================="
echo "一键代码SFT训练脚本"
echo "=========================================="

# ============ 步骤1: 下载数据 ============
echo ""
echo "[1/4] 下载 tiny-codes 数据集..."

python3 << EOF
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

# ============ 步骤2: 数据处理Pipeline ============
echo ""
echo "[2/4] 运行数据增强 pipeline..."

python3 << EOF
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

# ============ 步骤3: SFT训练 ============
echo ""
echo "[3/4] 开始 SFT 训练..."

python3 << EOF
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

# ============ 步骤4: 评估 ============
echo ""
echo "[4/4] 运行代码评估..."

python3 << EOF
import json
import re
import ast
import os
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "$DATA_DIR"
OUTPUT_DIR = "$OUTPUT_DIR"
MODEL_DIR = "$MODEL_DIR"

eval_file = os.path.join(DATA_DIR, "eval_python.json")
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

def evaluate_case(case, idx):
    code = extract_code(case.get('output', ''))
    if not code:
        return {'id': idx, 'status': 'NoCode'}
    valid, err = check_syntax(code)
    if not valid:
        return {'id': idx, 'status': 'SyntaxError', 'error': err}
    status, stdout, stderr = run_in_docker(code, idx)
    return {'id': idx, 'status': status, 'stdout': stdout[:200], 'stderr': stderr[:200]}

with open(eval_file, 'r') as f:
    data = json.load(f)[:50]

print(f"  评估 {len(data)} 条数据...")
results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(lambda x: evaluate_case(x[1], x[0]), enumerate(data)))

with open(report_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'status', 'stdout', 'stderr', 'error'])
    writer.writeheader()
    writer.writerows(results)

summary = {}
for r in results:
    summary[r['status']] = summary.get(r['status'], 0) + 1
print(f"  评估结果: {summary}")
EOF

# ============ 完成 ============
echo ""
echo "=========================================="
echo "全部完成!"
echo "=========================================="
echo "数据目录: $DATA_DIR"
echo "模型目录: $MODEL_DIR/checkpoint"
echo "评估报告: $OUTPUT_DIR/evaluation_report.csv"
echo ""
echo "Docker 镜像构建命令:"
echo "  docker build -t python-sandbox \$BASE_DIR"
