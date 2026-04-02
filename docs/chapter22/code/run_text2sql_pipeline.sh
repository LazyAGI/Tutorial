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
SFT_BASE_MODEL="/path/to/sft/base/model"
JUDGE_MODEL="/path/to/judge/model"

if [ ! -d "$LAZYLLM_PATH" ]; then
    log_error "LAZYLLM_PATH 不存在: $LAZYLLM_PATH"
    log "请修改脚本中的 LAZYLLM_PATH 配置"
    safe_exit 1
fi

log "=========================================="
log "一键Text2SQL训练脚本"
log "=========================================="
log ""
log_info "配置信息:"
log "  - 基础目录: $BASE_DIR"
log "  - 数据目录: $DATA_DIR"
log "  - 模型目录: $MODEL_DIR"
log "  - 输出目录: $OUTPUT_DIR"
log "  - 日志文件: $LOG_FILE"
log ""

# ============ 步骤1: 下载并准备数据 ============
log_step "[1/5] 下载 Text2SQL 数据集..."

python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import os
from datasets import load_dataset

DATA_DIR = "$DATA_DIR"
train_path = os.path.join(DATA_DIR, "train_text2sql.json")
test_path = os.path.join(DATA_DIR, "test_text2sql.jsonl")

if os.path.exists(train_path) and os.path.exists(test_path):
    print("  数据已存在，跳过下载")
    exit(0)

print("  正在从 Hugging Face 加载数据集 rirqing/text2sql...")
ds = load_dataset("rirqing/text2sql", trust_remote_code=True)

train_data = []
for idx, item in enumerate(ds['train']):
    train_data.append({
        "db_id": item.get('db_id', f"db_{idx}"),
        "question": item.get('question', ''),
        "schema": item.get('schema', ''),
        "gold_sql": item.get('gold_sql', ''),
        "prompt": item.get('prompt', ''),
        "instruction": item.get('instruction', ''),
        "input": item.get('input', ''),
        "output": item.get('output', '')
    })

test_data = []
for idx, item in enumerate(ds['test']):
    test_data.append({
        "db_id": item.get('db_id', f"db_{idx}"),
        "question": item.get('question', ''),
        "schema": item.get('schema', ''),
        "gold_sql": item.get('gold_sql', ''),
        "prompt": item.get('prompt', ''),
        "instruction": item.get('instruction', ''),
        "input": item.get('input', ''),
        "output": item.get('output', '')
    })

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(test_path, 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"  训练集: {len(train_data)} 条")
print(f"  测试集: {len(test_data)} 条")
EOF

if [ $? -ne 0 ]; then
    log_error "步骤1失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤2: 数据处理Pipeline ============
log_step "[2/5] 运行 Text2SQL Pipeline..."

python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys

LAZYLLM_PATH = "$LAZYLLM_PATH"
sys.path.insert(0, LAZYLLM_PATH)
sys.path.insert(0, os.path.dirname(LAZYLLM_PATH))

import lazyllm
from lazyllm.tools.data.pipelines import text2sql_synthetic_ppl

DATA_DIR = "$DATA_DIR"
input_file = os.path.join(DATA_DIR, "train_text2sql.json")
output_file = os.path.join(DATA_DIR, "ppl_text2sql.json")

if os.path.exists(output_file):
    print("  Pipeline 输出已存在，跳过处理")
    exit(0)

with open(input_file, 'r') as f:
    data = json.load(f)

print(f"  加载数据: {len(data)} 条")

class MockDatabaseManager:
    def __init__(self):
        self._db_schemas = {}

    def register_schema(self, db_id, schema_str):
        self._db_schemas[db_id] = schema_str

    def list_databases(self):
        return []

    def database_exists(self, db_id):
        return True

    def get_create_statements_and_insert_statements(self, db_id):
        schema = self._db_schemas.get(db_id, '')
        if schema:
            return [schema], []
        return [], []

    def batch_explain_queries(self, queries):
        class ExplainResult:
            def __init__(self, success):
                self.success = success
        return [ExplainResult(True) for _ in queries]

    def batch_execute_queries(self, queries):
        class ExecuteResult:
            def __init__(self, success, data=None, columns=None):
                self.success = success
                self.data = data or []
                self.columns = columns or []
        return [ExecuteResult(True, [{'id': 1}], ['id']) for _ in queries]

    def batch_compare_queries(self, comparisons):
        class CompareResult:
            def __init__(self, res):
                self.res = res
        import random
        return [CompareResult(random.choice([0, 1])) for _ in comparisons]

# 初始化数据库管理器
db_manager = MockDatabaseManager()
sample_data = data[:1000]  # 限制处理数量
for item in sample_data:
    db_id = item.get('db_id')
    schema = item.get('schema', '')
    if db_id and schema:
        db_manager.register_schema(db_id, schema)

print(f"  已注册 {len(db_manager._db_schemas)} 个数据库 schema")

# 加载模型
pipeline_model_path = "$PIPELINE_MODEL"
model = lazyllm.TrainableModule(pipeline_model_path)
model.start()

# 构建 Pipeline
ppl = text2sql_synthetic_ppl(
    model=model,
    embedding_model=None,
    database_manager=db_manager,
    output_num=2,
    input_query_num=3,
    num_generations=5,
    output_format='alpaca',
    target_complexity='hard'
)

print(f"  批量处理 {len(sample_data)} 条数据...")
results = ppl(sample_data)

output_data = []
for result in results:
    if result and isinstance(result, dict):
        output_data.append({
            "instruction": result.get("instruction", ""),
            "input": result.get("input", ""),
            "output": result.get("output", "")
        })

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

model.stop()
print(f"  生成数据: {len(output_data)} 条")
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
train_file = os.path.join(DATA_DIR, "ppl_text2sql.json")
checkpoint_dir = os.path.join(MODEL_DIR, "checkpoint")

if os.path.exists(checkpoint_dir):
    print("  模型已存在，跳过训练")
    exit(0)

sft_model_path = "$SFT_BASE_MODEL"
model = lazyllm.TrainableModule(sft_model_path, target_path=checkpoint_dir)\
    .mode('finetune')\
    .trainset(train_file)\
    .finetune_method((finetune.llamafactory, {
        'learning_rate': 1e-5,
        'cutoff_len': 4096,
        'max_samples': 7000,
        'val_size': 0.1,
        'optim': 'adamw_torch_fused',
        'bf16': True,
        'fp16': False,
        'per_device_train_batch_size': 8,
        'gradient_accumulation_steps': 4,
        'num_train_epochs': 3.0,
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
import glob
import re

local_path = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if local_path not in sys.path:
    sys.path.insert(0, local_path)

import lazyllm
from lazyllm import deploy

DATA_DIR = "$DATA_DIR"
OUTPUT_DIR = "$OUTPUT_DIR"
MODEL_DIR = "$MODEL_DIR"

test_file = os.path.join(DATA_DIR, "test_text2sql.jsonl")
inference_output = os.path.join(OUTPUT_DIR, "inference_results.json")

# 自动查找最新的 lazyllm_merge 目录
def find_latest_merge_model(base_dir):
    merge_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'lazyllm_merge' in dirs:
            path = os.path.join(root, 'lazyllm_merge')
            try:
                merge_dirs.append((path, os.path.getmtime(path)))
            except OSError:
                pass
    return max(merge_dirs, key=lambda x: x[1])[0] if merge_dirs else None

model_path = find_latest_merge_model(MODEL_DIR)
if not model_path:
    print(f"  错误: 在 {MODEL_DIR} 下未找到 lazyllm_merge 目录")
    exit(1)
print(f"  找到模型: {model_path}")

if os.path.exists(inference_output):
    print("  推理结果已存在，跳过推理")
    exit(0)

# 加载测试数据
print("  加载测试数据...")
test_data = []
with open(test_file, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))
print(f"  测试样本: {len(test_data)} 条")

# 加载训练好的模型
print("  加载训练好的模型...")
model = lazyllm.TrainableModule(model_path).deploy_method(deploy.vllm)
model.start()

# 构建prompt并推理
SYS_PROMPT = "You are a SQL expert. Based on the database schema provided, generate a SQL query to answer the question. Return ONLY the SQL query without any explanation."

def extract_sql(text):
    patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(SELECT\s+.*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()

print("  开始推理...")
results = []
for i, item in enumerate(test_data):
    schema = item.get('schema', '')
    question = item.get('question', '')
    gold_sql = item.get('gold_sql', '')

    prompt = f"{SYS_PROMPT}\n\nDatabase Schema:\n{schema}\n\nQuestion: {question}"
    response = model(prompt)
    extracted_sql = extract_sql(response)

    results.append({
        'test_case_id': i,
        'db_id': item.get('db_id', ''),
        'question': question,
        'schema': schema,
        'gold_sql': gold_sql,
        'raw_response': response,
        'predicted_sql': extracted_sql
    })

    if (i + 1) % 10 == 0:
        print(f"    已处理: {i+1}/{len(test_data)}")

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
log_step "[5/5] 运行 Text2SQL 评估..."

python3 << 'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = "$OUTPUT_DIR"
LAZYLLM_PATH = "$LAZYLLM_PATH"
JUDGE_MODEL = "$JUDGE_MODEL"
JUDGE_WORKERS = int(os.environ.get('JUDGE_WORKERS', '4'))

inference_file = os.path.join(OUTPUT_DIR, "inference_results.json")
report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")

if os.path.exists(report_path):
    print("  评估报告已存在，跳过评估")
    with open(report_path, 'r') as f:
        report = json.load(f)
    s = report['summary']
    print(f"  平均总分: {s['avg_overall_score']:.2f}/5.0")
    print(f"  语义得分: {s['avg_semantic_score']:.2f}/5.0")
    print(f"  语法得分: {s['avg_syntax_score']:.2f}/5.0")
    print(f"  等价得分: {s['avg_equivalence_score']:.2f}/3.0")
    print(f"  正确率: {s['accuracy']*100:.1f}%")
    exit(0)

sys.path.insert(0, LAZYLLM_PATH)
import lazyllm
from lazyllm import deploy

JUDGE_PROMPT = '''你是一个 非常非常严格的SQL 评估专家。请评估生成的 SQL 是否正确回答了用户问题。

【用户问题】
{question}

【标准答案 SQL】
{gold_sql}

【待评估 SQL】
{pred_sql}

请从以下维度评估（每题满分 5 分）：

1. **语义正确性** (0-5分): SQL 是否正确理解了用户问题的意图？
   - 5分: 完全正确理解意图
   - 3分: 部分理解，有 minor 错误
   - 1分: 理解有偏差
   - 0分: 完全错误

2. **语法正确性** (0-5分): SQL 语法是否正确？
   - 5分: 语法完全正确
   - 3分: 有小错误但不影响执行
   - 0分: 语法错误无法执行

3. **与标准答案一致性** (0-3分): 是否与标准答案等价？
   - 3分: 完全等价或更优
   - 0分: 不等价

请按以下格式输出严格的评估结果（只输出 JSON，不要有其他内容）：
```json
{
    "semantic_score": 5,
    "syntax_score": 5,
    "equivalence_score": 3,
    "overall_score": 5.0,
    "is_correct": true,
    "reason": "SQL 完全正确，正确理解了用户意图"
}
```'''

class SQLJudge:
    def __init__(self, model_path):
        self.model = lazyllm.TrainableModule(model_path).deploy_method((deploy.vllm, {
            'max_model_len': 4096,
            'gpu_memory_utilization': 0.9,
            'max_num_seqs': 8,
        })).start()

    def evaluate(self, question, gold_sql, pred_sql):
        prompt = JUDGE_PROMPT.format(
            question=question,
            gold_sql=gold_sql,
            pred_sql=pred_sql
        )
        try:
            result = self.model(prompt, max_tokens=256)
            json_match = re.search(r'```json\s*({.*?)\s*```', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(result.strip())
            return {
                'semantic_score': parsed.get('semantic_score', 0),
                'syntax_score': parsed.get('syntax_score', 0),
                'equivalence_score': parsed.get('equivalence_score', 0),
                'overall_score': parsed.get('overall_score', 0.0),
                'is_correct': parsed.get('is_correct', False),
                'reason': parsed.get('reason', '')
            }
        except Exception as e:
            return {
                'semantic_score': 0,
                'syntax_score': 0,
                'equivalence_score': 0,
                'overall_score': 0.0,
                'is_correct': False,
                'reason': f'评估失败: {str(e)}'
            }

# 加载推理结果
with open(inference_file, 'r') as f:
    inference_data = json.load(f)

print(f"  加载推理结果: {len(inference_data)} 条")

judge = SQLJudge(JUDGE_MODEL)

results = [None] * len(inference_data)

def evaluate_single(i, item):
    question = item.get('question', '')
    gold_sql = item.get('gold_sql', '')
    pred_sql = item.get('predicted_sql', '')
    eval_result = judge.evaluate(question, gold_sql, pred_sql)
    return i, {
        'test_case_id': item.get('test_case_id', i),
        'question': question[:100] + '...' if len(question) > 100 else question,
        'gold_sql': gold_sql[:200] + '...' if len(gold_sql) > 200 else gold_sql,
        'predicted_sql': pred_sql[:200] + '...' if len(pred_sql) > 200 else pred_sql,
        'evaluation': eval_result
    }

num_workers = max(1, min(JUDGE_WORKERS, len(inference_data)))
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(evaluate_single, i, item) for i, item in enumerate(inference_data)]

    for done_count, future in enumerate(as_completed(futures), 1):
        idx, result = future.result()
        results[idx] = result
        if done_count % 20 == 0 or done_count == len(inference_data):
            print(f"    已评估: {done_count}/{len(inference_data)}")

judge.model.stop()

total = len(results)
correct_count = sum(1 for r in results if r['evaluation']['is_correct'])
scores = [r['evaluation']['overall_score'] for r in results]
semantic_scores = [r['evaluation']['semantic_score'] for r in results]
syntax_scores = [r['evaluation']['syntax_score'] for r in results]
equivalence_scores = [r['evaluation']['equivalence_score'] for r in results]

stats = {
    'total_samples': total,
    'correct_count': correct_count,
    'accuracy': correct_count / total if total > 0 else 0,
    'avg_overall_score': sum(scores) / len(scores) if scores else 0,
    'avg_semantic_score': sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0,
    'avg_syntax_score': sum(syntax_scores) / len(syntax_scores) if syntax_scores else 0,
    'avg_equivalence_score': sum(equivalence_scores) / len(equivalence_scores) if equivalence_scores else 0,
}

print(f"  评估完成: {total} 条样本")
print(f"  平均总分: {stats['avg_overall_score']:.2f}/5.0")
print(f"  语义得分: {stats['avg_semantic_score']:.2f}/5.0")
print(f"  语法得分: {stats['avg_syntax_score']:.2f}/5.0")
print(f"  等价得分: {stats['avg_equivalence_score']:.2f}/3.0")
print(f"  正确率: {stats['accuracy']*100:.1f}%")

with open(report_path, 'w') as f:
    json.dump({
        'summary': stats,
        'details': results
    }, f, indent=2)

print(f"  报告保存: {report_path}")
PYEOF

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
log "  评估报告: $OUTPUT_DIR/evaluation_report.json"
log "  日志文件: $LOG_FILE"
log ""

# 显示评估结果摘要
if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
    log_info "评估统计:"
    python3 << RESULT 2>&1 | tee -a "$LOG_FILE"
import json
with open("$OUTPUT_DIR/evaluation_report.json", 'r') as f:
    report = json.load(f)
s = report['summary']
print(f"  - 总样本: {s['total_samples']}")
print(f"  - 平均总分: {s['avg_overall_score']:.2f}/5.0")
print(f"  - 语义得分: {s['avg_semantic_score']:.2f}/5.0")
print(f"  - 语法得分: {s['avg_syntax_score']:.2f}/5.0")
print(f"  - 等价得分: {s['avg_equivalence_score']:.2f}/3.0")
print(f"  - 正确率: {s['accuracy']*100:.1f}%")
RESULT
fi

log ""
log "=========================================="

safe_exit 0
