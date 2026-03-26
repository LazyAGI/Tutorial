#!/bin/bash
# Text2SQL 一键处理脚本
# 流程: 数据转换 -> Pipeline生成 -> 训练 -> 测试 -> 评估

# 防止被 source 执行导致终端退出
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "错误: 请不要用 source 执行此脚本，直接运行: bash ${BASH_SOURCE[0]}"
    return 1 2>/dev/null || exit 1
fi

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置文件路径
BASE_DIR="/home/mnt/huangchongjin/from-data-to-llm/docs/chapter22/code"
OUTPUT_BASE_DIR="/home/mnt/huangchongjin/from-data-to-llm/docs/chapter22/test"

# 数据文件
INPUT_DATA="${BASE_DIR}/hardest_5000_sql_sft.json"
HARD_SQL="${OUTPUT_BASE_DIR}/hard_sql.json"
PPL_SQL="${OUTPUT_BASE_DIR}/ppl_sql.json"
ENHANCED_PPL_SQL="${OUTPUT_BASE_DIR}/enhanced_ppl_sql.json"

# 训练配置
TRAIN_DATA="${OUTPUT_BASE_DIR}/bird/improved_ppl_sql.json"
CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/enhance_ppl_checkpoint"

# 测试配置
TEST_INPUT="${OUTPUT_BASE_DIR}/bird/hardest_1000_sql_test_format.jsonl"
TEST_OUTPUT="${OUTPUT_BASE_DIR}/spidertest/enhance_ppl.json"

# 评估输出
EVAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/enhance_ppl_eval_results"

# 日志文件
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# 记录开始时间
START_TIME=$(date +%s)

# 帮助信息
show_help() {
    cat << EOF
Text2SQL 一键处理脚本

用法: $0 [选项] [步骤]

步骤:
    all         运行完整流程 (默认)
    transform   仅运行数据转换
    pipeline    仅运行 Pipeline 生成
    train       仅运行训练
    test        仅运行测试
    eval        仅运行评估

选项:
    -h, --help      显示帮助信息
    -s, --skip      跳过指定步骤 (用逗号分隔, 如: transform,train)
    -m, --max-items 设置 Pipeline 处理的最大数据量 (默认: 1000)
    --dry-run       只显示要执行的命令, 不实际运行

示例:
    $0                      # 运行完整流程
    $0 transform            # 仅运行数据转换
    $0 pipeline             # 仅运行 Pipeline 生成
    $0 train                # 仅运行训练
    $0 test                 # 仅运行测试
    $0 eval                 # 仅运行评估
    $0 all -s train         # 运行除训练外的所有步骤
    $0 all -m 500           # 限制 Pipeline 处理 500 条数据
EOF
}

# 解析命令行参数
SKIP_STEPS=""
MAX_ITEMS=1000
DRY_RUN=false
STEP="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--skip)
            SKIP_STEPS="$2"
            shift 2
            ;;
        -m|--max-items)
            MAX_ITEMS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        transform|pipeline|train|test|eval|all)
            STEP="$1"
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查是否需要跳过某步骤
should_skip() {
    local step=$1
    if [[ ",${SKIP_STEPS}," == *",${step},"* ]]; then
        return 0
    fi
    return 1
}

# 执行命令或打印命令（dry-run模式）
run_cmd() {
    local cmd="$1"
    local step="$2"
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] 将要执行: $cmd"
    else
        log_info "执行: $step"
        echo "命令: $cmd" >> "${MASTER_LOG}"
        # 执行命令并捕获退出码，不使用 pipefail 避免终端退出
        local temp_output="${LOG_DIR}/temp_output_$$.log"
        eval "$cmd" > "$temp_output" 2>&1
        local exit_code=$?
        cat "$temp_output" | tee -a "${MASTER_LOG}"
        rm -f "$temp_output"
        if [[ $exit_code -ne 0 ]]; then
            log_error "$step 失败! (exit code: $exit_code)"
            exit 1
        fi
    fi
}

# 步骤1: 数据转换
step_transform() {
    log_info "========== 步骤 1/5: 数据转换 =========="

    if should_skip "transform"; then
        log_warn "跳过数据转换步骤"
        return 0
    fi

    if [[ ! -f "$INPUT_DATA" ]]; then
        log_error "输入数据文件不存在: $INPUT_DATA"
        exit 1
    fi

    # 创建临时转换脚本
    local temp_transform_script="${BASE_DIR}/.temp_transform_${TIMESTAMP}.py"

    cat > "$temp_transform_script" << 'TRANSFORM_SCRIPT'
#!/usr/bin/env python3
import json
import re
import os
import sys

def transform(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transformed = []
    for idx, item in enumerate(data):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_sql = item.get('output', '')

        # Extract schema and question
        parts = input_text.split("Question:", 1)
        if len(parts) == 2:
            schema = parts[0].replace("Database Schema:", "").strip()
            question = parts[1].strip()
        else:
            schema = ""
            question = input_text.strip()

        transformed.append({
            "db_id": f"spider_db_{idx}",
            "question": question,
            "schema": schema,
            "gold_sql": output_sql,
            "prompt": f"{instruction}\n\n{input_text}" if instruction else input_text,
            "_original_instruction": instruction,
            "_original_input": input_text
        })

    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "hard_sql.json")
    output_jsonl = os.path.join(output_dir, "hard_sql.jsonl")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in transformed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Transformed {len(transformed)} items")
    print(f"Output saved to: {output_json}")
    print(f"JSONL saved to: {output_jsonl}")

if __name__ == "__main__":
    transform(sys.argv[1], sys.argv[2])
TRANSFORM_SCRIPT

    run_cmd "python3 '${temp_transform_script}' '${INPUT_DATA}' '${OUTPUT_BASE_DIR}'" "数据转换"

    # 清理临时文件
    rm -f "$temp_transform_script"

    if [[ ! -f "$HARD_SQL" ]]; then
        log_error "数据转换失败,未生成: $HARD_SQL"
        exit 1
    fi

    log_success "数据转换完成: $HARD_SQL"
}

# 步骤2: Pipeline 生成
step_pipeline() {
    log_info "========== 步骤 2/5: Pipeline 生成 =========="

    if should_skip "pipeline"; then
        log_warn "跳过 Pipeline 生成步骤"
        return 0
    fi

    if [[ ! -f "$HARD_SQL" ]]; then
        log_error "找不到输入文件: $HARD_SQL, 请先运行 transform 步骤"
        exit 1
    fi

    # 创建临时修改版的 run_text2sql_ppl.py
    local temp_ppl_script="${BASE_DIR}/.temp_run_ppl_${TIMESTAMP}.py"

    cat > "$temp_ppl_script" << PPL_SCRIPT
#!/usr/bin/env python
import json
import os
import sys
import shutil

# 添加 lazyllm 路径
sys.path.insert(0, '/home/mnt/huangchongjin/ppl_new')

import lazyllm
from lazyllm.tools.data.pipelines import text2sql_synthetic_ppl

# 配置参数 - 从环境变量读取
USE_LOCAL_MODEL = os.environ.get('USE_LOCAL_MODEL', 'true').lower() == 'true'
DATA_FILE = os.environ.get('DATA_FILE', '/home/mnt/huangchongjin/ppl_new/lazyllm/tools/data/ex/text2sql/hard_sql.json')
OUTPUT_FILE = os.environ.get('OUTPUT_FILE', '/home/mnt/huangchongjin/from-data-to-llm/docs/chapter22/test/ppl_sql.json')
MAX_ITEMS = int(os.environ.get('MAX_ITEMS', '1000'))
OUTPUT_NUM = int(os.environ.get('OUTPUT_NUM', '2'))
INPUT_QUERY_NUM = int(os.environ.get('INPUT_QUERY_NUM', '3'))
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', '5'))
OUTPUT_FORMAT = os.environ.get('OUTPUT_FORMAT', 'alpaca')
TARGET_COMPLEXITY = os.environ.get('TARGET_COMPLEXITY', 'hard')


class MockDatabaseManager:
    """模拟数据库管理器用于演示"""

    def __init__(self):
        self.db_type = 'sqlite'
        self.databases = {}
        self._db_schemas = {}

    def register_schema(self, db_id, schema_str):
        """注册数据库 schema"""
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


def create_local_model():
    model_path = '/mnt/lustre/share_data/lazyllm/models/Qwen3-30B-A3B-Instruct-2507'
    model = lazyllm.TrainableModule(model_path)
    return model


def load_sample_data_from_json(filepath, max_items=1000):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > max_items:
        data = data[:max_items]
        print(f"   已限制为前 {max_items} 条数据")
    return data


def save_results(results, filepath):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {filepath}")


def run_text2sql_pipeline(use_local_model=True, data_file=None, max_items=1000,
                           output_num=2, input_query_num=3, num_generations=5,
                           output_format='alpaca', target_complexity='hard'):
    print("=" * 60)
    print("Text2SQL Pipeline")
    print("=" * 60)

    print("\n2. 初始化模拟数据库管理器")
    db_manager = MockDatabaseManager()
    print(f"   模拟数据库: {db_manager.list_databases()}")

    sample_data = load_sample_data_from_json(data_file, max_items)
    for item in sample_data:
        db_id = item.get('db_id')
        schema = item.get('schema', '')
        if db_id and schema:
            db_manager.register_schema(db_id, schema)
    print(f"   已注册 {len(db_manager._db_schemas)} 个数据库 schema")

    print("\n3. 初始化 TrainableModule")
    model = None
    if use_local_model:
        try:
            model = create_local_model()
            print("   模型初始化成功，正在启动...")
            model.start()
            print("   模型启动成功")
        except Exception as e:
            print(f"   模型初始化失败: {str(e)}")
            return []
    else:
        print("   Using mock mode (model=None)")

    print("\n4. 构建 Text2SQL Pipeline")
    print(f"   输出格式: {output_format}")
    pipeline = text2sql_synthetic_ppl(
        model=model,
        embedding_model=None,
        database_manager=db_manager,
        output_num=output_num,
        input_query_num=input_query_num,
        num_generations=num_generations,
        output_format=output_format,
        target_complexity=target_complexity
    )

    print("\n5. 准备输入数据")
    print(f"   准备处理 {len(sample_data)} 个数据库")

    print("\n6. 处理数据...")
    results = []
    try:
        results = pipeline(sample_data)
        print(f"\n   Pipeline 返回结果数量: {len(results)}")
    except Exception as e:
        print(f"   Pipeline 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        results = []

    if model is not None:
        print("\n7. 停止模型")
        try:
            model.stop()
            print("   模型已停止")
        except Exception as e:
            print(f"   停止模型时出错: {e}")

    return results


def main():
    print("=" * 60)
    print("Text2SQL Pipeline Runner")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  - Use local model: {USE_LOCAL_MODEL}")
    print(f"  - Data file: {DATA_FILE}")
    print(f"  - Output file: {OUTPUT_FILE}")
    print(f"  - Max items: {MAX_ITEMS}")

    results = run_text2sql_pipeline(
        use_local_model=USE_LOCAL_MODEL,
        data_file=DATA_FILE,
        max_items=MAX_ITEMS,
        output_num=OUTPUT_NUM,
        input_query_num=INPUT_QUERY_NUM,
        num_generations=NUM_GENERATIONS,
        output_format=OUTPUT_FORMAT,
        target_complexity=TARGET_COMPLEXITY
    )

    print("\n" + "=" * 60)
    print(f"处理统计:")
    print(f"  - 总数据: {len(results)}")

    success_count = sum(1 for r in results if isinstance(r, dict) and 'SQL' in r)
    print(f"  - 成功生成 SQL: {success_count}/{len(results)}")
    print("=" * 60)

    output_path = OUTPUT_FILE
    print(f"\n保存结果")
    save_results(results, output_path)
    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
PPL_SCRIPT

    # 设置环境变量
    export USE_LOCAL_MODEL=true
    export DATA_FILE="$HARD_SQL"
    export OUTPUT_FILE="$PPL_SQL"
    export MAX_ITEMS="$MAX_ITEMS"
    export OUTPUT_NUM=2
    export INPUT_QUERY_NUM=3
    export NUM_GENERATIONS=5
    export OUTPUT_FORMAT="alpaca"
    export TARGET_COMPLEXITY="hard"

    run_cmd "cd '${BASE_DIR}' && python3 '${temp_ppl_script}'" "Pipeline 生成"

    # 清理临时文件
    rm -f "$temp_ppl_script"

    if [[ ! -f "$PPL_SQL" ]]; then
        log_error "Pipeline 生成失败,未生成: $PPL_SQL"
        exit 1
    fi

    log_success "Pipeline 生成完成: $PPL_SQL"
}

# 步骤3: 训练
step_train() {
    log_info "========== 步骤 3/5: 模型训练 =========="

    if should_skip "train"; then
        log_warn "跳过训练步骤"
        return 0
    fi

    # 检查训练数据
    if [[ ! -f "$TRAIN_DATA" ]]; then
        # 尝试使用 pipeline 生成的数据
        if [[ -f "$ENHANCED_PPL_SQL" ]]; then
            log_warn "使用 $ENHANCED_PPL_SQL 作为训练数据"
            TRAIN_DATA="$ENHANCED_PPL_SQL"
        elif [[ -f "$PPL_SQL" ]]; then
            log_warn "使用 $PPL_SQL 作为训练数据"
            TRAIN_DATA="$PPL_SQL"
        else
            log_error "找不到训练数据文件"
            exit 1
        fi
    fi

    # 创建临时训练脚本
    local temp_train_script="${BASE_DIR}/.temp_train_${TIMESTAMP}.py"

    cat > "$temp_train_script" << TRAIN_SCRIPT
import sys
import os

# 添加 lazyllm 路径
sys.path.insert(0, '/home/mnt/huangchongjin/ppl_new')

local_path = "/home/mnt/huangchongjin/.local/lib/python3.10/site-packages"
if local_path not in sys.path:
    sys.path.insert(0, local_path)

import lazyllm
from lazyllm import finetune, deploy, launchers

model_path="/home/mnt/huangchongjin/.lazyllm/model/modelscope/Qwen/Qwen2.5-0.5B-Instruct"
model = lazyllm.TrainableModule(model_path, target_path='${CHECKPOINT_DIR}')\\
    .mode('finetune')\\
    .trainset('${TRAIN_DATA}')\\
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
        'stage':'sft',
        'save_steps': 10,
        'resume_from_checkpoint': None,
        'save_strategy': 'steps',
        'save_total_limit': 3,
        'launcher': launchers.sco(
            ngpus=1,
            partition='a800',
            resource='N3lS.Ii.I60.1',
        ),
    }))

model.update()
TRAIN_SCRIPT

    log_info "训练配置:"
    log_info "  - 基础模型: Qwen2.5-0.5B-Instruct"
    log_info "  - 训练数据: $TRAIN_DATA"
    log_info "  - 输出目录: $CHECKPOINT_DIR"

    run_cmd "cd '${BASE_DIR}' && python3 '${temp_train_script}'" "模型训练"

    # 清理临时文件
    rm -f "$temp_train_script"

    # 查找生成的 checkpoint
    CHECKPOINT_PATH=$(find "$CHECKPOINT_DIR" -name "lazyllm_merge" -type d 2>/dev/null | head -1)
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        log_warn "未找到合并后的 checkpoint, 但训练可能仍在进行中"
        log_warn "请训练完成后手动运行 test 步骤"
    else
        log_success "训练完成: $CHECKPOINT_PATH"
    fi
}

# 步骤4: 测试
step_test() {
    log_info "========== 步骤 4/5: 模型测试 =========="

    if should_skip "test"; then
        log_warn "跳过测试步骤"
        return 0
    fi

    # 查找 checkpoint
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        CHECKPOINT_PATH=$(find "$CHECKPOINT_DIR" -name "lazyllm_merge" -type d 2>/dev/null | head -1)
    fi

    if [[ -z "$CHECKPOINT_PATH" ]] || [[ ! -d "$CHECKPOINT_PATH" ]]; then
        log_error "找不到训练好的模型 checkpoint"
        log_error "请确认训练步骤已完成或手动指定 checkpoint 路径"
        exit 1
    fi

    if [[ ! -f "$TEST_INPUT" ]]; then
        log_error "找不到测试数据: $TEST_INPUT"
        exit 1
    fi

    # 创建临时测试脚本
    local temp_test_script="${BASE_DIR}/.temp_test_${TIMESTAMP}.py"

    cat > "$temp_test_script" << TESTSCRIPT
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/mnt/huangchongjin/ppl_new')

import lazyllm
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
import json
from datetime import datetime
from vllm import LLM, SamplingParams
import re
import os

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    os.environ["VLLM_USE_V1_ENGINE"] = "0"
    print(f"Using vLLM for accelerated inference...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        enforce_eager=True
    )
    print(f"Model loaded successfully with vLLM!")
    return llm

def extract_sql_from_response(response):
    sql_pattern = r'\`\`\`sql\s*(.*?)\s*\`\`\`'
    matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    code_pattern = r'\`\`\`\s*(.*?)\s*\`\`\`'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    select_pattern = r'(SELECT\s+.*?)\s*(?:\n|$)'
    matches = re.findall(select_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return response.strip()

def validate_sql(sql):
    issues = []
    if not sql:
        issues.append("Empty SQL")
        return False, issues
    sql_upper = sql.upper()
    if not sql_upper.startswith('SELECT'):
        issues.append("SQL doesn't start with SELECT")
    if 'FROM' not in sql_upper:
        issues.append("Missing FROM clause")
    if sql.count('(') != sql.count(')'):
        issues.append("Unbalanced parentheses")
    dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']
    for kw in dangerous:
        if kw in sql_upper:
            issues.append(f"Contains dangerous keyword: {kw}")
            break
    is_valid = len(issues) == 0 or all(i.startswith("SQL doesn't") or i.startswith("Missing") for i in issues)
    return is_valid, issues

def generate_response(llm, prompt, max_new_tokens=512, temperature=0.1):
    system_prompt = "You are a SQL expert. Based on the database schema provided, generate a SQL query to answer the question. Return ONLY the SQL query without any explanation."
    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
        stop=[";", "\n\n", "Question:", "<|im_end|>"],
    )
    outputs = llm.generate(formatted_prompt, sampling_params=sampling_params)
    raw_response = outputs[0].outputs[0].text
    extracted_sql = extract_sql_from_response(raw_response)
    is_valid, validation_issues = validate_sql(extracted_sql)
    return {
        'raw_response': raw_response,
        'extracted_sql': extracted_sql,
        'is_valid': is_valid,
        'validation_issues': validation_issues
    }

def test_model(llm, test_cases):
    results = []
    valid_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"DB ID: {test_case.get('db_id', 'N/A')}")
        print(f"Question: {test_case['question']}")
        try:
            result = generate_response(llm, test_case['prompt'])
            print(f"\nExtracted SQL: {result['extracted_sql']}")
            print(f"Valid SQL: {result['is_valid']}")
            if result['is_valid']:
                valid_count += 1
            results.append({
                "test_case_id": i,
                "db_id": test_case.get('db_id', ''),
                "question": test_case['question'],
                "gold_sql": test_case.get('gold_sql', ''),
                "raw_response": result['raw_response'],
                "predicted_sql": result['extracted_sql'],
                "is_valid": result['is_valid'],
                "validation_issues": result['validation_issues'],
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"\nError: {e}")
            results.append({
                "test_case_id": i,
                "db_id": test_case.get('db_id', ''),
                "question": test_case['question'],
                "gold_sql": test_case.get('gold_sql', ''),
                "raw_response": f"ERROR: {str(e)}",
                "predicted_sql": "",
                "is_valid": False,
                "validation_issues": [str(e)],
                "timestamp": datetime.now().isoformat()
            })
    print(f"\n{'='*80}")
    print(f"Valid SQL Rate: {valid_count}/{len(test_cases)} ({valid_count/len(test_cases)*100:.1f}%)")
    print(f"{'='*80}")
    return results

def load_test_cases(jsonl_file):
    test_cases = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'schema' in data and 'question' in data:
                prompt = f"Database Schema:\n{data['schema']}\n\nQuestion: {data['question']}"
                test_cases.append({
                    'prompt': prompt,
                    'question': data['question'],
                    'gold_sql': data.get('SQL', ''),
                    'db_id': data.get('db_id', '')
                })
    return test_cases

def main():
    model_path = '${CHECKPOINT_PATH}'
    output_path = '${TEST_OUTPUT}'
    test_cases_file = '${TEST_INPUT}'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_cases = load_test_cases(test_cases_file)

    print("="*80)
    print(" Model Testing Script")
    print("="*80)
    print(f"Model Path: {model_path}")
    print(f"Output Path: {output_path}")
    print(f"Number of Test Cases: {len(test_cases)}")
    print("="*80)

    llm = load_model(model_path)
    results = test_model(llm, test_cases)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "="*80)
    print("Testing Summary")
    print("="*80)
    print(f"Total test cases: {len(results)}")
    successful = sum(1 for r in results if not r["raw_response"].startswith("ERROR"))
    valid_sql = sum(1 for r in results if r.get("is_valid", False))
    print(f"Successful generations: {successful}")
    print(f"Valid SQL outputs: {valid_sql} ({valid_sql/len(results)*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
TESTSCRIPT

    log_info "测试配置:"
    log_info "  - 模型: $CHECKPOINT_PATH"
    log_info "  - 测试数据: $TEST_INPUT"
    log_info "  - 输出: $TEST_OUTPUT"

    run_cmd "cd '${BASE_DIR}' && python3 '${temp_test_script}'" "模型测试"

    # 清理临时文件
    rm -f "$temp_test_script"

    if [[ ! -f "$TEST_OUTPUT" ]]; then
        log_error "测试失败,未生成: $TEST_OUTPUT"
        exit 1
    fi

    log_success "测试完成: $TEST_OUTPUT"
}

# 步骤5: 评估
step_eval() {
    log_info "========== 步骤 5/5: 结果评估 =========="

    if should_skip "eval"; then
        log_warn "跳过评估步骤"
        return 0
    fi

    if [[ ! -f "$TEST_OUTPUT" ]]; then
        log_error "找不到测试结果: $TEST_OUTPUT"
        log_error "请先运行 test 步骤"
        exit 1
    fi

    if [[ ! -f "$TEST_INPUT" ]]; then
        log_error "找不到参考答案: $TEST_INPUT"
        exit 1
    fi

    log_info "评估配置:"
    log_info "  - 测试结果: $TEST_OUTPUT"
    log_info "  - 参考答案: $TEST_INPUT"
    log_info "  - 输出目录: $EVAL_OUTPUT_DIR"

    # 创建临时评估脚本
    local temp_eval_script="${BASE_DIR}/.temp_eval_${TIMESTAMP}.py"

    cat > "$temp_eval_script" << 'EVALSCRIPT'
#!/usr/bin/env python3
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, '/home/mnt/huangchongjin/ppl_new')

import lazyllm
from lazyllm import LOG


class Text2SQLJudge:
    """使用 LLM 评估 Text2SQL 结果"""

    JUDGE_PROMPT = """你是一个 非常非常严格的SQL 评估专家。请评估生成的 SQL 是否正确回答了用户问题。

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
\`\`\`json
{
    "semantic_score": 5,
    "syntax_score": 5,
    "equivalence_score": 3,
    "overall_score": 5.0,
    "is_correct": true,
    "reason": "SQL 完全正确，正确理解了用户意图"
}
\`\`\`
"""

    def __init__(self, model_path: str = "/mnt/lustre/share_data/lazyllm/models/Qwen3-30B-A3B-Instruct-2507"):
        print(f"正在加载评判模型: {model_path}")
        self.model = lazyllm.TrainableModule(model_path)
        self.model.start()
        print("评判模型加载完成！")

    def evaluate_single(self, question: str, gold_sql: str, pred_sql: str) -> dict:
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            gold_sql=gold_sql,
            pred_sql=pred_sql
        )
        try:
            response = self.model(prompt)
            import re
            json_match = re.search(r'\`\`\`json\s*(.*?)\s*\`\`\`', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(response.strip())
            return result
        except Exception as e:
            LOG.warning(f"解析评估结果失败: {e}")
            return {
                "semantic_score": 0,
                "syntax_score": 0,
                "equivalence_score": 0,
                "overall_score": 0.0,
                "is_correct": False,
                "reason": f"评估失败: {str(e)}"
            }

    def evaluate_batch(self, predictions: list, references: list) -> list:
        results = []
        total = len(predictions)
        ref_dict = {ref.get('instruction', ref.get('question', '')): ref['SQL']
                    for ref in references}

        for i, pred in enumerate(predictions, 1):
            question = pred.get('question', '')
            pred_sql = pred.get('raw_response', '') or pred.get('response', '')
            gold_sql = ref_dict.get(question, '')

            print(f"\n[{i}/{total}] 评估中...")
            print(f"  问题: {question[:50]}...")

            eval_result = self.evaluate_single(question, gold_sql, pred_sql)

            results.append({
                "test_case_id": pred.get('test_case_id', i),
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "evaluation": eval_result
            })

            print(f"  评分: {eval_result.get('overall_score', 0):.1f}/5.0, "
                  f"正确: {eval_result.get('is_correct', False)}")

        return results

    def stop(self):
        self.model.stop()


def load_jsonl_file(filepath: str) -> list:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def calculate_stats(results: list) -> dict:
    total = len(results)
    correct_count = sum(1 for r in results if r['evaluation'].get('is_correct', False))
    scores = [r['evaluation'].get('overall_score', 0) for r in results]
    semantic_scores = [r['evaluation'].get('semantic_score', 0) for r in results]
    syntax_scores = [r['evaluation'].get('syntax_score', 0) for r in results]
    equivalence_scores = [r['evaluation'].get('equivalence_score', 0) for r in results]

    return {
        "total_samples": total,
        "correct_count": correct_count,
        "accuracy": correct_count / total if total > 0 else 0,
        "avg_overall_score": sum(scores) / len(scores) if scores else 0,
        "avg_semantic_score": sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0,
        "avg_syntax_score": sum(syntax_scores) / len(syntax_scores) if syntax_scores else 0,
        "avg_equivalence_score": sum(equivalence_scores) / len(equivalence_scores) if equivalence_scores else 0,
    }


def print_report(stats: dict):
    print("\n" + "="*80)
    print("LLM as Judge 评估报告")
    print("="*80)
    print(f"\n【模型表现】")
    print(f"  平均总分: {stats['avg_overall_score']:.2f}/5.0")
    print(f"  语义得分: {stats['avg_semantic_score']:.2f}/5.0")
    print(f"  语法得分: {stats['avg_syntax_score']:.2f}/5.0")
    print(f"  等价得分: {stats['avg_equivalence_score']:.2f}/5.0")
    print(f"  正确率: {stats['accuracy']*100:.1f}%")
    print("\n" + "="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    ppl_file = args.predictions
    reference_file = args.references
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%H%M%S")

    print("加载数据...")
    ppl_data = json.load(open(ppl_file, 'r', encoding='utf-8'))
    reference_data = load_jsonl_file(reference_file)

    print(f"PPL结果: {len(ppl_data)} 条")
    print(f"参考答案: {len(reference_data)} 条")

    judge = Text2SQLJudge()

    try:
        print("\n" + "="*80)
        print("评估PPL模型...")
        print("="*80)
        ppl_eval_results = judge.evaluate_batch(ppl_data, reference_data)

        ppl_eval_file = os.path.join(output_dir, f"ppl_eval_{timestamp}.json")
        with open(ppl_eval_file, 'w', encoding='utf-8') as f:
            json.dump(ppl_eval_results, f, ensure_ascii=False, indent=2)
        print(f"PPL模型评估结果已保存: {ppl_eval_file}")

        # 统计
        stats = calculate_stats(ppl_eval_results)
        print_report(stats)

        stats_file = os.path.join(output_dir, f"stats_{timestamp}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计结果已保存: {stats_file}")

    finally:
        judge.stop()


if __name__ == "__main__":
    main()
EVALSCRIPT

    run_cmd "cd '${BASE_DIR}' && python3 '${temp_eval_script}' --predictions '${TEST_OUTPUT}' --references '${TEST_INPUT}' --output-dir '${EVAL_OUTPUT_DIR}'" "结果评估"

    # 清理临时文件
    rm -f "$temp_eval_script"

    log_success "评估完成,结果保存在: $EVAL_OUTPUT_DIR"
}

# 打印执行计划
print_execution_plan() {
    log_info "=========================================="
    log_info "Text2SQL 一键处理脚本"
    log_info "=========================================="
    log_info "执行计划:"

    case "$STEP" in
        transform)
            log_info "  [1/1] 数据转换"
            ;;
        pipeline)
            log_info "  [1/1] Pipeline 生成"
            ;;
        train)
            log_info "  [1/1] 模型训练"
            ;;
        test)
            log_info "  [1/1] 模型测试"
            ;;
        eval)
            log_info "  [1/1] 结果评估"
            ;;
        all)
            if ! should_skip "transform"; then log_info "  [1/5] 数据转换"; fi
            if ! should_skip "pipeline"; then log_info "  [2/5] Pipeline 生成"; fi
            if ! should_skip "train"; then log_info "  [3/5] 模型训练"; fi
            if ! should_skip "test"; then log_info "  [4/5] 模型测试"; fi
            if ! should_skip "eval"; then log_info "  [5/5] 结果评估"; fi
            ;;
    esac

    if [[ -n "$SKIP_STEPS" ]]; then
        log_warn "跳过的步骤: $SKIP_STEPS"
    fi

    log_info "=========================================="
}

# 主执行流程
main() {
    print_execution_plan

    if [[ "$DRY_RUN" == true ]]; then
        log_warn "Dry-run 模式: 只显示将要执行的命令"
    fi

    case "$STEP" in
        transform)
            step_transform
            ;;
        pipeline)
            step_pipeline
            ;;
        train)
            step_train
            ;;
        test)
            step_test
            ;;
        eval)
            step_eval
            ;;
        all)
            step_transform
            step_pipeline
            step_train
            step_test
            step_eval
            ;;
        *)
            log_error "未知步骤: $STEP"
            show_help
            exit 1
            ;;
    esac

    # 计算总耗时
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    log_info "=========================================="
    log_success "所有步骤执行完成!"
    log_info "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    log_info "日志文件: ${MASTER_LOG}"
    log_info "=========================================="
}

# 运行主流程
main
