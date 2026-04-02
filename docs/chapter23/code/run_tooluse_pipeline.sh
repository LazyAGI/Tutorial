#!/bin/bash

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
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then
        return "$code"
    else
        exit "$code"
    fi
}

# ============================================
# 用户配置区域 - 请根据实际情况修改
# ============================================

# LazyLLM 路径
LAZYLLM_PATH="/path/to/lazyllm"

# 模型路径
PIPELINE_MODEL="/path/to/pipeline/model"      # 如: Qwen3-30B-A3B-Instruct
SFT_BASE_MODEL="/path/to/sft/base/model"      # 如: qwen2.5-0.5b-instruct
JUDGE_MODEL="/path/to/judge/model"            # 如: qwen2.5-14b-instruct

# 训练参数
TRAIN_NUM_SAMPLES=20000      # 训练样本数
EVAL_NUM_SAMPLES=1000        # 评测样本数
SFT_EPOCHS=3.0               # SFT训练轮数
SFT_LEARNING_RATE=5e-5       # 学习率
SFT_BATCH_SIZE=8             # 批次大小

# ============================================

# 检查配置
if [ ! -d "$LAZYLLM_PATH" ]; then
    echo "错误: LAZYLLM_PATH 不存在: $LAZYLLM_PATH"
    echo "请修改脚本中的 LAZYLLM_PATH 配置"
    safe_exit 1
fi

log "=========================================="
log "一键 Tool Use Pipeline 训练脚本"
log "=========================================="
log ""
log "配置信息:"
log "  - 基础目录: $BASE_DIR"
log "  - 数据目录: $DATA_DIR"
log "  - 模型目录: $MODEL_DIR"
log "  - 输出目录: $OUTPUT_DIR"
log "  - 日志文件: $LOG_FILE"
log "  - Pipeline模型: $PIPELINE_MODEL"
log "  - SFT基础模型: $SFT_BASE_MODEL"
log "  - 训练样本数: $TRAIN_NUM_SAMPLES"
log "  - 评测样本数: $EVAL_NUM_SAMPLES"
log ""

# ============ 步骤1: 下载并准备数据 ============
log_step "[1/5] 下载并准备 Tool Use 原始数据..."

python3 << STEP1 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys

DATA_DIR = "$DATA_DIR"
TRAIN_NUM = $TRAIN_NUM_SAMPLES
EVAL_NUM = $EVAL_NUM_SAMPLES
LAZYLLM_PATH = "$LAZYLLM_PATH"

raw_train_path = os.path.join(DATA_DIR, "tooluse_raw_train.json")
raw_eval_path = os.path.join(DATA_DIR, "tooluse_raw_eval.jsonl")

if os.path.exists(raw_train_path) and os.path.exists(raw_eval_path):
    print("  数据已存在，跳过下载")
    with open(raw_train_path, 'r') as f:
        train_count = len(json.load(f))
    with open(raw_eval_path, 'r') as f:
        eval_count = sum(1 for _ in f)
    print(f"  训练集: {train_count} 条")
    print(f"  评测集: {eval_count} 条")
else:
    # 导入 load_data 模块
    sys.path.insert(0, os.path.join(LAZYLLM_PATH, "lazyllm/tools/data/ex/tooluse"))
    from load_data import prepare_universal_data

    # 生成数据
    temp_raw = os.path.join(DATA_DIR, "temp_raw.json")
    total_needed = TRAIN_NUM + EVAL_NUM
    prepare_universal_data(output_file=temp_raw, num_samples=total_needed)

    # 分割训练集和评测集
    with open(temp_raw, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    train_data = all_data[:TRAIN_NUM]
    eval_data = all_data[TRAIN_NUM:TRAIN_NUM+EVAL_NUM]

    # 保存训练集 (JSON格式，用于pipeline)
    with open(raw_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # 保存评测集 (JSONL格式)
    with open(raw_eval_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(eval_data):
            item['test_case_id'] = i
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 清理临时文件
    os.remove(temp_raw)

    print(f"  原始训练数据: {raw_train_path} ({len(train_data)} 条)")
    print(f"  原始评测数据: {raw_eval_path} ({len(eval_data)} 条)")
STEP1

if [ $? -ne 0 ]; then
    log_error "步骤1失败！详细错误请查看日志: $LOG_FILE"
    read -p "按回车键退出..."
    safe_exit 1
fi

# ============ 步骤2: 运行 Tool Use Pipeline ============
log_step "[2/5] 运行 Tool Use Pipeline 生成 SFT 数据..."

python3 << STEP2 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys
import shutil

DATA_DIR = "$DATA_DIR"
LAZYLLM_PATH = "$LAZYLLM_PATH"
PIPELINE_MODEL = "$PIPELINE_MODEL"

sys.path.insert(0, LAZYLLM_PATH)

raw_data_path = os.path.join(DATA_DIR, "tooluse_raw_train.json")
ppl_output_path = os.path.join(DATA_DIR, "train_tooluse_sft.json")

if os.path.exists(ppl_output_path):
    print("  Pipeline 输出已存在，跳过处理")
    with open(ppl_output_path, 'r') as f:
        data = json.load(f)
    print(f"  已有数据: {len(data)} 条")
else:
    import lazyllm
    from lazyllm.tools.data import tool_use_ops
    from lazyllm import pipeline

    # 加载数据
    print(f"  加载原始数据...")
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"  共 {len(raw_data)} 条数据")

    # 初始化模型
    print("  初始化 Pipeline 模型...")
    model = lazyllm.TrainableModule(PIPELINE_MODEL)
    model.start()

    # 清除 pipeline 状态
    state_dir = os.path.join(os.getcwd(), 'data_pipeline_res')
    if os.path.exists(state_dir):
        for filename in os.listdir(state_dir):
            file_path = os.path.join(state_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except:
                pass

    # 构建 pipeline
    print("  构建 Tool Use Pipeline...")

    dialogue_system_prompt = (
        'You are a multi-turn dialogue data generation assistant. You need to simulate a multi-turn '
        'dialogue based on the composed task and available functions.\\n'
        'The dialogue consists of three roles: User/Assistant/Tool:\\n'
        '- User: Proposes requirements and supplementary information (in English)\\n'
        '- Assistant: Plans and calls Tool when appropriate (responds in English). '
        'The assistant must include thinking steps before providing the final answer. '
        'Use the following format for assistant responses:\\n\\n'
        '<answer>Your tool call or final answer</answer>\\n'
        'Output only JSON, no extra text.'
    )

    with pipeline() as ppl:
        ppl.contextual_beacon = tool_use_ops.ContextualBeacon(
            model=model, input_key='content', output_key='scenario'
        )
        ppl.decomposition_kernel = tool_use_ops.DecompositionKernel(
            model=model, input_key='scenario', output_key='atomic_tasks', n=2
        )
        ppl.protocol_specifier = tool_use_ops.ProtocolSpecifier(
            model=model, input_composition_key='atomic_tasks',
            input_atomic_key='atomic_tasks', output_key='functions'
        )
        ppl.dialogue_simulator = tool_use_ops.DialogueSimulator(
            model=model, input_composition_key='atomic_tasks',
            input_functions_key='functions', output_key='conversation',
            n_turns=2, system_prompt=dialogue_system_prompt
        )
        ppl.formatter = tool_use_ops.ToolUseToSFTFormatter(
            input_key='conversation', output_key='formatted', format_type='alpaca'
        )
        ppl.quality_filter = tool_use_ops.ToolUseQualityFilter(
            model=model, min_completeness_score=4, min_feasibility_score=4
        )

    # 处理数据
    print("  处理数据...")
    results = ppl(raw_data)
    print(f"  Pipeline 返回结果数量: {len(results)}")

    # 提取格式化后的数据
    formatted_results = []
    for item in results:
        if isinstance(item, dict) and 'formatted' in item:
            formatted = item['formatted']
            if isinstance(formatted, dict):
                formatted_results.append(formatted)

    print(f"  成功格式化: {len(formatted_results)} 条")

    # 保存结果
    with open(ppl_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

    print(f"  SFT数据保存: {ppl_output_path}")

    model.stop()
STEP2

if [ $? -ne 0 ]; then
    log_error "步骤2失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤3: SFT训练 ============
log_step "[3/5] 开始 SFT 训练..."

python3 << STEP3 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys

DATA_DIR = "$DATA_DIR"
MODEL_DIR = "$MODEL_DIR"
LAZYLLM_PATH = "$LAZYLLM_PATH"
SFT_BASE_MODEL = "$SFT_BASE_MODEL"
SFT_EPOCHS = $SFT_EPOCHS
SFT_LEARNING_RATE = $SFT_LEARNING_RATE
SFT_BATCH_SIZE = $SFT_BATCH_SIZE

sys.path.insert(0, LAZYLLM_PATH)

import lazyllm
from lazyllm import finetune, deploy, launchers

train_file = os.path.join(DATA_DIR, "train_tooluse_sft.json")
checkpoint_dir = os.path.join(MODEL_DIR, "tooluse_sft_checkpoint")

if os.path.exists(checkpoint_dir):
    print("  模型已存在，跳过训练")
else:
    print(f"  加载训练数据: {train_file}")
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    print(f"  训练样本: {len(train_data)} 条")

    print("  开始训练...")
    model = lazyllm.TrainableModule(SFT_BASE_MODEL, target_path=checkpoint_dir)\\
        .mode('finetune')\\
        .trainset(train_file)\\
        .finetune_method((finetune.llamafactory, {
            'learning_rate': SFT_LEARNING_RATE,
            'cutoff_len': 4096,
            'max_samples': 10000,
            'val_size': 0.01,
            'optim': 'adamw_torch_fused',
            'bf16': True,
            'fp16': False,
            'per_device_train_batch_size': SFT_BATCH_SIZE,
            'gradient_accumulation_steps': 4,
            'num_train_epochs': SFT_EPOCHS,
            'warmup_ratio': 0.1,
            'template': 'qwen',
            'stage': 'sft',
            'save_steps': 10,
            'resume_from_checkpoint': None,
            'save_strategy': 'steps',
            'save_total_limit': 3,
            'launcher': launchers.sco(ngpus=1, partition='a800'),
        }))

    model.update()
    print(f"  模型保存: {checkpoint_dir}")
STEP3

if [ $? -ne 0 ]; then
    log_error "步骤3失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤4: 推理测试 ============
log_step "[4/5] 运行推理测试..."

python3 << STEP4 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys

DATA_DIR = "$DATA_DIR"
OUTPUT_DIR = "$OUTPUT_DIR"
MODEL_DIR = "$MODEL_DIR"
LAZYLLM_PATH = "$LAZYLLM_PATH"

sys.path.insert(0, LAZYLLM_PATH)

import lazyllm
from lazyllm import deploy

eval_data_path = os.path.join(DATA_DIR, "tooluse_raw_eval.jsonl")
inference_output = os.path.join(OUTPUT_DIR, "inference_results.json")
model_path = os.path.join(MODEL_DIR, "tooluse_sft_checkpoint")

if os.path.exists(inference_output):
    print("  推理结果已存在，跳过推理")
else:
    SYS_PROMPT = """You are a helpful assistant with tool use capabilities.
When you need to use a tool, format your response as:
<tool>tool_name</tool>
<args>{"arg1": "value1", "arg2": "value2"}</args>

Available tools will be provided in the context."""

    # 加载评测数据
    print("  加载评测数据...")
    eval_prompts = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            eval_prompts.append(item.get('content', ''))

    print(f"  评测样本: {len(eval_prompts)} 条")

    # 加载模型
    print("  加载训练好的模型...")
    model = (
        lazyllm.TrainableModule(model_path)
        .prompt(dict(system=SYS_PROMPT, drop_builtin_system=True))
        .deploy_method((deploy.Vllm, {"max_num_seqs": 128}))
    )

    model.evalset(eval_prompts)
    model.start()
    model.eval()

    # 保存结果
    results = [{"test_case_id": i, "prompt": p, "response": r}
               for i, (p, r) in enumerate(zip(eval_prompts, model.eval_result))]

    with open(inference_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  推理结果保存: {inference_output}")
STEP4

if [ $? -ne 0 ]; then
    log_error "步骤4失败！详细错误请查看日志: $LOG_FILE"
    safe_exit 1
fi

# ============ 步骤5: 评测结果 ============
log_step "[5/5] 运行 Tool Use 评测..."

python3 << STEP5 2>&1 | tee -a "$LOG_FILE"
import json
import os
import sys
import re
import time

DATA_DIR = "$DATA_DIR"
OUTPUT_DIR = "$OUTPUT_DIR"
LAZYLLM_PATH = "$LAZYLLM_PATH"
JUDGE_MODEL = "$JUDGE_MODEL"

sys.path.insert(0, LAZYLLM_PATH)

inference_output = os.path.join(OUTPUT_DIR, "inference_results.json")
eval_report_path = os.path.join(OUTPUT_DIR, "tooluse_evaluation.json")

if os.path.exists(eval_report_path):
    print("  评测报告已存在，跳过评测")
    with open(eval_report_path, 'r') as f:
        report = json.load(f)
    print(f"\\n  总分平均分: {report['summary']['avg_total_score']:.2f} / 20")
    print(f"  满分完美比例: {report['summary']['perfect_rate']:.2f}%")
else:
    import lazyllm

    # 评测模板
    JUDGE_PROMPT_TEMPLATE = """You are a strict AI model evaluation expert. Your task is to evaluate a small model's performance on "Tool-use" tasks.

### Evaluation Context:
1. **User Input**: {user_input}
2. **Model Output (Prediction)**: {pred_output}

### Scoring Dimensions (1-5 points each):
1. **Format Correctness**: Is the output valid JSON or required text format?
2. **Tool Selection**: Did it select the correct tool? Or correctly identify no tool is needed?
3. **Argument Accuracy**: Are extracted parameters accurate? Are key information from Input included?
4. **Logic Reasoning**: Is the thinking process reasonable? Any hallucinations?

### Output Requirements:
Return JSON format only, no extra explanation:
{{
    "format_score": 1-5,
    "tool_score": 1-5,
    "arg_score": 1-5,
    "logic_score": 1-5,
    "total_score": sum of above,
    "reason": "Brief explanation for deductions (or 'Perfect' if full marks)"
}}"""

    # 加载裁判模型
    print("  加载裁判模型...")
    judge_model = lazyllm.TrainableModule(JUDGE_MODEL).start()

    def call_judge(prompt):
        """调用裁判模型"""
        response = judge_model(prompt)
        try:
            result = json.loads(response)
            return result
        except:
            json_match = re.search(r'\\{.*?\\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
        return {
            "format_score": 0, "tool_score": 0, "arg_score": 0, "logic_score": 0,
            "total_score": 0, "reason": "Parse failed"
        }

    # 加载推理结果
    print("  加载推理结果...")
    with open(inference_output, 'r') as f:
        inference_results = json.load(f)

    print(f"  评测样本: {len(inference_results)} 条")

    # 评测
    metrics = {"total": 0, "format": 0, "tool": 0, "arg": 0, "logic": 0, "perfect": 0}
    details = []

    for i, item in enumerate(inference_results):
        user_input = item.get('prompt', '')
        pred_output = item.get('response', '')

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            user_input=user_input,
            pred_output=pred_output
        )

        eval_result = call_judge(prompt)

        metrics["total"] += 1
        metrics["format"] += eval_result.get("format_score", 0)
        metrics["tool"] += eval_result.get("tool_score", 0)
        metrics["arg"] += eval_result.get("arg_score", 0)
        metrics["logic"] += eval_result.get("logic_score", 0)

        if eval_result.get("total_score", 0) == 20:
            metrics["perfect"] += 1

        details.append({
            "test_case_id": item.get("test_case_id", i),
            "input": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "evaluation": eval_result
        })

        if (i + 1) % 10 == 0:
            print(f"    已评测: {i+1}/{len(inference_results)}")

        time.sleep(0.05)

    # 计算平均分
    count = metrics["total"]
    avg_format = metrics["format"] / count if count > 0 else 0
    avg_tool = metrics["tool"] / count if count > 0 else 0
    avg_arg = metrics["arg"] / count if count > 0 else 0
    avg_logic = metrics["logic"] / count if count > 0 else 0
    avg_total = (avg_format + avg_tool + avg_arg + avg_logic)
    perfect_rate = metrics["perfect"] / count * 100 if count > 0 else 0

    print(f"\\n{'='*50}")
    print("--- Tool Use 评测报告 ---")
    print(f"{'='*50}")
    print(f"总样本量: {count}")
    print(f"平均格式得分: {avg_format:.2f} / 5")
    print(f"平均工具选择得分: {avg_tool:.2f} / 5")
    print(f"平均参数准确得分: {avg_arg:.2f} / 5")
    print(f"平均逻辑合理性得分: {avg_logic:.2f} / 5")
    print(f"总分平均分: {avg_total:.2f} / 20 ({avg_total/20*100:.2f}%)")
    print(f"满分完美比例: {perfect_rate:.2f}%")
    print(f"{'='*50}")

    # 保存报告
    report = {
        "summary": {
            "total": count,
            "avg_format_score": avg_format,
            "avg_tool_score": avg_tool,
            "avg_arg_score": avg_arg,
            "avg_logic_score": avg_logic,
            "avg_total_score": avg_total,
            "perfect_rate": perfect_rate
        },
        "details": details
    }

    with open(eval_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\\n  评测报告保存: {eval_report_path}")

    judge_model.stop()
STEP5

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
log "结果汇总:"
log "  数据目录: $DATA_DIR"
log "  模型目录: $MODEL_DIR/tooluse_sft_checkpoint"
log "  推理结果: $OUTPUT_DIR/inference_results.json"
log "  评测报告: $OUTPUT_DIR/tooluse_evaluation.json"
log "  日志文件: $LOG_FILE"
log ""

# 显示评测结果
if [ -f "$OUTPUT_DIR/tooluse_evaluation.json" ]; then
    python3 << RESULT 2>&1 | tee -a "$LOG_FILE"
import json
with open("$OUTPUT_DIR/tooluse_evaluation.json") as f:
    report = json.load(f)
s = report['summary']
print("评测指标:")
print(f"  - 总样本: {s['total']}")
print(f"  - 格式得分: {s['avg_format_score']:.2f}/5")
print(f"  - 工具选择: {s['avg_tool_score']:.2f}/5")
print(f"  - 参数准确: {s['avg_arg_score']:.2f}/5")
print(f"  - 逻辑合理: {s['avg_logic_score']:.2f}/5")
print(f"  - 总分: {s['avg_total_score']:.2f}/20 ({s['avg_total_score']/20*100:.1f}%)")
print(f"  - 满分率: {s['perfect_rate']:.1f}%")
RESULT
fi

log ""
log "=========================================="
