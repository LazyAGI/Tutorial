#!/usr/bin/env python3
'''
DPO训练脚本 (基于PKU-SafeRLHF数据)

用法:
    python run_dpo.py [选项]

示例:
    python run_dpo.py --dpo-base-model /path/to/model
    python run_dpo.py --skip-steps 1 --dpo-base-model /path/to/model
'''

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ============ 配置路径 ============
LAZYLLM_PATH = '/path/to/your/lazyllm'
DPO_BASE_MODEL = '/path/to/dpo/base/model'

# ============ 目录设置 ============
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
LOG_DIR = BASE_DIR / 'logs'

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# ============ 日志工具 ============
def log(msg: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f'[{timestamp}] {msg}'
    print(formatted)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')

def log_error(msg: str):
    log(f'[ERROR] {msg}')

def log_info(msg: str):
    log(f'[INFO] {msg}')

def log_step(msg: str):
    log(f'[STEP] {msg}')

def safe_exit(code: int = 0):
    if code != 0:
        log_error(f'脚本异常退出，退出码: {code}')
    else:
        log_info('脚本正常完成')
    sys.exit(code)


# ============ 步骤1: 下载并准备数据 ============
def step1_prepare_data():
    log_step('[1/4] 下载并准备 PKU-SafeRLHF 数据集...')

    train_path = DATA_DIR / 'train_dpo.json'
    eval_path = DATA_DIR / 'eval_dpo.json'

    if train_path.exists() and eval_path.exists():
        log('  数据已存在，跳过下载')
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        log_error('请先安装 datasets: pip install datasets')
        return False

    log('  正在从 Hugging Face 加载 PKU-SafeRLHF...')
    try:
        ds = load_dataset('PKU-Alignment/PKU-SafeRLHF', trust_remote_code=True)
    except Exception as e:
        log_error(f'加载数据集失败: {e}')
        log('')
        log_info('提示: 如果遇到权限问题，请尝试以下方法:')
        log('  1. 登录 Hugging Face: huggingface-cli login')
        log('  2. 或设置环境变量: export HF_TOKEN=your_token')
        log('  3. 或手动下载数据集并放置到 data/ 目录')
        log('')
        return False

    # 将数据转换为 DPO 偏好对格式
    def convert_to_dpo(example):
        r0 = example.get('response_0', '')
        r1 = example.get('response_1', '')
        prompt = example.get('prompt', '')
        chosen_idx = example.get(
            'safer_response_id',
            example.get('better_response_id', 0),
        )
        return {
            'prompt': prompt,
            'chosen': r0 if chosen_idx == 0 else r1,
            'rejected': r1 if chosen_idx == 0 else r0,
        }

    dpo_dataset = ds['train'].map(
        convert_to_dpo,
        remove_columns=ds['train'].column_names,
    )
    data_list = list(dpo_dataset)

    # 划分训练集和验证集
    train_data = data_list[:9000]
    eval_data = data_list[9000:10000]

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2)

    log(f'  训练集: {train_path} ({len(train_data)} 条)')
    log(f'  验证集: {eval_path} ({len(eval_data)} 条)')
    return True


# ============ 步骤2: DPO训练 ============
def step2_dpo_training():
    log_step('[2/4] 开始 DPO 训练...')

    local_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

    import lazyllm
    from lazyllm import finetune, launchers

    train_file = DATA_DIR / 'train_dpo.json'
    checkpoint_dir = MODEL_DIR / 'dpo_checkpoint'

    if checkpoint_dir.exists():
        log('  模型已存在，跳过训练')
        return True

    log('  DPO训练参数:')
    log('    - 学习率: 5e-6')
    log('    - 批次大小: 8')
    log('    - 训练轮数: 2.0')
    log('    - 模板: qwen')
    log('    - DPO Beta: 0.1')

    model = lazyllm.TrainableModule(DPO_BASE_MODEL, target_path=str(checkpoint_dir))\
        .mode('finetune')\
        .trainset(str(train_file))\
        .finetune_method((finetune.llamafactory, {
            'learning_rate': 5e-6,
            'cutoff_len': 2048,
            'max_samples': 10000,
            'val_size': 0.1,
            'optim': 'adamw_torch_fused',
            'bf16': True,
            'fp16': False,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'num_train_epochs': 2.0,
            'template': 'qwen',
            'stage': 'dpo',
            'dpo_beta': 0.1,
            'save_steps': 100,
            'save_total_limit': 2,
            'launcher': launchers.sco(ngpus=1, partition='a800'),
        }))

    model.update()
    log(f'  模型保存: {checkpoint_dir}')
    return True


# ============ 步骤3: 评测集推理 ============
def step3_inference():
    log_step('[3/4] 运行评测集推理...')

    local_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

    import lazyllm
    from lazyllm import deploy

    eval_file = DATA_DIR / 'eval_dpo.json'
    inference_output = OUTPUT_DIR / 'inference_results.json'

    # 自动查找最新的 lazyllm_merge 目录
    def find_latest_merge_model(base_dir):
        merge_dirs = []
        for root, dirs, _ in os.walk(base_dir):
            if 'lazyllm_merge' in dirs:
                path = Path(root) / 'lazyllm_merge'
                try:
                    merge_dirs.append((path, path.stat().st_mtime))
                except OSError:
                    pass
        return max(merge_dirs, key=lambda x: x[1])[0] if merge_dirs else None

    model_path = find_latest_merge_model(MODEL_DIR)
    if not model_path:
        log(f'  错误: 在 {MODEL_DIR} 下未找到 lazyllm_merge 目录')
        return False
    log(f'  找到模型: {model_path}')

    if inference_output.exists():
        log('  推理结果已存在，跳过推理')
        return True

    log('  加载评测数据...')
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    log(f'  评测样本: {len(eval_data)} 条')

    log('  加载训练好的模型...')
    model = lazyllm.TrainableModule(str(model_path)).deploy_method(deploy.vllm)
    model.start()

    log('  开始推理...')
    results = []
    for i, item in enumerate(eval_data):
        prompt = item.get('prompt', '')
        reference_chosen = item.get('chosen', '')
        reference_rejected = item.get('rejected', '')

        response = model(prompt)

        results.append({
            'id': i,
            'prompt': prompt,
            'reference_chosen': reference_chosen,
            'reference_rejected': reference_rejected,
            'prediction': response,
        })

        if (i + 1) % 10 == 0:
            log(f'    已处理: {i+1}/{len(eval_data)}')

    with open(inference_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log(f'  推理完成: {inference_output}')
    model.stop()
    return True


# ============ 步骤4: 简单评估 ============
def step4_evaluation():
    log_step('[4/4] 运行评估...')

    inference_file = OUTPUT_DIR / 'inference_results.json'
    report_path = OUTPUT_DIR / 'evaluation_report.json'

    if report_path.exists():
        log('  评估报告已存在，跳过评估')
        return True

    with open(inference_file, 'r', encoding='utf-8') as f:
        inference_data = json.load(f)

    log(f'  评估 {len(inference_data)} 条推理结果...')

    # 简单统计：计算非空回复比例
    valid_count = sum(1 for item in inference_data if item.get('prediction', '').strip())
    total = len(inference_data)

    report = {
        'summary': {
            'total': total,
            'valid_responses': valid_count,
            'valid_rate': valid_count / total * 100 if total > 0 else 0
        },
        'details': inference_data
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    log(f'  评估完成: {report_path}')
    log(f'  有效回复率: {valid_count}/{total} ({valid_count/total*100:.2f}%)')
    return True


# ============ 命令行参数解析 ============
def parse_args():
    parser = argparse.ArgumentParser(description='DPO训练脚本')
    parser.add_argument('--lazyllm-path', type=str, default=None,
                        help='LazyLLM库路径')
    parser.add_argument('--dpo-base-model', type=str, default=None,
                        help='DPO基础模型路径')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='模型目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--skip-steps', type=str, default='',
                        help='跳过的步骤，用逗号分隔，如 "1" 跳过数据下载')
    parser.add_argument('--only-step', type=int, default=None,
                        help='只运行指定步骤 (1-4)')
    return parser.parse_args()


# ============ 主函数 ============
def apply_cli_overrides(args):
    global LAZYLLM_PATH, DPO_BASE_MODEL, DATA_DIR, MODEL_DIR, OUTPUT_DIR

    if args.lazyllm_path:
        LAZYLLM_PATH = args.lazyllm_path
    if args.dpo_base_model:
        DPO_BASE_MODEL = args.dpo_base_model
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.model_dir:
        MODEL_DIR = Path(args.model_dir)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)


def setup_runtime_paths():
    global LOG_FILE

    for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'


def log_configuration():
    log('==========================================')
    log('DPO训练脚本')
    log('==========================================')
    log('')
    log_info('配置信息:')
    log(f'  - LazyLLM路径: {LAZYLLM_PATH}')
    log(f'  - DPO基础模型: {DPO_BASE_MODEL}')
    log(f'  - 数据目录: {DATA_DIR}')
    log(f'  - 模型目录: {MODEL_DIR}')
    log(f'  - 输出目录: {OUTPUT_DIR}')
    log(f'  - 日志文件: {LOG_FILE}')
    log('')


def parse_skip_steps(skip_steps_arg):
    if not skip_steps_arg:
        return set()
    return {
        int(x.strip())
        for x in skip_steps_arg.split(',')
        if x.strip().isdigit()
    }


def get_pipeline_steps():
    return [
        ('数据准备', step1_prepare_data),
        ('DPO训练', step2_dpo_training),
        ('评测推理', step3_inference),
        ('评估', step4_evaluation),
    ]


def run_pipeline_steps(steps, skip_steps, only_step):
    for i, (name, step_func) in enumerate(steps, 1):
        if only_step is not None and i != only_step:
            continue
        if i in skip_steps:
            log_info(f'跳过步骤{i}: {name}')
            continue

        try:
            if not step_func():
                log_error(f'步骤{i}失败！')
                return False
        except Exception as e:
            import traceback
            log_error(f'步骤{i}异常: {type(e).__name__}: {e}')
            log_error(traceback.format_exc())
            return False
    return True


def log_final_summary():
    log('')
    log('==========================================')
    log('全部完成!')
    log('==========================================')
    log('')
    log_info('结果汇总:')
    log(f'  数据目录: {DATA_DIR}')
    log(f'  模型目录: {MODEL_DIR}/dpo_checkpoint')
    log(f'  推理结果: {OUTPUT_DIR}/inference_results.json')
    log(f'  评估报告: {OUTPUT_DIR}/evaluation_report.json')
    log(f'  日志文件: {LOG_FILE}')
    log('')
    log('==========================================')


def main():
    args = parse_args()
    apply_cli_overrides(args)
    setup_runtime_paths()
    log_configuration()

    skip_steps = parse_skip_steps(args.skip_steps)
    steps = get_pipeline_steps()
    success = run_pipeline_steps(steps, skip_steps, args.only_step)
    if not success:
        safe_exit(1)

    log_final_summary()
    safe_exit(0)


if __name__ == '__main__':
    main()
