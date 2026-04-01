#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import run as base

SFT_MODEL = base.SFT_MODEL

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
LOG_DIR = BASE_DIR / 'logs'

CONFIG = {}


def step1_prepare_data():
    base.log_step('[1/4] 下载并准备 tiny-codes 数据集...')

    train_path = base.DATA_DIR / 'train_python.json'
    eval_path = base.DATA_DIR / 'eval_python.json'

    if train_path.exists() and eval_path.exists():
        base.log('  数据已存在，跳过下载')
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        base.log_error('请先安装 datasets: pip install datasets')
        return False

    base.log('  正在从 Hugging Face 加载数据集...')
    try:
        dataset = load_dataset(
            'nampdn-ai/tiny-codes', split='train', streaming=True
        )
    except Exception as exc:
        base.log_error(f'加载数据集失败: {exc}')
        base.log('')
        base.log_info('提示: 如果遇到权限问题，请尝试以下方法:')
        base.log('  1. 登录 Hugging Face: huggingface-cli login')
        base.log('  2. 或设置环境变量: export HF_TOKEN=your_token')
        base.log('  3. 或手动下载数据集并放置到 data/ 目录')
        base.log('')
        return False

    python_data = []
    target_count = 6000

    for entry in dataset:
        if entry.get('programming_language', '').lower() != 'python':
            continue

        prompt = entry.get('prompt', '').strip()
        response = entry.get('response', '').strip()
        if len(prompt) <= 5 or len(response) <= 10:
            continue

        python_data.append(
            {'instruction': prompt, 'input': '', 'output': response}
        )
        if len(python_data) % 500 == 0:
            base.log(f'    已收集 {len(python_data)} 条...')
        if len(python_data) >= target_count:
            break

    train_data = python_data[:5000]
    eval_data = python_data[5000:6000]

    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, indent=2, ensure_ascii=False)
    with open(eval_path, 'w', encoding='utf-8') as file:
        json.dump(eval_data, file, indent=2, ensure_ascii=False)

    base.log(f'  SFT训练集: {train_path} ({len(train_data)} 条)')
    base.log(f'  评测集: {eval_path} ({len(eval_data)} 条)')
    return True


def step2_sft_training():
    base.log_step('[2/4] 直接使用步骤1数据开始 SFT 训练...')

    train_file = base.DATA_DIR / 'train_python.json'
    checkpoint_dir = base.MODEL_DIR / 'checkpoint'

    if not train_file.exists():
        base.log_error(f'SFT 训练集不存在: {train_file}')
        return False

    if checkpoint_dir.exists():
        base.log('  模型已存在，跳过训练')
        return True

    local_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

    import lazyllm
    from lazyllm import finetune, launchers

    model = (
        lazyllm.TrainableModule(
            CONFIG['sft_model'], target_path=str(checkpoint_dir)
        )
        .mode('finetune')
        .trainset(str(train_file))
        .finetune_method(
            (
                finetune.llamafactory,
                {
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
                },
            )
        )
    )

    model.update()
    base.log(f'  训练数据: {train_file}')
    base.log(f'  模型保存: {checkpoint_dir}')
    return True


def step3_inference():
    return base.step4_inference('[3/4]')


def step4_evaluation():
    return base.step5_evaluation('[4/4]')


def parse_args():
    parser = argparse.ArgumentParser(description='无 pipeline 的一键代码SFT训练脚本')
    parser.add_argument(
        '--sft-model',
        type=str,
        default=None,
        help='SFT基础模型路径 (默认: /path/to/sft/base/model)',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据目录 (默认: 脚本所在目录/data)',
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='模型目录 (默认: 脚本所在目录/models)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录 (默认: 脚本所在目录/output)',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='日志目录 (默认: 脚本所在目录/logs)',
    )
    parser.add_argument(
        '--lazyllm-path',
        type=str,
        default=None,
        help='兼容旧参数，run_sft.py 中已不再使用',
    )
    parser.add_argument(
        '--pipeline-model',
        type=str,
        default=None,
        help='兼容旧参数，run_sft.py 中已不再使用',
    )
    parser.add_argument(
        '--vllm-max-model-len',
        type=int,
        default=None,
        help='推理阶段 vLLM 的 max_model_len',
    )
    parser.add_argument(
        '--vllm-gpu-memory-utilization',
        type=float,
        default=None,
        help='推理阶段 vLLM 的 gpu_memory_utilization',
    )
    parser.add_argument(
        '--vllm-max-num-seqs',
        type=int,
        default=None,
        help='推理阶段 vLLM 的 max_num_seqs',
    )
    parser.add_argument(
        '--vllm-max-num-batched-tokens',
        type=int,
        default=None,
        help='推理阶段 vLLM 的 max_num_batched_tokens',
    )
    parser.add_argument(
        '--vllm-response-max-tokens',
        type=int,
        default=None,
        help='单次推理响应的最大 token 数',
    )
    parser.add_argument(
        '--inference-workers',
        type=int,
        default=None,
        help='推理阶段并发数',
    )
    parser.add_argument(
        '--eval-workers',
        type=int,
        default=None,
        help='评测阶段并发数',
    )
    parser.add_argument(
        '--skip-steps',
        type=str,
        default='',
        help='跳过的步骤，用逗号分隔，如 "1,3" 跳过步骤1和3',
    )
    parser.add_argument(
        '--only-step', type=int, default=None, help='只运行指定步骤 (1-4)'
    )
    return parser.parse_args()


def init_config(args):
    global CONFIG
    CONFIG = base.init_config(args)
    return CONFIG


def main():
    args = parse_args()
    config = init_config(args)

    base.LOG_FILE = config['log_dir'] / (
        'run_sft_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    )

    base.log('==========================================')
    base.log('无 pipeline 的一键代码SFT训练脚本')
    base.log('==========================================')
    base.log('')
    base.log_info('配置信息:')
    base.log(f'  - 基础目录: {BASE_DIR}')
    base.log(f"  - SFT模型: {config['sft_model']}")
    base.log(f"  - 数据目录: {config['data_dir']}")
    base.log(f"  - 模型目录: {config['model_dir']}")
    base.log(f"  - 输出目录: {config['output_dir']}")
    base.log(f"  - 日志目录: {config['log_dir']}")
    base.log(f"  - VLLM_MAX_MODEL_LEN: {config['vllm_max_model_len']}")
    base.log(
        '  - VLLM_GPU_MEMORY_UTILIZATION: '
        f"{config['vllm_gpu_memory_utilization']}"
    )
    base.log(f"  - VLLM_MAX_NUM_SEQS: {config['vllm_max_num_seqs']}")
    base.log(
        '  - VLLM_MAX_NUM_BATCHED_TOKENS: '
        f"{config['vllm_max_num_batched_tokens']}"
    )
    base.log(
        '  - VLLM_RESPONSE_MAX_TOKENS: '
        f"{config['vllm_response_max_tokens']}"
    )
    base.log(f"  - INFERENCE_WORKERS: {config['inference_workers']}")
    base.log(f"  - EVAL_WORKERS: {config['eval_workers']}")
    base.log(f'  - 日志文件: {base.LOG_FILE}')
    base.log('')

    if args.lazyllm_path or args.pipeline_model:
        base.log_info(
            'run_sft.py 已移除 pipeline，--lazyllm-path 和 '
            '--pipeline-model 参数会被忽略'
        )

    skip_steps = set()
    if args.skip_steps:
        skip_steps = set(
            int(item.strip())
            for item in args.skip_steps.split(',')
            if item.strip().isdigit()
        )

    steps = [
        ('数据准备', step1_prepare_data),
        ('SFT训练', step2_sft_training),
        ('评测推理', step3_inference),
        ('代码评估', step4_evaluation),
    ]

    for index, (name, step_func) in enumerate(steps, 1):
        if args.only_step is not None and index != args.only_step:
            continue
        if index in skip_steps:
            base.log_info(f'跳过步骤{index}: {name}')
            continue

        try:
            if not step_func():
                base.log_error(
                    f'步骤{index}失败！详细错误请查看日志: {base.LOG_FILE}'
                )
                base.safe_exit(1)
        except Exception as exc:
            base.log_error(f'步骤{index}异常: {type(exc).__name__}: {exc}')
            base.safe_exit(1)

    base.log('')
    base.log('==========================================')
    base.log('全部完成!')
    base.log('==========================================')
    base.log('')
    base.log_info('结果汇总:')
    base.log(f"  数据目录: {config['data_dir']}")
    base.log(f"  模型目录: {config['model_dir']}/checkpoint")
    base.log(f"  推理结果: {config['output_dir']}/inference_results.json")
    base.log(f"  评估报告: {config['output_dir']}/evaluation_report.csv")
    base.log(f'  日志文件: {base.LOG_FILE}')
    base.log('')

    report_file = config['output_dir'] / 'evaluation_report.csv'
    if report_file.exists():
        base.log_info('评估统计:')
        with open(report_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            statuses = [row['status'] for row in reader]
            summary = Counter(statuses)
            total = len(statuses)
            pass_count = summary.get('Pass', 0)
            pass_ratio = (pass_count / total * 100) if total else 0.0
            base.log(f'  - 总样本: {total}')
            base.log(f'  - 通过: {pass_count} ({pass_ratio:.1f}%)')
            base.log(f"  - 失败: {summary.get('Fail', 0)}")
            base.log(f"  - 语法错误: {summary.get('SyntaxError', 0)}")
            base.log(f"  - 无代码: {summary.get('NoCode', 0)}")
            base.log(f"  - 错误: {summary.get('Error', 0)}")

    base.log('')
    base.log('==========================================')
    base.safe_exit(0)


if __name__ == '__main__':
    main()
