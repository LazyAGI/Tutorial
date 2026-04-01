#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

import run_text2sql_pipeline as base

SFT_BASE_MODEL = base.SFT_BASE_MODEL

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
LOG_DIR = BASE_DIR / 'logs'

CONFIG = {}


def step1_prepare_data():
    return base.step1_prepare_data('[1/4]')


def step2_sft_training():
    base.log_step('[2/4] 开始 SFT 训练...')

    train_file = base.DATA_DIR / 'train_text2sql.json'
    checkpoint_dir = base.MODEL_DIR / 'checkpoint'

    if not train_file.exists():
        base.log_error(f'SFT 训练集不存在: {train_file}')
        return False

    if checkpoint_dir.exists():
        base.log('  模型已存在，跳过训练')
        return True

    base.ensure_local_site_packages()

    import lazyllm
    from lazyllm import finetune, launchers

    model = (
        lazyllm.TrainableModule(
            CONFIG['sft_base_model'], target_path=str(checkpoint_dir)
        )
        .mode('finetune')
        .trainset(str(train_file))
        .finetune_method(
            (
                finetune.llamafactory,
                {
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
    parser = argparse.ArgumentParser(
        description='无 Pipeline 的一键 Text2SQL 训练脚本'
    )
    parser.add_argument(
        '--lazyllm-path',
        type=str,
        default=None,
        help='LazyLLM 库路径',
    )
    parser.add_argument(
        '--pipeline-model',
        type=str,
        default=None,
        help='兼容旧参数，run_sft.py 中已不再使用',
    )
    parser.add_argument(
        '--sft-base-model',
        type=str,
        default=None,
        help='SFT 基础模型路径',
    )
    parser.add_argument(
        '--judge-model',
        type=str,
        default=None,
        help='Judge 模型路径',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据目录',
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='模型目录',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='日志目录',
    )
    parser.add_argument(
        '--pipeline-limit',
        type=int,
        default=None,
        help='兼容旧参数，run_sft.py 中已不再使用',
    )
    parser.add_argument(
        '--judge-workers',
        type=int,
        default=None,
        help='评估阶段并发数',
    )
    parser.add_argument(
        '--judge-max-model-len',
        type=int,
        default=None,
        help='Judge vLLM 的 max_model_len',
    )
    parser.add_argument(
        '--judge-gpu-memory-utilization',
        type=float,
        default=None,
        help='Judge vLLM 的 gpu_memory_utilization',
    )
    parser.add_argument(
        '--judge-max-num-seqs',
        type=int,
        default=None,
        help='Judge vLLM 的 max_num_seqs',
    )
    parser.add_argument(
        '--judge-response-max-tokens',
        type=int,
        default=None,
        help='Judge 单次响应最大 token 数',
    )
    parser.add_argument(
        '--skip-steps',
        type=str,
        default='',
        help='跳过的步骤，用逗号分隔，如 "1,3"',
    )
    parser.add_argument(
        '--only-step',
        type=int,
        default=None,
        help='只运行指定步骤 (1-4)',
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
    base.log('无 Pipeline 的一键 Text2SQL 训练脚本')
    base.log('==========================================')
    base.log('')
    base.log_info('配置信息:')
    base.log(f'  - 基础目录: {BASE_DIR}')
    base.log(f"  - LazyLLM路径: {config['lazyllm_path']}")
    base.log(f"  - SFT模型: {config['sft_base_model']}")
    base.log(f"  - Judge模型: {config['judge_model']}")
    base.log(f"  - 数据目录: {config['data_dir']}")
    base.log(f"  - 模型目录: {config['model_dir']}")
    base.log(f"  - 输出目录: {config['output_dir']}")
    base.log(f"  - 日志目录: {config['log_dir']}")
    base.log(f"  - JUDGE_WORKERS: {config['judge_workers']}")
    base.log(f"  - JUDGE_MAX_MODEL_LEN: {config['judge_max_model_len']}")
    base.log(
        '  - JUDGE_GPU_MEMORY_UTILIZATION: '
        f"{config['judge_gpu_memory_utilization']}"
    )
    base.log(f"  - JUDGE_MAX_NUM_SEQS: {config['judge_max_num_seqs']}")
    base.log(
        '  - JUDGE_RESPONSE_MAX_TOKENS: '
        f"{config['judge_response_max_tokens']}"
    )
    base.log(f'  - 日志文件: {base.LOG_FILE}')
    base.log('')

    if args.pipeline_model is not None or args.pipeline_limit is not None:
        base.log_info(
            'run_sft.py 已移除 Pipeline，--pipeline-model 和 '
            '--pipeline-limit 参数会被忽略'
        )

    skip_steps = set()
    if args.skip_steps:
        skip_steps = set(
            int(item.strip())
            for item in args.skip_steps.split(',')
            if item.strip().isdigit()
        )

    steps = [
        ('下载数据', step1_prepare_data),
        ('SFT训练', step2_sft_training),
        ('评测推理', step3_inference),
        ('Text2SQL评估', step4_evaluation),
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
    base.log(f"  评估报告: {config['output_dir']}/evaluation_report.json")
    base.log(f'  日志文件: {base.LOG_FILE}')
    base.log('')

    report_file = config['output_dir'] / 'evaluation_report.json'
    if report_file.exists():
        base.log_info('评估统计:')
        with open(report_file, 'r', encoding='utf-8') as file:
            report = json.load(file)
        summary = report['summary']
        base.log(f"  - 总样本: {summary['total_samples']}")
        base.log(f"  - 平均总分: {summary['avg_overall_score']:.2f}/5.0")
        base.log(f"  - 语义得分: {summary['avg_semantic_score']:.2f}/5.0")
        base.log(f"  - 语法得分: {summary['avg_syntax_score']:.2f}/5.0")
        base.log(f"  - 等价得分: {summary['avg_equivalence_score']:.2f}/3.0")
        base.log(f"  - 正确率: {summary['accuracy'] * 100:.1f}%")

    base.log('')
    base.log('==========================================')
    base.safe_exit(0)


if __name__ == '__main__':
    main()
