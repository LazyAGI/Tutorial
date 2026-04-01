#!/usr/bin/env python3
'''
一键代码SFT训练脚本
'''

import ast
import csv
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

LAZYLLM_PATH = '/path/to/your/lazyllm'
PIPELINE_MODEL = '/path/to/pipeline/model'
SFT_MODEL = '/path/to/sft/base/model'
VLLM_MAX_MODEL_LEN = 4096
VLLM_GPU_MEMORY_UTILIZATION = 0.8
VLLM_MAX_NUM_SEQS = 16
VLLM_MAX_NUM_BATCHED_TOKENS = 16384
VLLM_RESPONSE_MAX_TOKENS = 1536
INFERENCE_WORKERS = 8
EVAL_WORKERS = 8

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
LOG_DIR = BASE_DIR / 'logs'

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / (
    'run_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
)


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


def step1_download_data():
    log_step('[1/5] 下载 tiny-codes 数据集...')

    train_path = DATA_DIR / 'train_python.json'
    eval_path = DATA_DIR / 'eval_python.json'

    if train_path.exists() and eval_path.exists():
        log('  数据已存在，跳过下载')
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        log_error('请先安装 datasets: pip install datasets')
        return False

    log('  正在从 Hugging Face 加载数据集...')
    try:
        ds = load_dataset(
            'nampdn-ai/tiny-codes', split='train', streaming=True
        )
    except Exception as e:
        log_error(f'加载数据集失败: {e}')
        log('')
        log_info('提示: 如果遇到权限问题，请尝试以下方法:')
        log('  1. 登录 Hugging Face: huggingface-cli login')
        log('  2. 或设置环境变量: export HF_TOKEN=your_token')
        log('  3. 或手动下载数据集并放置到 data/ 目录')
        log('')
        return False

    python_data = []
    target_count = 6000

    for entry in ds:
        if entry.get('programming_language', '').lower() == 'python':
            prompt = entry.get('prompt', '').strip()
            response = entry.get('response', '').strip()
            if len(prompt) > 5 and len(response) > 10:
                python_data.append(
                    {'instruction': prompt, 'input': '', 'output': response}
                )
            if len(python_data) % 500 == 0:
                log(f'    已收集 {len(python_data)} 条...')
            if len(python_data) >= target_count:
                break

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(python_data[:5000], f, indent=2)
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(python_data[5000:6000], f, indent=2)

    log(f'  训练集: {len(python_data[:5000])} 条')
    log(f'  验证集: {len(python_data[5000:6000])} 条')
    return True


# ============ 步骤2: 数据处理Pipeline ============
def step2_data_pipeline():
    log_step('[2/5] 运行数据增强 pipeline...')

    if not Path(LAZYLLM_PATH).exists():
        log_error(f'LAZYLLM_PATH 不存在: {LAZYLLM_PATH}')
        log('请修改脚本中的 LAZYLLM_PATH 配置')
        return False

    sys.path.insert(0, LAZYLLM_PATH)
    sys.path.insert(0, str(Path(LAZYLLM_PATH).parent))

    import lazyllm
    from lazyllm.tools.data.pipelines.codegen_pipelines import (
        build_codegen_pipeline,
    )

    input_file = DATA_DIR / 'train_python.json'
    output_file = DATA_DIR / 'codegen.json'

    if output_file.exists():
        log('  Pipeline 输出已存在，跳过处理')
        return True

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    log(f'  加载数据: {len(data)} 条')

    model = lazyllm.TrainableModule(PIPELINE_MODEL)
    model.start()

    ppl = build_codegen_pipeline(
        model=model, input_key='messages', min_score=8, max_score=10
    )

    formatted_data = []
    for item in data:
        messages = [
            {
                'role': 'system',
                'content': 'You are an expert Python programmer.',
            },
            {'role': 'user', 'content': item['instruction']},
        ]
        formatted_data.append({'messages': messages, 'metadata': {}})

    log(f'  批量处理 {len(formatted_data)} 条数据...')

    results = ppl(formatted_data)

    output_data = []
    for result in results:
        if result:
            output_data.append(
                {
                    'instruction': result.get('instruction', ''),
                    'input': result.get('input', ''),
                    'output': result.get('output', ''),
                }
            )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    model.stop()
    log(f'  生成数据: {len(output_data)} 条')
    return True


# ============ 步骤3: SFT训练 ============
def step3_sft_training():
    log_step('[3/5] 开始 SFT 训练...')

    local_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

    import lazyllm
    from lazyllm import finetune, launchers

    train_file = DATA_DIR / 'codegen.json'
    checkpoint_dir = MODEL_DIR / 'checkpoint'

    if checkpoint_dir.exists():
        log('  模型已存在，跳过训练')
        return True

    model = (
        lazyllm.TrainableModule(SFT_MODEL, target_path=str(checkpoint_dir))
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
    log(f'  模型保存: {checkpoint_dir}')
    return True


# ============ 步骤4: 评测集推理 ============
def find_latest_merge_model(base_dir):
    '''查找最新的合并模型目录.'''
    merge_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            if d == 'lazyllm_merge':
                full_path = Path(root) / d
                try:
                    merge_dirs.append((full_path, full_path.stat().st_mtime))
                except OSError:
                    pass
    if not merge_dirs:
        return None
    merge_dirs.sort(key=lambda x: x[1], reverse=True)
    return merge_dirs[0][0]


def load_eval_data(eval_file):
    '''加载评测数据.'''
    log('  加载评测数据...')
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    log(f'  评测样本: {len(eval_data)} 条')
    return eval_data


def run_inference(
    model,
    eval_data,
    sys_prompt,
    response_max_tokens,
    inference_workers,
):
    '''运行推理并返回结果.'''
    log('  开始推理...')
    if not eval_data:
        return []

    results = [None] * len(eval_data)

    def infer_single(i, item):
        prompt = item.get('instruction', '')
        reference = item.get('output', '')

        full_prompt = (
            f'{sys_prompt}\n\n### Problem:\n{prompt}\n\n### Solution:\n'
        )
        response = model(full_prompt, max_tokens=response_max_tokens)
        return i, {
            'id': i,
            'prompt': prompt,
            'reference': reference,
            'prediction': response,
        }

    num_workers = max(1, min(inference_workers, len(eval_data)))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(infer_single, i, item)
            for i, item in enumerate(eval_data)
        ]

        for done_count, future in enumerate(as_completed(futures), 1):
            idx, result = future.result()
            results[idx] = result
            if done_count % 10 == 0 or done_count == len(eval_data):
                log(f'    已处理: {done_count}/{len(eval_data)}')
    return results


def save_inference_results(results, inference_output):
    '''保存推理结果到文件.'''
    with open(inference_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f'  推理完成: {inference_output}')


def step4_inference():
    log_step('[4/5] 运行评测集推理...')

    local_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

    import lazyllm
    from lazyllm import deploy

    eval_file = DATA_DIR / 'eval_python.json'
    inference_output = OUTPUT_DIR / 'inference_results.json'

    model_path = find_latest_merge_model(MODEL_DIR)
    if not model_path:
        log(f'  错误: 在 {MODEL_DIR} 下未找到 lazyllm_merge 目录')
        return False
    log(f'  找到模型: {model_path}')

    if inference_output.exists():
        log('  推理结果已存在，跳过推理')
        return True

    eval_data = load_eval_data(eval_file)

    log('  加载训练好的模型...')
    model = lazyllm.TrainableModule(str(model_path)).deploy_method(
        (
            deploy.vllm,
            {
                'max_model_len': CONFIG['vllm_max_model_len'],
                'gpu_memory_utilization': CONFIG[
                    'vllm_gpu_memory_utilization'
                ],
                'max_num_seqs': CONFIG['vllm_max_num_seqs'],
                'max_num_batched_tokens': CONFIG[
                    'vllm_max_num_batched_tokens'
                ],
            },
        )
    )
    model.start()
    log(
        '  vLLM 配置: '
        f"max_model_len={CONFIG['vllm_max_model_len']}, "
        f"gpu_memory_utilization={CONFIG['vllm_gpu_memory_utilization']}, "
        f"max_num_seqs={CONFIG['vllm_max_num_seqs']}, "
        f"max_num_batched_tokens={CONFIG['vllm_max_num_batched_tokens']}, "
        f"response_max_tokens={CONFIG['vllm_response_max_tokens']}, "
        f"inference_workers={CONFIG['inference_workers']}"
    )

    try:
        sys_prompt = (
            'You are an expert Python programmer. Write clean, correct Python '
            'code to solve the given problem.'
        )

        results = run_inference(
            model,
            eval_data,
            sys_prompt,
            CONFIG['vllm_response_max_tokens'],
            CONFIG['inference_workers'],
        )
        save_inference_results(results, inference_output)
    finally:
        model.stop()
    return True


# ============ 步骤5: 评估 ============
def extract_code(text):
    '''从文本中提取代码块.'''
    code_start = r'(^|\n)\s*(def\s+|import\s+|class\s+)'
    for pattern in (r'```python\s*(.*?)\s*```', r'```\s*(.*?)\s*```'):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    for marker in ('### Solution:', 'Solution:'):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            match = re.search(code_start, text)
            return text[match.start():].strip() if match else text
    match = re.search(r'(.*?)专家反馈:', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    match = re.search(code_start, text)
    if match:
        return text[match.start():].strip()
    return None


def get_case_id(item):
    '''获取案例ID.'''
    return item.get('test_case_id', item.get('id', 0))


def get_case_response(item):
    '''获取案例响应.'''
    return item.get('response', item.get('prediction', ''))


def check_syntax(code):
    '''检查代码语法.'''
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def run_in_docker(code, case_id):
    '''在Docker中运行代码.'''
    file_name = f'tmp_run_{case_id}.py'
    lines = code.split('\n')
    indented = '\n'.join('        ' + line for line in lines)
    mock_wrapper = f'''
import unittest.mock
import sys
def mock_input(prompt=''):
    return '25'
with unittest.mock.patch('builtins.input', side_effect=mock_input):
    try:
{indented}
    except Exception as e:
        print(f'RUNTIME_ERROR: {{e}}', file=sys.stderr)
'''
    final_code = mock_wrapper if 'input(' in code else code
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('import sys\n' + final_code)

    try:
        result = subprocess.run(
            [
                'docker',
                'run',
                '--rm',
                '--network',
                'none',
                '--memory',
                '128m',
                '-v',
                f'{os.path.abspath(file_name)}:/app/test.py',
                'python-sandbox',
                'python',
                '/app/test.py',
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        os.remove(file_name)
        return (
            'Pass' if result.returncode == 0 else 'Fail',
            result.stdout,
            result.stderr,
        )
    except Exception as e:
        if os.path.exists(file_name):
            os.remove(file_name)
        return 'Error', '', str(e)


def evaluate_case(item):
    '''评估单个案例.'''
    idx = get_case_id(item)
    code = extract_code(get_case_response(item))
    if not code:
        return {'id': idx, 'status': 'NoCode'}
    valid, err = check_syntax(code)
    if not valid:
        return {'id': idx, 'status': 'SyntaxError', 'error': err}
    status, stdout, stderr = run_in_docker(code, idx)
    return {
        'id': idx,
        'status': status,
        'stdout': stdout[:200],
        'stderr': stderr[:200],
    }


def save_evaluation_report(results, report_path):
    '''保存评估报告.'''
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['id', 'status', 'stdout', 'stderr', 'error']
        )
        writer.writeheader()
        writer.writerows(results)


def calculate_summary(results):
    '''计算评估汇总统计.'''
    summary = {}
    for r in results:
        summary[r['status']] = summary.get(r['status'], 0) + 1
    return summary


def step5_evaluation():
    log_step('[5/5] 运行代码评估...')

    inference_file = OUTPUT_DIR / 'inference_results.json'
    report_path = OUTPUT_DIR / 'evaluation_report.csv'

    if report_path.exists():
        log('  评估报告已存在，跳过评估')
        return True

    with open(inference_file, 'r', encoding='utf-8') as f:
        inference_data = json.load(f)

    log(f'  评估 {len(inference_data)} 条推理结果...')
    log(f"  评估并发: {CONFIG['eval_workers']}")
    with ThreadPoolExecutor(max_workers=CONFIG['eval_workers']) as executor:
        results = list(executor.map(evaluate_case, inference_data))

    save_evaluation_report(results, report_path)

    summary = calculate_summary(results)
    log(f'  评估结果: {summary}')
    return True


# ============ 命令行参数解析 ============
def parse_args():
    parser = argparse.ArgumentParser(description='一键代码SFT训练脚本')
    parser.add_argument(
        '--lazyllm-path',
        type=str,
        default=None,
        help='LazyLLM库路径 (默认: /path/to/your/lazyllm)',
    )
    parser.add_argument(
        '--pipeline-model',
        type=str,
        default=None,
        help='Pipeline模型路径 (默认: /path/to/pipeline/model)',
    )
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
        '--only-step', type=int, default=None, help='只运行指定步骤 (1-5)'
    )
    return parser.parse_args()


# ============ 全局配置变量 (可被命令行参数覆盖) ============
CONFIG = {}


def init_config(args):
    '''根据命令行参数初始化配置'''
    global CONFIG
    global LAZYLLM_PATH, PIPELINE_MODEL, SFT_MODEL
    global DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR
    CONFIG = {
        'lazyllm_path': args.lazyllm_path or LAZYLLM_PATH,
        'pipeline_model': args.pipeline_model or PIPELINE_MODEL,
        'sft_model': args.sft_model or SFT_MODEL,
        'data_dir': Path(args.data_dir) if args.data_dir else DATA_DIR,
        'model_dir': Path(args.model_dir) if args.model_dir else MODEL_DIR,
        'output_dir': Path(args.output_dir) if args.output_dir else OUTPUT_DIR,
        'log_dir': Path(args.log_dir) if args.log_dir else LOG_DIR,
        'vllm_max_model_len': args.vllm_max_model_len or VLLM_MAX_MODEL_LEN,
        'vllm_gpu_memory_utilization': (
            args.vllm_gpu_memory_utilization
            or VLLM_GPU_MEMORY_UTILIZATION
        ),
        'vllm_max_num_seqs': args.vllm_max_num_seqs or VLLM_MAX_NUM_SEQS,
        'vllm_max_num_batched_tokens': (
            args.vllm_max_num_batched_tokens
            or VLLM_MAX_NUM_BATCHED_TOKENS
        ),
        'vllm_response_max_tokens': (
            args.vllm_response_max_tokens or VLLM_RESPONSE_MAX_TOKENS
        ),
        'inference_workers': args.inference_workers or INFERENCE_WORKERS,
        'eval_workers': args.eval_workers or EVAL_WORKERS,
    }
    LAZYLLM_PATH = CONFIG['lazyllm_path']
    PIPELINE_MODEL = CONFIG['pipeline_model']
    SFT_MODEL = CONFIG['sft_model']
    DATA_DIR = CONFIG['data_dir']
    MODEL_DIR = CONFIG['model_dir']
    OUTPUT_DIR = CONFIG['output_dir']
    LOG_DIR = CONFIG['log_dir']
    # 确保目录存在
    for d in [
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        LOG_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    return CONFIG


# ============ 主函数 ============
def main():
    args = parse_args()
    config = init_config(args)

    # 重新初始化日志文件路径
    global LOG_FILE
    LOG_FILE = config['log_dir'] / (
        'run_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    )

    log('==========================================')
    log('一键代码SFT训练脚本')
    log('==========================================')
    log('')
    log_info('配置信息:')
    log(f'  - 基础目录: {BASE_DIR}')
    log(f"  - LazyLLM路径: {config['lazyllm_path']}")
    log(f"  - Pipeline模型: {config['pipeline_model']}")
    log(f"  - SFT模型: {config['sft_model']}")
    log(f"  - 数据目录: {config['data_dir']}")
    log(f"  - 模型目录: {config['model_dir']}")
    log(f"  - 输出目录: {config['output_dir']}")
    log(f"  - 日志目录: {config['log_dir']}")
    log(f"  - VLLM_MAX_MODEL_LEN: {config['vllm_max_model_len']}")
    log(
        '  - VLLM_GPU_MEMORY_UTILIZATION: '
        f"{config['vllm_gpu_memory_utilization']}"
    )
    log(f"  - VLLM_MAX_NUM_SEQS: {config['vllm_max_num_seqs']}")
    log(
        '  - VLLM_MAX_NUM_BATCHED_TOKENS: '
        f"{config['vllm_max_num_batched_tokens']}"
    )
    log(
        '  - VLLM_RESPONSE_MAX_TOKENS: '
        f"{config['vllm_response_max_tokens']}"
    )
    log(f"  - INFERENCE_WORKERS: {config['inference_workers']}")
    log(f"  - EVAL_WORKERS: {config['eval_workers']}")
    log(f'  - 日志文件: {LOG_FILE}')
    log('')

    # 解析跳过的步骤
    skip_steps = set()
    if args.skip_steps:
        skip_steps = set(
            int(x.strip())
            for x in args.skip_steps.split(',')
            if x.strip().isdigit()
        )

    steps = [
        ('数据下载', step1_download_data),
        ('数据处理', step2_data_pipeline),
        ('SFT训练', step3_sft_training),
        ('评测推理', step4_inference),
        ('代码评估', step5_evaluation),
    ]

    for i, (name, step_func) in enumerate(steps, 1):
        # 如果只运行指定步骤
        if args.only_step is not None and i != args.only_step:
            continue
        # 如果步骤在跳过列表中
        if i in skip_steps:
            log_info(f'跳过步骤{i}: {name}')
            continue

        try:
            if not step_func():
                log_error(f'步骤{i}失败！详细错误请查看日志: {LOG_FILE}')
                safe_exit(1)
        except Exception as e:
            log_error(f'步骤{i}异常: {type(e).__name__}: {e}')
            safe_exit(1)

    log('')
    log('==========================================')
    log('全部完成!')
    log('==========================================')
    log('')
    log_info('结果汇总:')
    log(f"  数据目录: {config['data_dir']}")
    log(f"  模型目录: {config['model_dir']}/checkpoint")
    log(f"  推理结果: {config['output_dir']}/inference_results.json")
    log(f"  评估报告: {config['output_dir']}/evaluation_report.csv")
    log(f'  日志文件: {LOG_FILE}')
    log('')

    # 显示评估结果摘要
    report_file = config['output_dir'] / 'evaluation_report.csv'
    if report_file.exists():
        log_info('评估统计:')
        with open(report_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            statuses = [row['status'] for row in reader]
            summary = Counter(statuses)
            total = len(statuses)
            pass_count = summary.get('Pass', 0)
            log(f'  - 总样本: {total}')
            log(f'  - 通过: {pass_count} ({pass_count / total * 100:.1f}%)')
            log(f"  - 失败: {summary.get('Fail', 0)}")
            log(f"  - 语法错误: {summary.get('SyntaxError', 0)}")
            log(f"  - 无代码: {summary.get('NoCode', 0)}")
            log(f"  - 错误: {summary.get('Error', 0)}")

    log('')
    log('==========================================')

    safe_exit(0)


if __name__ == '__main__':
    main()
