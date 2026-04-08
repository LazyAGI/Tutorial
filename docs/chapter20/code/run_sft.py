#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_ENDPOINT', HF_ENDPOINT)

SFT_MODEL = '/models/qwen2.5-0.5b-instruct'


TINY_CODES_DATASET_ENDPOINT = os.environ.get(
    'TINY_CODES_DATASET_ENDPOINT', HF_ENDPOINT
)
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

for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / (
    'run_sft_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
)

CONFIG = {}


def log(msg: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f'[{timestamp}] {msg}'
    print(formatted)
    with open(LOG_FILE, 'a', encoding='utf-8') as file:
        file.write(formatted + '\n')


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


def ensure_local_site_packages():
    import site
    import sysconfig

    candidate_paths = []

    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        candidate_paths.append(user_site)
    else:
        candidate_paths.extend(user_site)

    try:
        candidate_paths.extend(site.getsitepackages())
    except AttributeError:
        pass

    sysconfig_paths = sysconfig.get_paths()
    for key in ['purelib', 'platlib']:
        path = sysconfig_paths.get(key)
        if path:
            candidate_paths.append(path)

    versioned_local_path = (
        Path.home()
        / '.local'
        / 'lib'
        / f'python{sys.version_info.major}.{sys.version_info.minor}'
        / 'site-packages'
    )
    candidate_paths.append(str(versioned_local_path))

    for path in dict.fromkeys(candidate_paths):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def override_hf_endpoint(endpoint=None):
    previous_endpoint = os.environ.get('HF_ENDPOINT')
    if endpoint:
        os.environ['HF_ENDPOINT'] = endpoint
    elif 'HF_ENDPOINT' in os.environ:
        del os.environ['HF_ENDPOINT']
    return previous_endpoint


def restore_hf_endpoint(previous_endpoint):
    if previous_endpoint is not None:
        os.environ['HF_ENDPOINT'] = previous_endpoint
    elif 'HF_ENDPOINT' in os.environ:
        del os.environ['HF_ENDPOINT']


def step1_prepare_data():
    log_step('[1/4] 下载并准备 tiny-codes 数据集...')
    ensure_local_site_packages()

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

    dataset_endpoint = CONFIG.get(
        'dataset_endpoint', TINY_CODES_DATASET_ENDPOINT
    )
    previous_hf_endpoint = override_hf_endpoint(dataset_endpoint)
    log(f'  当前数据集下载端点: {dataset_endpoint}')

    try:
        log('  正在从 Hugging Face 加载数据集...')
        try:
            dataset = load_dataset(
                'nampdn-ai/tiny-codes', split='train', streaming=True
            )
        except Exception as exc:
            log_error(f'加载数据集失败: {exc}')
            log('')
            log_info('提示: 如果遇到权限问题，请尝试以下方法:')
            log('  1. 登录 Hugging Face: huggingface-cli login')
            log('  2. 或设置环境变量: export HF_TOKEN=your_token')
            log('  3. 或手动下载数据集并放置到 data/ 目录')
            log('')
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
                log(f'    已收集 {len(python_data)} 条...')
            if len(python_data) >= target_count:
                break

        train_data = python_data[:5000]
        eval_data = python_data[5000:6000]

        with open(train_path, 'w', encoding='utf-8') as file:
            json.dump(train_data, file, indent=2, ensure_ascii=False)
        with open(eval_path, 'w', encoding='utf-8') as file:
            json.dump(eval_data, file, indent=2, ensure_ascii=False)

        log(f'  SFT训练集: {train_path} ({len(train_data)} 条)')
        log(f'  评测集: {eval_path} ({len(eval_data)} 条)')
        return True
    finally:
        restore_hf_endpoint(previous_hf_endpoint)


def step2_sft_training():
    log_step('[2/4] 直接使用步骤1数据开始 SFT 训练...')

    train_file = DATA_DIR / 'train_python.json'
    checkpoint_dir = MODEL_DIR / 'checkpoint'

    if not train_file.exists():
        log_error(f'SFT 训练集不存在: {train_file}')
        return False

    if checkpoint_dir.exists():
        log('  模型已存在，跳过训练')
        return True

    ensure_local_site_packages()

    import lazyllm
    from lazyllm import finetune, launchers

    model = (
        lazyllm.TrainableModule(
            CONFIG['sft_model'],
            target_path=str(checkpoint_dir)
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
    log(f'  训练数据: {train_file}')
    log(f'  模型保存: {checkpoint_dir}')
    return True


def find_latest_merge_model(base_dir):
    merge_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        for directory in dirs:
            if directory == 'lazyllm_merge':
                full_path = Path(root) / directory
                try:
                    merge_dirs.append((full_path, full_path.stat().st_mtime))
                except OSError:
                    pass
    if not merge_dirs:
        return None
    merge_dirs.sort(key=lambda item: item[1], reverse=True)
    return merge_dirs[0][0]


def load_eval_data(eval_file):
    log('  加载评测数据...')
    with open(eval_file, 'r', encoding='utf-8') as file:
        eval_data = json.load(file)
    log(f'  评测样本: {len(eval_data)} 条')
    return eval_data


def run_inference(
    model,
    eval_data,
    sys_prompt,
    response_max_tokens,
    inference_workers,
):
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
    with open(inference_output, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    log(f'  推理完成: {inference_output}')


def step3_inference():
    log_step('[3/4] 运行评测集推理...')

    ensure_local_site_packages()

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


def extract_code(text):
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
    return item.get('test_case_id', item.get('id', 0))


def get_case_response(item):
    return item.get('response', item.get('prediction', ''))


def check_syntax(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        return False, str(exc)


def run_in_docker(code, case_id):
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
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write('import sys\n' + final_code)

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
    except Exception as exc:
        if os.path.exists(file_name):
            os.remove(file_name)
        return 'Error', '', str(exc)


def evaluate_case(item):
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
    with open(report_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file, fieldnames=['id', 'status', 'stdout', 'stderr', 'error']
        )
        writer.writeheader()
        writer.writerows(results)


def calculate_summary(results):
    summary = {}
    for result in results:
        summary[result['status']] = summary.get(result['status'], 0) + 1
    return summary


def step4_evaluation():
    log_step('[4/4] 运行代码评估...')

    inference_file = OUTPUT_DIR / 'inference_results.json'
    report_path = OUTPUT_DIR / 'evaluation_report.csv'

    if report_path.exists():
        log('  评估报告已存在，跳过评估')
        return True

    with open(inference_file, 'r', encoding='utf-8') as file:
        inference_data = json.load(file)

    log(f'  评估 {len(inference_data)} 条推理结果...')
    log(f"  评估并发: {CONFIG['eval_workers']}")
    with ThreadPoolExecutor(max_workers=CONFIG['eval_workers']) as executor:
        results = list(executor.map(evaluate_case, inference_data))

    save_evaluation_report(results, report_path)

    summary = calculate_summary(results)
    log(f'  评估结果: {summary}')
    return True


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
        '--dataset-endpoint',
        type=str,
        default=None,
        help='Hugging Face 数据集下载端点',
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
    global SFT_MODEL, DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR
    global TINY_CODES_DATASET_ENDPOINT

    CONFIG = {
        'sft_model': (
            args.sft_model if args.sft_model is not None else SFT_MODEL
        ),
        'data_dir': Path(args.data_dir) if args.data_dir else DATA_DIR,
        'model_dir': Path(args.model_dir) if args.model_dir else MODEL_DIR,
        'output_dir': Path(args.output_dir) if args.output_dir else OUTPUT_DIR,
        'log_dir': Path(args.log_dir) if args.log_dir else LOG_DIR,
        'dataset_endpoint': (
            args.dataset_endpoint
            if args.dataset_endpoint is not None
            else TINY_CODES_DATASET_ENDPOINT
        ),
        'vllm_max_model_len': (
            args.vllm_max_model_len
            if args.vllm_max_model_len is not None
            else VLLM_MAX_MODEL_LEN
        ),
        'vllm_gpu_memory_utilization': (
            args.vllm_gpu_memory_utilization
            if args.vllm_gpu_memory_utilization is not None
            else VLLM_GPU_MEMORY_UTILIZATION
        ),
        'vllm_max_num_seqs': (
            args.vllm_max_num_seqs
            if args.vllm_max_num_seqs is not None
            else VLLM_MAX_NUM_SEQS
        ),
        'vllm_max_num_batched_tokens': (
            args.vllm_max_num_batched_tokens
            if args.vllm_max_num_batched_tokens is not None
            else VLLM_MAX_NUM_BATCHED_TOKENS
        ),
        'vllm_response_max_tokens': (
            args.vllm_response_max_tokens
            if args.vllm_response_max_tokens is not None
            else VLLM_RESPONSE_MAX_TOKENS
        ),
        'inference_workers': (
            args.inference_workers
            if args.inference_workers is not None
            else INFERENCE_WORKERS
        ),
        'eval_workers': (
            args.eval_workers
            if args.eval_workers is not None
            else EVAL_WORKERS
        ),
    }

    SFT_MODEL = CONFIG['sft_model']
    DATA_DIR = CONFIG['data_dir']
    MODEL_DIR = CONFIG['model_dir']
    OUTPUT_DIR = CONFIG['output_dir']
    LOG_DIR = CONFIG['log_dir']
    TINY_CODES_DATASET_ENDPOINT = CONFIG['dataset_endpoint']

    for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    return CONFIG


def main():
    args = parse_args()
    config = init_config(args)

    global LOG_FILE
    LOG_FILE = config['log_dir'] / (
        'run_sft_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    )

    if not Path(config['sft_model']).exists():
        log_error(f"SFT_MODEL 不存在: {config['sft_model']}")
        log('请使用 --sft-model 参数指定正确路径')
        safe_exit(1)

    log('==========================================')
    log('无 pipeline 的一键代码SFT训练脚本')
    log('==========================================')
    log('')
    log_info('配置信息:')
    log(f'  - 基础目录: {BASE_DIR}')
    log(f"  - SFT模型: {config['sft_model']}")
    log(f"  - 数据目录: {config['data_dir']}")
    log(f"  - 模型目录: {config['model_dir']}")
    log(f"  - 输出目录: {config['output_dir']}")
    log(f"  - 日志目录: {config['log_dir']}")
    log(f"  - DATASET_ENDPOINT: {config['dataset_endpoint']}")
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
            log_info(f'跳过步骤{index}: {name}')
            continue

        try:
            if not step_func():
                log_error(f'步骤{index}失败！详细错误请查看日志: {LOG_FILE}')
                safe_exit(1)
        except Exception as exc:
            log_error(f'步骤{index}异常: {type(exc).__name__}: {exc}')
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

    report_file = config['output_dir'] / 'evaluation_report.csv'
    if report_file.exists():
        log_info('评估统计:')
        with open(report_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            statuses = [row['status'] for row in reader]
            summary = Counter(statuses)
            total = len(statuses)
            pass_count = summary.get('Pass', 0)
            pass_ratio = (pass_count / total * 100) if total else 0.0
            log(f'  - 总样本: {total}')
            log(f'  - 通过: {pass_count} ({pass_ratio:.1f}%)')
            log(f"  - 失败: {summary.get('Fail', 0)}")
            log(f"  - 语法错误: {summary.get('SyntaxError', 0)}")
            log(f"  - 无代码: {summary.get('NoCode', 0)}")
            log(f"  - 错误: {summary.get('Error', 0)}")

    log('')
    log('==========================================')
    safe_exit(0)


if __name__ == '__main__':
    main()
