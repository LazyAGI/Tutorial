#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_ENDPOINT', HF_ENDPOINT)

SFT_BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
LAZYLLM_PATH = None
JUDGE_MODEL = 'Qwen/Qwen2.5-14B-Instruct'


TRAIN_DATASET_REPO = 'glaiveai/glaive-function-calling-v2'
TEST_DATASET_REPO = 'rirqing/tool_use'
TRAIN_DATASET_ENDPOINT = os.environ.get(
    'TOOLUSE_TRAIN_DATASET_ENDPOINT', HF_ENDPOINT
)
TEST_DATASET_ENDPOINT = os.environ.get(
    'TOOLUSE_TEST_DATASET_ENDPOINT', HF_ENDPOINT
)

TRAIN_NUM_SAMPLES = int(os.environ.get('TOOLUSE_TRAIN_NUM_SAMPLES', '10000'))
EVAL_NUM_SAMPLES = int(os.environ.get('TOOLUSE_EVAL_NUM_SAMPLES', '1000'))
GLAIVE_SCAN_LIMIT = int(os.environ.get('GLAIVE_SCAN_LIMIT', '10000'))
SFT_EPOCHS = float(os.environ.get('TOOLUSE_SFT_EPOCHS', '3.0'))
SFT_LEARNING_RATE = float(os.environ.get('TOOLUSE_SFT_LEARNING_RATE', '5e-5'))
SFT_BATCH_SIZE = int(os.environ.get('TOOLUSE_SFT_BATCH_SIZE', '8'))
SFT_MAX_SAMPLES = int(os.environ.get('TOOLUSE_SFT_MAX_SAMPLES', '10000'))
INFERENCE_MAX_MODEL_LEN = int(
    os.environ.get('TOOLUSE_INFERENCE_MAX_MODEL_LEN', '4096')
)
INFERENCE_GPU_MEMORY_UTILIZATION = float(
    os.environ.get('TOOLUSE_INFERENCE_GPU_MEMORY_UTILIZATION', '0.8')
)
INFERENCE_MAX_NUM_SEQS = int(
    os.environ.get('TOOLUSE_INFERENCE_MAX_NUM_SEQS', '128')
)
INFERENCE_TEMPERATURE = float(
    os.environ.get('TOOLUSE_INFERENCE_TEMPERATURE', '0.1')
)
INFERENCE_TOP_P = float(os.environ.get('TOOLUSE_INFERENCE_TOP_P', '0.9'))
INFERENCE_MAX_TOKENS = int(
    os.environ.get('TOOLUSE_INFERENCE_MAX_TOKENS', '512')
)
JUDGE_WORKERS = int(os.environ.get('JUDGE_WORKERS', '4'))
JUDGE_MAX_MODEL_LEN = int(os.environ.get('JUDGE_MAX_MODEL_LEN', '4096'))
JUDGE_GPU_MEMORY_UTILIZATION = float(
    os.environ.get('JUDGE_GPU_MEMORY_UTILIZATION', '0.9')
)
JUDGE_MAX_NUM_SEQS = int(os.environ.get('JUDGE_MAX_NUM_SEQS', '8'))
JUDGE_RESPONSE_MAX_TOKENS = int(
    os.environ.get('JUDGE_RESPONSE_MAX_TOKENS', '256')
)

TRAIN_DATA_FILE = 'glaive_alpaca_sft.json'
TEST_DATA_FILE = 'tooluse_eval.jsonl'
INFERENCE_FILE = 'glaive_inference_results.json'
REPORT_FILE = 'glaive_tooluse_evaluation.json'
CHECKPOINT_DIR_NAME = 'glaive_tooluse_sft_checkpoint'

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
LOG_DIR = BASE_DIR / 'logs'

for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / (
    'run_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
)
CONFIG = {}


def get_lazyllm_path():
    '''自动检测 lazyllm 安装路径'''
    import importlib.util

    spec = importlib.util.find_spec('lazyllm')
    if spec and spec.origin:
        return str(Path(spec.origin).parent.parent)
    return None


SUPPORTED_TEST_DATA_SUFFIXES = (
    '.jsonl',
    '.json',
    '.parquet',
    '.csv',
    '.json.gz',
    '.jsonl.gz',
)

JUDGE_PROMPT = '''You are a strict tool-use evaluation expert.
Please evaluate whether the predicted answer correctly solves
the user task compared with the reference answer.

### User Input
{user_input}

### Reference Output
{gold_output}

### Predicted Output
{pred_output}

### Scoring Dimensions
1. Format Correctness (0-5): Is the output structurally valid for tool use?
2. Tool Selection (0-5): Is the chosen tool or no-tool decision correct?
3. Argument Accuracy (0-5): Are the arguments accurate and complete?
4. Logic Reasoning (0-5): Is the response logically consistent
   with the task and reference?

Return JSON only:
```json
{{
  "format_score": 5,
  "tool_score": 5,
  "arg_score": 5,
  "logic_score": 5,
  "total_score": 20,
  "is_perfect": true,
  "reason": "Perfect"
}}
```
'''


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


def ensure_lazyllm_paths():
    lazyllm_path = Path(CONFIG.get('lazyllm_path', LAZYLLM_PATH)).resolve()
    candidate_paths = [lazyllm_path, lazyllm_path.parent]
    for path in candidate_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_json_file(path: Path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_jsonl_file(path: Path):
    items = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_json_file(path: Path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_jsonl_file(path: Path, data):
    with open(path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def truncate_text(text, max_length: int):
    text = text if isinstance(text, str) else str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'


def parse_json_from_text(text: str):
    candidates = [text.strip()]
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            candidates.insert(0, match.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find('{')
            end = candidate.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(candidate[start:end + 1])
                except json.JSONDecodeError:
                    continue

    raise ValueError('无法从文本中解析 JSON')


def override_hf_endpoint(endpoint=None):
    previous = os.environ.get('HF_ENDPOINT')
    if endpoint:
        os.environ['HF_ENDPOINT'] = endpoint
    else:
        os.environ.pop('HF_ENDPOINT', None)
    return previous


def restore_hf_endpoint(previous):
    if previous is None:
        os.environ.pop('HF_ENDPOINT', None)
    else:
        os.environ['HF_ENDPOINT'] = previous


def coerce_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_prompt(instruction: str, input_text: str):
    instruction = coerce_text(instruction).strip()
    input_text = coerce_text(input_text).strip()
    if instruction and input_text:
        return f'{instruction}\n\n{input_text}'
    return instruction or input_text


def extract_answer_field(item):
    for key in [
        'output',
        'answer',
        'assistant',
        'assistant_response',
        'response',
        'completion',
        'target',
        'gold',
        'gold_output',
        'reference',
        'expected_output',
    ]:
        value = item.get(key)
        if value not in [None, '']:
            return coerce_text(value).strip()
    return ''


def convert_glaive_to_alpaca(example):
    system_prompt = example.get('system', '')
    chat_history = example.get('chat', '')

    lines = chat_history.split('\n')
    user_query = ''
    assistant_response = ''

    for line in lines:
        if line.startswith('USER:'):
            user_query = line.replace('USER:', '').strip()
        elif line.startswith('ASSISTANT:'):
            assistant_response = line.replace('ASSISTANT:', '').strip()

    return {
        'instruction': system_prompt,
        'input': user_query,
        'output': assistant_response,
    }


def extract_from_messages(messages):
    instruction_parts = []
    user_query = ''
    assistant_response = ''

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get('role', '')).lower()
        content = coerce_text(message.get('content', '')).strip()
        if not content:
            continue
        if role == 'system':
            instruction_parts.append(content)
        elif role in {'user', 'human'}:
            user_query = content
        elif role in {'assistant', 'model', 'gpt'}:
            assistant_response = content

    return '\n'.join(instruction_parts).strip(), user_query, assistant_response


def extract_from_conversations(conversations):
    instruction_parts = []
    user_query = ''
    assistant_response = ''

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get('role', turn.get('from', ''))).lower()
        content = coerce_text(
            turn.get('content', turn.get('value', turn.get('text', '')))
        ).strip()
        if not content:
            continue
        if role in {'system'}:
            instruction_parts.append(content)
        elif role in {'user', 'human'}:
            user_query = content
        elif role in {'assistant', 'gpt', 'model'}:
            assistant_response = content

    return '\n'.join(instruction_parts).strip(), user_query, assistant_response


def normalize_test_record(item, index: int):
    if not isinstance(item, dict):
        prompt = coerce_text(item).strip()
        if not prompt:
            return None
        return {
            'test_case_id': index,
            'instruction': '',
            'input': prompt,
            'prompt': prompt,
            'gold_output': '',
        }

    instruction = ''
    input_text = ''
    gold_output = ''

    if 'instruction' in item or 'input' in item:
        instruction = coerce_text(item.get('instruction', '')).strip()
        input_text = coerce_text(
            item.get('input', item.get('question', item.get('prompt', '')))
        ).strip()
        gold_output = extract_answer_field(item)
    elif 'prompt' in item:
        input_text = coerce_text(item.get('prompt', '')).strip()
        gold_output = extract_answer_field(item)
    elif 'question' in item:
        instruction = coerce_text(
            item.get('system', item.get('instruction', ''))
        ).strip()
        input_text = coerce_text(item.get('question', '')).strip()
        gold_output = extract_answer_field(item)
    elif 'system' in item and 'chat' in item:
        converted = convert_glaive_to_alpaca(item)
        instruction = converted['instruction']
        input_text = converted['input']
        gold_output = converted['output']
    elif isinstance(item.get('messages'), list):
        instruction, input_text, gold_output = extract_from_messages(
            item['messages']
        )
    elif isinstance(item.get('conversation'), list):
        instruction, input_text, gold_output = extract_from_conversations(
            item['conversation']
        )
    elif isinstance(item.get('conversations'), list):
        instruction, input_text, gold_output = extract_from_conversations(
            item['conversations']
        )
    else:
        instruction = coerce_text(item.get('system', '')).strip()
        input_text = coerce_text(
            item.get('query', item.get('text', item.get('content', '')))
        ).strip()
        gold_output = extract_answer_field(item)

    prompt = build_prompt(instruction, input_text)
    if not prompt:
        return None

    return {
        'test_case_id': item.get('test_case_id', index),
        'instruction': instruction,
        'input': input_text,
        'prompt': prompt,
        'gold_output': gold_output,
    }


def is_train_dataset_ready(path: Path):
    if not path.exists():
        return False
    try:
        data = load_json_file(path)
    except Exception:
        return False
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    required_keys = {'instruction', 'input', 'output'}
    return required_keys.issubset(data[0].keys())


def is_test_dataset_ready(path: Path):
    if not path.exists():
        return False
    try:
        data = load_jsonl_file(path)
    except Exception:
        return False
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    required_keys = {'test_case_id', 'prompt', 'gold_output'}
    return required_keys.issubset(data[0].keys())


def is_inference_ready(path: Path):
    if not path.exists():
        return False
    try:
        data = load_json_file(path)
    except Exception:
        return False
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    required_keys = {'test_case_id', 'prompt', 'response', 'gold_output'}
    return required_keys.issubset(data[0].keys())


def is_report_ready(path: Path):
    if not path.exists():
        return False
    try:
        data = load_json_file(path)
    except Exception:
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get('summary'), dict)
        and isinstance(data.get('details'), list)
    )


def score_test_repo_file(filename: str):
    lower_name = filename.lower()
    if not any(
        lower_name.endswith(suffix) for suffix in SUPPORTED_TEST_DATA_SUFFIXES
    ):
        return None

    score = 1000
    base_name = Path(lower_name).name

    if 'test' in base_name:
        score -= 400
    if 'eval' in base_name or 'evaluation' in base_name:
        score -= 350
    if 'validation' in base_name or 'dev' in base_name:
        score -= 300
    if '/test' in lower_name or 'test/' in lower_name:
        score -= 100
    if '/data/' in lower_name:
        score -= 30
    if 'train' in lower_name:
        score += 500
    if 'readme' in lower_name or lower_name.endswith('.md'):
        score += 800

    return score, len(filename), filename


def resolve_test_dataset_file(repo_id: str, specified_file: str | None = None):
    from huggingface_hub import list_repo_files

    if specified_file:
        return specified_file

    files = list_repo_files(repo_id=repo_id, repo_type='dataset')
    scored = []
    for filename in files:
        result = score_test_repo_file(filename)
        if result is not None:
            scored.append(result)

    if not scored:
        raise RuntimeError(f'在 {repo_id} 中未找到可用的测试数据文件')

    scored.sort()
    top_candidates = [item[2] for item in scored[:5]]
    log(f'  测试集候选文件: {top_candidates}')
    return scored[0][2]


def load_records_from_local_dataset_file(path: Path):
    from datasets import load_dataset

    lower_name = path.name.lower()
    if lower_name.endswith(('.json', '.jsonl', '.json.gz', '.jsonl.gz')):
        loader = 'json'
    elif lower_name.endswith('.parquet'):
        loader = 'parquet'
    elif lower_name.endswith('.csv'):
        loader = 'csv'
    else:
        raise ValueError(f'不支持的测试数据文件格式: {path}')

    dataset = load_dataset(loader, data_files=str(path), split='train')
    return [dict(item) for item in dataset]


def find_latest_merge_model(base_dir: Path):
    merge_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        if 'lazyllm_merge' in dirs:
            path = Path(root) / 'lazyllm_merge'
            try:
                merge_dirs.append((path, path.stat().st_mtime))
            except OSError:
                pass
    return max(merge_dirs, key=lambda item: item[1])[0] if merge_dirs else None


class ToolUseJudge:
    def __init__(
        self,
        model_path,
        max_model_len,
        gpu_memory_utilization,
        max_num_seqs,
        response_max_tokens,
    ):
        import lazyllm
        from lazyllm import deploy

        self.model = (
            lazyllm.TrainableModule(model_path)
            .deploy_method(
                (
                    deploy.vllm,
                    {
                        'max_model_len': max_model_len,
                        'gpu_memory_utilization': gpu_memory_utilization,
                        'max_num_seqs': max_num_seqs,
                    },
                )
            )
            .start()
        )
        self.response_max_tokens = response_max_tokens

    def evaluate(self, user_input, gold_output, pred_output):
        try:
            prompt = JUDGE_PROMPT.format(
                user_input=user_input,
                gold_output=gold_output,
                pred_output=pred_output,
            )
            result = self.model(prompt, max_tokens=self.response_max_tokens)
            parsed = parse_json_from_text(
                result if isinstance(result, str) else str(result)
            )
            return {
                'format_score': parsed.get('format_score', 0),
                'tool_score': parsed.get('tool_score', 0),
                'arg_score': parsed.get('arg_score', 0),
                'logic_score': parsed.get('logic_score', 0),
                'total_score': parsed.get('total_score', 0),
                'is_perfect': parsed.get('is_perfect', False),
                'reason': parsed.get('reason', ''),
            }
        except Exception as exc:
            return {
                'format_score': 0,
                'tool_score': 0,
                'arg_score': 0,
                'logic_score': 0,
                'total_score': 0,
                'is_perfect': False,
                'reason': f'评估失败: {exc}',
            }


def _prepare_train_data(train_path: Path) -> bool:
    '''准备训练数据。'''
    previous_endpoint = override_hf_endpoint(
        CONFIG['train_dataset_endpoint']
    )
    try:
        log(
            f'  正在从 Hugging Face 下载训练集: '
            f'{CONFIG["train_dataset_repo"]}'
        )
        if CONFIG['train_dataset_endpoint']:
            log(f'  训练集下载端点: {CONFIG["train_dataset_endpoint"]}')
        from datasets import load_dataset
        dataset = load_dataset(CONFIG['train_dataset_repo'], split='train')
    finally:
        restore_hf_endpoint(previous_endpoint)

    alpaca_data = []
    log('  开始按给定逻辑转换 Glaive 训练集...')
    for i, example in enumerate(dataset):
        if not example.get('system') or not example.get('chat'):
            continue
        alpaca_item = convert_glaive_to_alpaca(example)
        alpaca_data.append(alpaca_item)
        if i > CONFIG['glaive_scan_limit']:
            break

    train_data = alpaca_data[: CONFIG['train_num_samples']]
    if not train_data:
        log_error('训练集转换失败，未得到有效样本')
        return False

    write_json_file(train_path, train_data)
    log(f'  训练集保存: {train_path} ({len(train_data)} 条)')
    return True


def _prepare_test_data(test_path: Path) -> bool:  # noqa: C901
    '''准备测试数据。'''
    from huggingface_hub import hf_hub_download

    previous_endpoint = override_hf_endpoint(
        CONFIG['test_dataset_endpoint']
    )
    try:
        log(f'  正在解析测试集仓库: {CONFIG["test_dataset_repo"]}')
        if CONFIG['test_dataset_endpoint']:
            log(f'  测试集下载端点: {CONFIG["test_dataset_endpoint"]}')
        try:
            test_data_file = resolve_test_dataset_file(
                CONFIG['test_dataset_repo'], CONFIG['test_data_file']
            )
        except Exception as exc:
            log_error(f'解析测试集文件失败: {exc}')
            return False

        log(f'  选中的测试集文件: {test_data_file}')
        try:
            local_test_file = hf_hub_download(
                repo_id=CONFIG['test_dataset_repo'],
                filename=test_data_file,
                repo_type='dataset',
            )
        except Exception as exc:
            log_error(f'下载测试集文件失败: {exc}')
            return False
    finally:
        restore_hf_endpoint(previous_endpoint)

    try:
        raw_test_data = load_records_from_local_dataset_file(
            Path(local_test_file)
        )
    except Exception as exc:
        log_error(f'加载测试集文件失败: {exc}')
        return False

    normalized_test_data = []
    for index, item in enumerate(raw_test_data):
        normalized = normalize_test_record(item, index)
        if normalized:
            normalized_test_data.append(normalized)
        if len(normalized_test_data) >= CONFIG['eval_num_samples']:
            break

    if not normalized_test_data:
        log_error('测试集归一化失败，未得到有效样本')
        return False

    write_jsonl_file(test_path, normalized_test_data)
    log(f'  测试集保存: {test_path} ({len(normalized_test_data)} 条)')
    return True


def step1_prepare_data(step_label='[1/4]'):  # noqa: C901
    log_step(f'{step_label} 下载并准备训练集和测试集...')
    ensure_local_site_packages()

    train_path = DATA_DIR / TRAIN_DATA_FILE
    test_path = DATA_DIR / TEST_DATA_FILE

    train_ready = is_train_dataset_ready(train_path)
    test_ready = is_test_dataset_ready(test_path)

    if train_ready and test_ready:
        train_data = load_json_file(train_path)
        test_data = load_jsonl_file(test_path)
        log('  数据已存在且格式正确，跳过下载')
        log(f'  训练集: {len(train_data)} 条')
        log(f'  测试集: {len(test_data)} 条')
        return True

    try:
        from datasets import load_dataset  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError as exc:
        log_error(f'缺少依赖，请先安装 datasets 和 huggingface_hub: {exc}')
        return False

    if not train_ready and not _prepare_train_data(train_path):
        return False

    if not test_ready and not _prepare_test_data(test_path):
        return False

    return True


def step2_sft_training():
    log_step('[2/4] 开始 SFT 训练...')
    ensure_local_site_packages()
    ensure_lazyllm_paths()

    import lazyllm
    from lazyllm import finetune, launchers

    train_file = DATA_DIR / TRAIN_DATA_FILE
    checkpoint_dir = MODEL_DIR / CHECKPOINT_DIR_NAME

    if checkpoint_dir.exists():
        log('  模型已存在，跳过训练')
        return True

    if not is_train_dataset_ready(train_file):
        log_error(f'SFT 训练数据不存在或格式错误: {train_file}')
        return False

    train_data = load_json_file(train_file)
    log(f'  训练样本: {len(train_data)} 条')

    model = (
        lazyllm.TrainableModule(
            CONFIG['sft_base_model'],
            target_path=str(checkpoint_dir)
        )
        .mode('finetune')
        .trainset(str(train_file))
        .finetune_method(
            (
                finetune.llamafactory,
                {
                    'learning_rate': CONFIG['sft_learning_rate'],
                    'cutoff_len': 4096,
                    'max_samples': CONFIG['sft_max_samples'],
                    'val_size': 0.1,
                    'optim': 'adamw_torch_fused',
                    'bf16': True,
                    'fp16': False,
                    'per_device_train_batch_size': CONFIG['sft_batch_size'],
                    'gradient_accumulation_steps': 4,
                    'num_train_epochs': CONFIG['sft_epochs'],
                    'warmup_ratio': 0.1,
                    'template': 'qwen',
                    'stage': 'sft',
                    'save_steps': 10,
                    'resume_from_checkpoint': None,
                    'save_strategy': 'steps',
                    'save_total_limit': 3,
                    'launcher': launchers.empty(ngpus=1),
                },
            )
        )
    )

    # `update()` will run train + server + eval in LazyLLM.
    # Only run the training stage here so inference/evaluation happen later.
    model._update(mode=['train'])
    log(f'  模型保存: {checkpoint_dir}')
    return True


def step3_inference(step_label='[3/4]'):
    log_step(f'{step_label} 运行测试集推理...')
    ensure_local_site_packages()
    ensure_lazyllm_paths()

    import lazyllm
    from lazyllm import deploy

    test_file = DATA_DIR / TEST_DATA_FILE
    inference_output = OUTPUT_DIR / INFERENCE_FILE

    if is_inference_ready(inference_output):
        data = load_json_file(inference_output)
        log('  推理结果已存在，跳过推理')
        log(f'  已有结果: {len(data)} 条')
        return True

    if not is_test_dataset_ready(test_file):
        log_error(f'测试集不存在或格式错误: {test_file}')
        return False

    test_data = load_jsonl_file(test_file)
    model_path = find_latest_merge_model(MODEL_DIR)
    if not model_path:
        fallback_model_path = MODEL_DIR / CHECKPOINT_DIR_NAME
        if fallback_model_path.exists():
            model_path = fallback_model_path

    if not model_path:
        log_error(f'在 {MODEL_DIR} 下未找到可用模型目录')
        return False

    log(f'  找到模型: {model_path}')
    log(f'  测试样本: {len(test_data)} 条')

    model = lazyllm.TrainableModule(str(model_path)).deploy_method(
        (
            deploy.vllm,
            {
                'max_model_len': INFERENCE_MAX_MODEL_LEN,
                'gpu_memory_utilization': (
                    INFERENCE_GPU_MEMORY_UTILIZATION
                ),
                'max_num_seqs': CONFIG['inference_max_num_seqs'],
            },
        )
    )
    model.start()

    try:
        results = []
        for index, item in enumerate(test_data):
            prompt = item.get('prompt', '')
            response = model(
                prompt,
                temperature=INFERENCE_TEMPERATURE,
                top_p=INFERENCE_TOP_P,
                max_tokens=INFERENCE_MAX_TOKENS,
            )
            response_text = (
                response if isinstance(response, str) else str(response)
            )

            results.append(
                {
                    'test_case_id': item.get('test_case_id', index),
                    'instruction': item.get('instruction', ''),
                    'input': item.get('input', ''),
                    'prompt': prompt,
                    'gold_output': item.get('gold_output', ''),
                    'response': response_text,
                }
            )

            if (index + 1) % 10 == 0 or index + 1 == len(test_data):
                log(f'    已处理: {index + 1}/{len(test_data)}')

        write_json_file(inference_output, results)
        log(f'  推理结果保存: {inference_output}')
        return True
    finally:
        model.stop()


def step4_evaluation(step_label='[4/4]'):
    log_step(f'{step_label} 运行 Tool Use 评测...')
    ensure_local_site_packages()
    ensure_lazyllm_paths()

    inference_output = OUTPUT_DIR / INFERENCE_FILE
    report_path = OUTPUT_DIR / REPORT_FILE

    if is_report_ready(report_path):
        log('  评测报告已存在，跳过评测')
        report = load_json_file(report_path)
        summary = report['summary']
        log(f'  总分平均分: {summary["avg_total_score"]:.2f} / 20')
        log(f'  满分完美比例: {summary["perfect_rate"]:.2f}%')
        return True

    if not is_inference_ready(inference_output):
        log_error(f'推理结果不存在或格式错误: {inference_output}')
        return False

    inference_results = load_json_file(inference_output)
    log(f'  加载推理结果: {len(inference_results)} 条')

    judge = None
    results = [None] * len(inference_results)

    def evaluate_single(index, item):
        user_input = item.get('prompt', '')
        gold_output = item.get('gold_output', '')
        pred_output = item.get('response', '')
        evaluation = judge.evaluate(user_input, gold_output, pred_output)
        return index, {
            'test_case_id': item.get('test_case_id', index),
            'input': truncate_text(user_input, 100),
            'gold_output': truncate_text(gold_output, 200),
            'response': truncate_text(pred_output, 200),
            'evaluation': evaluation,
        }

    try:
        judge = ToolUseJudge(
            CONFIG['judge_model'],
            CONFIG['judge_max_model_len'],
            CONFIG['judge_gpu_memory_utilization'],
            CONFIG['judge_max_num_seqs'],
            CONFIG['judge_response_max_tokens'],
        )

        num_workers = max(
            1, min(CONFIG['judge_workers'], len(inference_results))
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(evaluate_single, index, item)
                for index, item in enumerate(inference_results)
            ]
            for done_count, future in enumerate(as_completed(futures), 1):
                result_index, result = future.result()
                results[result_index] = result
                if done_count % 10 == 0 or done_count == len(
                    inference_results
                ):
                    log(f'    已评测: {done_count}/{len(inference_results)}')
    finally:
        if judge is not None:
            judge.model.stop()

    metrics = {
        'total': len(results),
        'format': sum(item['evaluation']['format_score'] for item in results),
        'tool': sum(item['evaluation']['tool_score'] for item in results),
        'arg': sum(item['evaluation']['arg_score'] for item in results),
        'logic': sum(item['evaluation']['logic_score'] for item in results),
        'perfect': sum(
            1
            for item in results
            if item['evaluation']['is_perfect']
            or item['evaluation']['total_score'] == 20
        ),
    }

    count = metrics['total']
    avg_format = metrics['format'] / count if count else 0
    avg_tool = metrics['tool'] / count if count else 0
    avg_arg = metrics['arg'] / count if count else 0
    avg_logic = metrics['logic'] / count if count else 0
    avg_total = avg_format + avg_tool + avg_arg + avg_logic
    perfect_rate = metrics['perfect'] / count * 100 if count else 0

    summary = {
        'total': count,
        'avg_format_score': avg_format,
        'avg_tool_score': avg_tool,
        'avg_arg_score': avg_arg,
        'avg_logic_score': avg_logic,
        'avg_total_score': avg_total,
        'perfect_rate': perfect_rate,
    }

    log(f'  评测完成: {count} 条样本')
    log(f'  平均格式得分: {avg_format:.2f} / 5')
    log(f'  平均工具选择得分: {avg_tool:.2f} / 5')
    log(f'  平均参数准确得分: {avg_arg:.2f} / 5')
    log(f'  平均逻辑合理性得分: {avg_logic:.2f} / 5')
    log(f'  总分平均分: {avg_total:.2f} / 20')
    log(f'  满分完美比例: {perfect_rate:.2f}%')

    write_json_file(report_path, {'summary': summary, 'details': results})
    log(f'  评测报告保存: {report_path}')
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='一键 Tool Use SFT 训练脚本')
    parser.add_argument(
        '--lazyllm-path', type=str, default=None, help='LazyLLM 库路径'
    )
    parser.add_argument(
        '--sft-base-model', type=str, default=None, help='SFT 基础模型路径'
    )
    parser.add_argument(
        '--judge-model', type=str, default=None, help='评测裁判模型路径'
    )
    parser.add_argument(
        '--train-dataset-repo', type=str, default=None, help='训练集仓库'
    )
    parser.add_argument(
        '--train-dataset-endpoint',
        type=str,
        default=None,
        help='训练集下载端点，默认使用 hf-mirror',
    )
    parser.add_argument(
        '--test-dataset-repo', type=str, default=None, help='测试集仓库'
    )
    parser.add_argument(
        '--test-dataset-endpoint',
        type=str,
        default=None,
        help='测试集下载端点，默认使用 hf-mirror',
    )
    parser.add_argument(
        '--test-data-file', type=str, default=None, help='测试集文件路径'
    )
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录')
    parser.add_argument('--model-dir', type=str, default=None, help='模型目录')
    parser.add_argument(
        '--output-dir', type=str, default=None, help='输出目录'
    )
    parser.add_argument('--log-dir', type=str, default=None, help='日志目录')
    parser.add_argument(
        '--train-num-samples',
        type=int,
        default=None,
        help='训练样本数量',
    )
    parser.add_argument(
        '--eval-num-samples',
        type=int,
        default=None,
        help='测试样本数量',
    )
    parser.add_argument(
        '--glaive-scan-limit',
        type=int,
        default=None,
        help='按给定逻辑扫描 Glaive 的最大索引',
    )
    parser.add_argument(
        '--sft-epochs', type=float, default=None, help='SFT 训练轮数'
    )
    parser.add_argument(
        '--sft-learning-rate',
        type=float,
        default=None,
        help='SFT 学习率',
    )
    parser.add_argument(
        '--sft-batch-size',
        type=int,
        default=None,
        help='SFT 单卡 batch size',
    )
    parser.add_argument(
        '--sft-max-samples',
        type=int,
        default=None,
        help='SFT 最大训练样本数',
    )
    parser.add_argument(
        '--inference-max-num-seqs',
        type=int,
        default=None,
        help='推理阶段 vLLM 的 max_num_seqs',
    )
    parser.add_argument(
        '--judge-workers',
        type=int,
        default=None,
        help='评测阶段并发数',
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
    global LAZYLLM_PATH, SFT_BASE_MODEL, JUDGE_MODEL
    global DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR

    # 自动检测 lazyllm 路径
    lazyllm_path = args.lazyllm_path if args.lazyllm_path is not None else LAZYLLM_PATH
    if lazyllm_path is None:
        lazyllm_path = get_lazyllm_path()
        if lazyllm_path is None:
            raise RuntimeError(
                '未找到 lazyllm 安装路径。请通过 pip install lazyllm 安装，'
                '或使用 --lazyllm-path 参数指定路径。'
            )

    train_num_samples = (
        args.train_num_samples
        if args.train_num_samples is not None
        else TRAIN_NUM_SAMPLES
    )
    glaive_scan_limit = (
        args.glaive_scan_limit
        if args.glaive_scan_limit is not None
        else GLAIVE_SCAN_LIMIT
    )

    CONFIG = {
        'lazyllm_path': lazyllm_path,
        'sft_base_model': (
            args.sft_base_model
            if args.sft_base_model is not None
            else SFT_BASE_MODEL
        ),
        'judge_model': args.judge_model
        if args.judge_model is not None
        else JUDGE_MODEL,
        'train_dataset_repo': (
            args.train_dataset_repo
            if args.train_dataset_repo is not None
            else TRAIN_DATASET_REPO
        ),
        'train_dataset_endpoint': (
            args.train_dataset_endpoint
            if args.train_dataset_endpoint is not None
            else TRAIN_DATASET_ENDPOINT
        ),
        'test_dataset_repo': (
            args.test_dataset_repo
            if args.test_dataset_repo is not None
            else TEST_DATASET_REPO
        ),
        'test_dataset_endpoint': (
            args.test_dataset_endpoint
            if args.test_dataset_endpoint is not None
            else TEST_DATASET_ENDPOINT
        ),
        'test_data_file': args.test_data_file,
        'data_dir': Path(args.data_dir) if args.data_dir else DATA_DIR,
        'model_dir': Path(args.model_dir) if args.model_dir else MODEL_DIR,
        'output_dir': Path(args.output_dir) if args.output_dir else OUTPUT_DIR,
        'log_dir': Path(args.log_dir) if args.log_dir else LOG_DIR,
        'train_num_samples': train_num_samples,
        'eval_num_samples': (
            args.eval_num_samples
            if args.eval_num_samples is not None
            else EVAL_NUM_SAMPLES
        ),
        'glaive_scan_limit': max(glaive_scan_limit, train_num_samples),
        'sft_epochs': args.sft_epochs
        if args.sft_epochs is not None
        else SFT_EPOCHS,
        'sft_learning_rate': (
            args.sft_learning_rate
            if args.sft_learning_rate is not None
            else SFT_LEARNING_RATE
        ),
        'sft_batch_size': (
            args.sft_batch_size
            if args.sft_batch_size is not None
            else SFT_BATCH_SIZE
        ),
        'sft_max_samples': (
            args.sft_max_samples
            if args.sft_max_samples is not None
            else SFT_MAX_SAMPLES
        ),
        'inference_max_num_seqs': (
            args.inference_max_num_seqs
            if args.inference_max_num_seqs is not None
            else INFERENCE_MAX_NUM_SEQS
        ),
        'judge_workers': (
            args.judge_workers
            if args.judge_workers is not None
            else JUDGE_WORKERS
        ),
        'judge_max_model_len': (
            args.judge_max_model_len
            if args.judge_max_model_len is not None
            else JUDGE_MAX_MODEL_LEN
        ),
        'judge_gpu_memory_utilization': (
            args.judge_gpu_memory_utilization
            if args.judge_gpu_memory_utilization is not None
            else JUDGE_GPU_MEMORY_UTILIZATION
        ),
        'judge_max_num_seqs': (
            args.judge_max_num_seqs
            if args.judge_max_num_seqs is not None
            else JUDGE_MAX_NUM_SEQS
        ),
        'judge_response_max_tokens': (
            args.judge_response_max_tokens
            if args.judge_response_max_tokens is not None
            else JUDGE_RESPONSE_MAX_TOKENS
        ),
    }

    LAZYLLM_PATH = CONFIG['lazyllm_path']
    SFT_BASE_MODEL = CONFIG['sft_base_model']
    JUDGE_MODEL = CONFIG['judge_model']
    DATA_DIR = CONFIG['data_dir']
    MODEL_DIR = CONFIG['model_dir']
    OUTPUT_DIR = CONFIG['output_dir']
    LOG_DIR = CONFIG['log_dir']

    for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    return CONFIG


def main():
    args = parse_args()
    config = init_config(args)

    global LOG_FILE
    LOG_FILE = config['log_dir'] / (
        'run_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    )

    os.chdir(BASE_DIR)

    # 检查模型路径是否存在（LAZYLLM_PATH 可以是 Python 包路径，不需要检查）
    model_paths = [
        ('SFT_BASE_MODEL', config['sft_base_model']),
        ('JUDGE_MODEL', config['judge_model']),
    ]
    for name, path in model_paths:
        if not Path(path).exists():
            log_error(f'{name} 不存在: {path}')
            log(f'请使用 --{name.lower().replace("_", "-")} 参数指定正确路径')
            safe_exit(1)

    log('==========================================')
    log('一键 Tool Use SFT 训练脚本')
    log('==========================================')
    log('')
    log_info('配置信息:')
    log(f'  - 基础目录: {BASE_DIR}')
    log(f'  - LazyLLM路径: {config["lazyllm_path"]}')
    log(f'  - SFT模型: {config["sft_base_model"]}')
    log(f'  - Judge模型: {config["judge_model"]}')
    log(f'  - 训练集仓库: {config["train_dataset_repo"]}')
    log(f'  - 训练集端点: {config["train_dataset_endpoint"]}')
    log(f'  - 测试集仓库: {config["test_dataset_repo"]}')
    log(f'  - 测试集端点: {config["test_dataset_endpoint"]}')
    log(f'  - 测试集文件: {config["test_data_file"] or "自动发现"}')
    log(f'  - 数据目录: {config["data_dir"]}')
    log(f'  - 模型目录: {config["model_dir"]}')
    log(f'  - 输出目录: {config["output_dir"]}')
    log(f'  - 日志目录: {config["log_dir"]}')
    log(f'  - TRAIN_NUM_SAMPLES: {config["train_num_samples"]}')
    log(f'  - EVAL_NUM_SAMPLES: {config["eval_num_samples"]}')
    log(f'  - GLAIVE_SCAN_LIMIT: {config["glaive_scan_limit"]}')
    log(f'  - SFT_EPOCHS: {config["sft_epochs"]}')
    log(f'  - SFT_LEARNING_RATE: {config["sft_learning_rate"]}')
    log(f'  - SFT_BATCH_SIZE: {config["sft_batch_size"]}')
    log(f'  - SFT_MAX_SAMPLES: {config["sft_max_samples"]}')
    log(f'  - INFERENCE_MAX_NUM_SEQS: {config["inference_max_num_seqs"]}')
    log(f'  - JUDGE_WORKERS: {config["judge_workers"]}')
    log(f'  - JUDGE_MAX_MODEL_LEN: {config["judge_max_model_len"]}')
    log(
        '  - JUDGE_GPU_MEMORY_UTILIZATION: '
        f'{config["judge_gpu_memory_utilization"]}'
    )
    log(f'  - JUDGE_MAX_NUM_SEQS: {config["judge_max_num_seqs"]}')
    log(
        f'  - JUDGE_RESPONSE_MAX_TOKENS: {config["judge_response_max_tokens"]}'
    )
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
        ('准备训练集和测试集', step1_prepare_data),
        ('SFT训练', step2_sft_training),
        ('测试集推理', step3_inference),
        ('Tool Use评估', step4_evaluation),
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
    log(f'  训练集: {config["data_dir"] / TRAIN_DATA_FILE}')
    log(f'  测试集: {config["data_dir"] / TEST_DATA_FILE}')
    log(f'  模型目录: {config["model_dir"] / CHECKPOINT_DIR_NAME}')
    log(f'  推理结果: {config["output_dir"] / INFERENCE_FILE}')
    log(f'  评测报告: {config["output_dir"] / REPORT_FILE}')
    log(f'  日志文件: {LOG_FILE}')
    log('')

    report_file = config['output_dir'] / REPORT_FILE
    if report_file.exists():
        log_info('评测指标:')
        report = load_json_file(report_file)
        summary = report['summary']
        log(f'  - 总样本: {summary["total"]}')
        log(f'  - 格式得分: {summary["avg_format_score"]:.2f}/5')
        log(f'  - 工具选择: {summary["avg_tool_score"]:.2f}/5')
        log(f'  - 参数准确: {summary["avg_arg_score"]:.2f}/5')
        log(f'  - 逻辑合理: {summary["avg_logic_score"]:.2f}/5')
        log(
            f'  - 总分: {summary["avg_total_score"]:.2f}/20 '
            f'({summary["avg_total_score"] / 20 * 100:.1f}%)'
        )
        log(f'  - 满分率: {summary["perfect_rate"]:.1f}%')

    log('')
    log('==========================================')
    safe_exit(0)


if __name__ == '__main__':
    main()
