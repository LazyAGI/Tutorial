#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

LAZYLLM_PATH = '/LAZYLLM'
PIPELINE_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
SFT_BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
JUDGE_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
PIPELINE_LIMIT = 1000
JUDGE_WORKERS = int(os.environ.get('JUDGE_WORKERS', '4'))
JUDGE_MAX_MODEL_LEN = int(os.environ.get('JUDGE_MAX_MODEL_LEN', '4096'))
JUDGE_GPU_MEMORY_UTILIZATION = float(
    os.environ.get('JUDGE_GPU_MEMORY_UTILIZATION', '0.9')
)
JUDGE_MAX_NUM_SEQS = int(os.environ.get('JUDGE_MAX_NUM_SEQS', '8'))
JUDGE_RESPONSE_MAX_TOKENS = int(
    os.environ.get('JUDGE_RESPONSE_MAX_TOKENS', '256')
)
HF_DATASET_REPO = 'rirqing/text2sql'
HF_DATA_FILES = {
    'train': 'data/hardest_5000_sql_sft.json',
    'test': 'data/hardest_1000_sql_test_format.jsonl',
}
TRAIN_DATA_FILE = 'hardest_5000_sql_sft.json'
TEST_DATA_FILE = 'hardest_1000_sql_test_format.jsonl'

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


def split_schema_and_question(input_text: str):
    parts = input_text.split('Question:', 1)
    if len(parts) == 2:
        schema = parts[0].replace('Database Schema:', '').strip()
        question = parts[1].strip()
    else:
        schema = ''
        question = input_text.strip()
    return schema, question


def normalize_train_record(item, index: int):
    if all(key in item for key in ['db_id', 'question', 'schema', 'gold_sql']):
        normalized = dict(item)
        normalized.setdefault(
            'prompt',
            (
                f"{normalized.get('instruction', '')}\n\n"
                f"{normalized.get('input', '')}"
            ).strip()
            or normalized.get('input', ''),
        )
        normalized.setdefault('instruction', '')
        normalized.setdefault('input', '')
        normalized.setdefault('output', normalized.get('gold_sql', ''))
        return normalized

    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    output_sql = item.get('output', '')
    schema, question = split_schema_and_question(input_text)
    return {
        'db_id': item.get('db_id', f'text2sql_train_{index}'),
        'question': question,
        'schema': schema,
        'gold_sql': output_sql,
        'prompt': f'{instruction}\n\n{input_text}'.strip() if instruction else input_text,
        'instruction': instruction,
        'input': input_text,
        'output': output_sql,
    }


def normalize_test_record(item):
    normalized = dict(item)
    if 'gold_sql' not in normalized and 'SQL' in normalized:
        normalized['gold_sql'] = normalized['SQL']
    return normalized


def is_train_dataset_ready(path: Path):
    if not path.exists():
        return False
    try:
        data = load_json_file(path)
    except Exception:
        return False
    if not isinstance(data, list) or not data:
        return False
    required_keys = {
        'db_id', 'question', 'schema', 'gold_sql', 'instruction', 'input', 'output'
    }
    return required_keys.issubset(data[0].keys())


def is_test_dataset_ready(path: Path):
    if not path.exists():
        return False
    try:
        data = load_jsonl_file(path)
    except Exception:
        return False
    if not data:
        return False
    required_keys = {'db_id', 'question', 'schema', 'gold_sql'}
    return required_keys.issubset(data[0].keys())


def find_latest_merge_model(base_dir):
    merge_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        if 'lazyllm_merge' in dirs:
            path = Path(root) / 'lazyllm_merge'
            try:
                merge_dirs.append((path, path.stat().st_mtime))
            except OSError:
                pass
    return max(merge_dirs, key=lambda item: item[1])[0] if merge_dirs else None


def extract_sql(text):
    patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(SELECT\s+.*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip()


class MockDatabaseManager:
    def __init__(self):
        self._db_schemas = {}

    def register_schema(self, db_id, schema_str):
        self._db_schemas[db_id] = schema_str

    def list_databases(self):
        return []

    def database_exists(self, _db_id):
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
        import random

        class CompareResult:
            def __init__(self, res):
                self.res = res

        return [CompareResult(random.choice([0, 1])) for _ in comparisons]


class SQLJudge:
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

        self.model = lazyllm.TrainableModule(model_path).deploy_method(
            (
                deploy.vllm,
                {
                    'max_model_len': max_model_len,
                    'gpu_memory_utilization': gpu_memory_utilization,
                    'max_num_seqs': max_num_seqs,
                },
            )
        ).start()
        self.response_max_tokens = response_max_tokens

    def evaluate(self, question, gold_sql, pred_sql):
        try:
            prompt = JUDGE_PROMPT.format(
                question=question,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
            )
            result = self.model(prompt, max_tokens=self.response_max_tokens)
            json_match = re.search(
                r'```json\s*({.*?)\s*```', result, re.DOTALL
            )
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
                'reason': parsed.get('reason', ''),
            }
        except Exception as exc:
            return {
                'semantic_score': 0,
                'syntax_score': 0,
                'equivalence_score': 0,
                'overall_score': 0.0,
                'is_correct': False,
                'reason': f'评估失败: {str(exc)}',
            }


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
{{
    "semantic_score": 5,
    "syntax_score": 5,
    "equivalence_score": 3,
    "overall_score": 5.0,
    "is_correct": true,
    "reason": "SQL 完全正确，正确理解了用户意图"
}}
```'''


def step1_prepare_data(step_label='[1/5]'):
    log_step(f'{step_label} 下载 Text2SQL 数据集...')
    ensure_local_site_packages()

    train_path = DATA_DIR / TRAIN_DATA_FILE
    test_path = DATA_DIR / TEST_DATA_FILE

    if is_train_dataset_ready(train_path) and is_test_dataset_ready(test_path):
        train_data = load_json_file(train_path)
        test_data = load_jsonl_file(test_path)
        log('  数据已存在且格式正确，跳过下载')
        log(f'  训练集: {len(train_data)} 条')
        log(f'  测试集: {len(test_data)} 条')
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log_error(
            '请先安装 huggingface_hub: pip install huggingface_hub'
        )
        return False

    log(
        '  正在从 Hugging Face 数据集仓库 '
        f'{HF_DATASET_REPO} 的 data/ 目录下载文件...'
    )

    download_targets = [
        (HF_DATA_FILES['train'], train_path),
        (HF_DATA_FILES['test'], test_path),
    ]

    for hf_filename, local_path in download_targets:
        if local_path.exists():
            log(f'  本地已存在，直接复用: {local_path.name}')
            continue

        try:
            cached_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=hf_filename,
                repo_type='dataset',
            )
        except Exception as exc:
            log_error(f'下载 {hf_filename} 失败: {exc}')
            return False

        source_path = Path(cached_path)
        local_path.write_bytes(source_path.read_bytes())
        log(f'  已下载: {hf_filename} -> {local_path.name}')

    raw_train_data = load_json_file(train_path)
    if not isinstance(raw_train_data, list) or not raw_train_data:
        log_error(f'训练集为空或格式错误: {train_path}')
        return False
    train_data = [
        normalize_train_record(item, index)
        for index, item in enumerate(raw_train_data)
    ]
    write_json_file(train_path, train_data)

    raw_test_data = load_jsonl_file(test_path)
    if not raw_test_data:
        log_error(f'测试集为空或格式错误: {test_path}')
        return False
    test_data = [
        normalize_test_record(item) for item in raw_test_data
    ]
    write_jsonl_file(test_path, test_data)

    log(f'  训练集: {len(train_data)} 条')
    log(f'  测试集: {len(test_data)} 条')
    return True


def step2_text2sql_pipeline():
    log_step('[2/5] 运行 Text2SQL Pipeline...')

    if not Path(LAZYLLM_PATH).exists():
        log_error(f'LAZYLLM_PATH 不存在: {LAZYLLM_PATH}')
        log('请修改脚本中的 LAZYLLM_PATH 配置')
        return False

    sys.path.insert(0, LAZYLLM_PATH)
    sys.path.insert(0, str(Path(LAZYLLM_PATH).parent))

    import lazyllm
    from lazyllm.tools.data.pipelines import text2sql_synthetic_ppl

    input_file = DATA_DIR / TRAIN_DATA_FILE
    output_file = DATA_DIR / 'ppl_text2sql.json'

    if output_file.exists():
        log('  Pipeline 输出已存在，跳过处理')
        return True

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    log(f'  加载数据: {len(data)} 条')

    db_manager = MockDatabaseManager()
    sample_data = data[: CONFIG['pipeline_limit']]
    for item in sample_data:
        db_id = item.get('db_id')
        schema = item.get('schema', '')
        if db_id and schema:
            db_manager.register_schema(db_id, schema)

    log(f'  已注册 {len(db_manager._db_schemas)} 个数据库 schema')

    model = lazyllm.TrainableModule(PIPELINE_MODEL)
    model.start()

    try:
        ppl = text2sql_synthetic_ppl(
            model=model,
            embedding_model=None,
            database_manager=db_manager,
            output_num=2,
            input_query_num=3,
            num_generations=5,
            output_format='alpaca',
            target_complexity='hard',
        )

        log(f'  批量处理 {len(sample_data)} 条数据...')
        results = ppl(sample_data)

        output_data = []
        for result in results:
            if result and isinstance(result, dict):
                output_data.append(
                    {
                        'instruction': result.get('instruction', ''),
                        'input': result.get('input', ''),
                        'output': result.get('output', ''),
                    }
                )

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=2)
    finally:
        model.stop()

    log(f'  生成数据: {len(output_data)} 条')
    return True


def step3_sft_training():
    log_step('[3/5] 开始 SFT 训练...')

    ensure_local_site_packages()

    import lazyllm
    from lazyllm import finetune, launchers

    train_file = DATA_DIR / 'ppl_text2sql.json'
    checkpoint_dir = MODEL_DIR / 'checkpoint'

    if checkpoint_dir.exists():
        log('  模型已存在，跳过训练')
        return True

    model = (
        lazyllm.TrainableModule(
            SFT_BASE_MODEL, target_path=str(checkpoint_dir)
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
    log(f'  模型保存: {checkpoint_dir}')
    return True


def step4_inference(step_label='[4/5]'):
    log_step(f'{step_label} 运行评测集推理...')

    ensure_local_site_packages()

    import lazyllm
    from lazyllm import deploy

    test_file = DATA_DIR / TEST_DATA_FILE
    inference_output = OUTPUT_DIR / 'inference_results.json'

    model_path = find_latest_merge_model(MODEL_DIR)
    if not model_path:
        log(f'  错误: 在 {MODEL_DIR} 下未找到 lazyllm_merge 目录')
        return False
    log(f'  找到模型: {model_path}')

    if inference_output.exists():
        log('  推理结果已存在，跳过推理')
        return True

    log('  加载测试数据...')
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as file:
        for line in file:
            test_data.append(json.loads(line))
    log(f'  测试样本: {len(test_data)} 条')

    log('  加载训练好的模型...')
    model = lazyllm.TrainableModule(str(model_path)).deploy_method(deploy.vllm)
    model.start()

    sys_prompt = (
        'You are a SQL expert. Based on the database schema provided, '
        'generate a SQL query to answer the question. Return ONLY the SQL '
        'query without any explanation.'
    )

    try:
        log('  开始推理...')
        results = []
        for index, item in enumerate(test_data):
            schema = item.get('schema', '')
            question = item.get('question', '')
            gold_sql = item.get('gold_sql', '')

            prompt = (
                f'{sys_prompt}\n\nDatabase Schema:\n{schema}\n\nQuestion: '
                f'{question}'
            )
            response = model(prompt)
            response_text = response if isinstance(response, str) else str(response)
            extracted_sql = extract_sql(response_text)

            results.append(
                {
                    'test_case_id': index,
                    'db_id': item.get('db_id', ''),
                    'question': question,
                    'schema': schema,
                    'gold_sql': gold_sql,
                    'raw_response': response_text,
                    'predicted_sql': extracted_sql,
                }
            )

            if (index + 1) % 10 == 0:
                log(f'    已处理: {index + 1}/{len(test_data)}')

        with open(inference_output, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
    finally:
        model.stop()

    log(f'  推理完成: {inference_output}')
    return True


def step5_evaluation(step_label='[5/5]'):
    log_step(f'{step_label} 运行 Text2SQL 评估...')

    inference_file = OUTPUT_DIR / 'inference_results.json'
    report_path = OUTPUT_DIR / 'evaluation_report.json'

    if report_path.exists():
        log('  评估报告已存在，跳过评估')
        with open(report_path, 'r', encoding='utf-8') as file:
            report = json.load(file)
        summary = report['summary']
        log(f"  平均总分: {summary['avg_overall_score']:.2f}/5.0")
        log(f"  语义得分: {summary['avg_semantic_score']:.2f}/5.0")
        log(f"  语法得分: {summary['avg_syntax_score']:.2f}/5.0")
        log(f"  等价得分: {summary['avg_equivalence_score']:.2f}/3.0")
        log(f"  正确率: {summary['accuracy'] * 100:.1f}%")
        return True

    sys.path.insert(0, LAZYLLM_PATH)

    with open(inference_file, 'r', encoding='utf-8') as file:
        inference_data = json.load(file)

    log(f'  加载推理结果: {len(inference_data)} 条')

    judge = SQLJudge(
        JUDGE_MODEL,
        CONFIG['judge_max_model_len'],
        CONFIG['judge_gpu_memory_utilization'],
        CONFIG['judge_max_num_seqs'],
        CONFIG['judge_response_max_tokens'],
    )

    results = [None] * len(inference_data)

    def evaluate_single(i, item):
        question = item.get('question', '')
        gold_sql = item.get('gold_sql', '')
        pred_sql = item.get('predicted_sql', '')
        eval_result = judge.evaluate(question, gold_sql, pred_sql)
        return i, {
            'test_case_id': item.get('test_case_id', i),
            'question': (
                question[:100] + '...' if len(question) > 100 else question
            ),
            'gold_sql': (
                gold_sql[:200] + '...' if len(gold_sql) > 200 else gold_sql
            ),
            'predicted_sql': (
                pred_sql[:200] + '...'
                if len(pred_sql) > 200
                else pred_sql
            ),
            'evaluation': eval_result,
        }

    num_workers = max(1, min(CONFIG['judge_workers'], len(inference_data)))
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(evaluate_single, i, item)
                for i, item in enumerate(inference_data)
            ]

            for done_count, future in enumerate(as_completed(futures), 1):
                idx, result = future.result()
                results[idx] = result
                if (
                    done_count % 20 == 0
                    or done_count == len(inference_data)
                ):
                    log(f'    已评估: {done_count}/{len(inference_data)}')
    finally:
        judge.model.stop()

    total = len(results)
    correct_count = sum(1 for item in results if item['evaluation']['is_correct'])
    scores = [item['evaluation']['overall_score'] for item in results]
    semantic_scores = [
        item['evaluation']['semantic_score'] for item in results
    ]
    syntax_scores = [item['evaluation']['syntax_score'] for item in results]
    equivalence_scores = [
        item['evaluation']['equivalence_score'] for item in results
    ]

    stats = {
        'total_samples': total,
        'correct_count': correct_count,
        'accuracy': correct_count / total if total > 0 else 0,
        'avg_overall_score': sum(scores) / len(scores) if scores else 0,
        'avg_semantic_score': (
            sum(semantic_scores) / len(semantic_scores)
            if semantic_scores
            else 0
        ),
        'avg_syntax_score': (
            sum(syntax_scores) / len(syntax_scores) if syntax_scores else 0
        ),
        'avg_equivalence_score': (
            sum(equivalence_scores) / len(equivalence_scores)
            if equivalence_scores
            else 0
        ),
    }

    log(f'  评估完成: {total} 条样本')
    log(f"  平均总分: {stats['avg_overall_score']:.2f}/5.0")
    log(f"  语义得分: {stats['avg_semantic_score']:.2f}/5.0")
    log(f"  语法得分: {stats['avg_syntax_score']:.2f}/5.0")
    log(f"  等价得分: {stats['avg_equivalence_score']:.2f}/3.0")
    log(f"  正确率: {stats['accuracy'] * 100:.1f}%")

    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump({'summary': stats, 'details': results}, file, indent=2)

    log(f'  报告保存: {report_path}')
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='一键 Text2SQL 训练脚本')
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
        help='Pipeline 模型路径',
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
        help='Pipeline 阶段处理的训练样本数量上限',
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
        help='只运行指定步骤 (1-5)',
    )
    return parser.parse_args()


def init_config(args):
    global CONFIG
    global LAZYLLM_PATH, PIPELINE_MODEL, SFT_BASE_MODEL, JUDGE_MODEL
    global DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR

    CONFIG = {
        'lazyllm_path': (
            args.lazyllm_path if args.lazyllm_path is not None else LAZYLLM_PATH
        ),
        'pipeline_model': (
            args.pipeline_model
            if args.pipeline_model is not None
            else PIPELINE_MODEL
        ),
        'sft_base_model': (
            args.sft_base_model
            if args.sft_base_model is not None
            else SFT_BASE_MODEL
        ),
        'judge_model': (
            args.judge_model if args.judge_model is not None else JUDGE_MODEL
        ),
        'data_dir': Path(args.data_dir) if args.data_dir else DATA_DIR,
        'model_dir': Path(args.model_dir) if args.model_dir else MODEL_DIR,
        'output_dir': Path(args.output_dir) if args.output_dir else OUTPUT_DIR,
        'log_dir': Path(args.log_dir) if args.log_dir else LOG_DIR,
        'pipeline_limit': (
            args.pipeline_limit
            if args.pipeline_limit is not None
            else PIPELINE_LIMIT
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
    PIPELINE_MODEL = CONFIG['pipeline_model']
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

    model_paths = [
        ('LAZYLLM_PATH', config['lazyllm_path']),
        ('PIPELINE_MODEL', config['pipeline_model']),
        ('SFT_BASE_MODEL', config['sft_base_model']),
        ('JUDGE_MODEL', config['judge_model']),
    ]
    for name, path in model_paths:
        if not Path(path).exists():
            log_error(f'{name} 不存在: {path}')
            log(f'请使用 --{name.lower().replace("_", "-")} 参数指定正确路径')
            safe_exit(1)

    log('==========================================')
    log('一键Text2SQL训练脚本')
    log('==========================================')
    log('')
    log_info('配置信息:')
    log(f'  - 基础目录: {BASE_DIR}')
    log(f"  - LazyLLM路径: {config['lazyllm_path']}")
    log(f"  - Pipeline模型: {config['pipeline_model']}")
    log(f"  - SFT模型: {config['sft_base_model']}")
    log(f"  - Judge模型: {config['judge_model']}")
    log(f"  - 数据目录: {config['data_dir']}")
    log(f"  - 模型目录: {config['model_dir']}")
    log(f"  - 输出目录: {config['output_dir']}")
    log(f"  - 日志目录: {config['log_dir']}")
    log(f"  - PIPELINE_LIMIT: {config['pipeline_limit']}")
    log(f"  - JUDGE_WORKERS: {config['judge_workers']}")
    log(f"  - JUDGE_MAX_MODEL_LEN: {config['judge_max_model_len']}")
    log(
        '  - JUDGE_GPU_MEMORY_UTILIZATION: '
        f"{config['judge_gpu_memory_utilization']}"
    )
    log(f"  - JUDGE_MAX_NUM_SEQS: {config['judge_max_num_seqs']}")
    log(
        '  - JUDGE_RESPONSE_MAX_TOKENS: '
        f"{config['judge_response_max_tokens']}"
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
        ('下载数据', step1_prepare_data),
        ('Pipeline处理', step2_text2sql_pipeline),
        ('SFT训练', step3_sft_training),
        ('评测推理', step4_inference),
        ('Text2SQL评估', step5_evaluation),
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
    log(f'  数据目录: {config["data_dir"]}')
    log(f'  模型目录: {config["model_dir"]}/checkpoint')
    log(f'  推理结果: {config["output_dir"]}/inference_results.json')
    log(f'  评估报告: {config["output_dir"]}/evaluation_report.json')
    log(f'  日志文件: {LOG_FILE}')
    log('')

    report_file = config['output_dir'] / 'evaluation_report.json'
    if report_file.exists():
        log_info('评估统计:')
        with open(report_file, 'r', encoding='utf-8') as file:
            report = json.load(file)
        summary = report['summary']
        log(f"  - 总样本: {summary['total_samples']}")
        log(f"  - 平均总分: {summary['avg_overall_score']:.2f}/5.0")
        log(f"  - 语义得分: {summary['avg_semantic_score']:.2f}/5.0")
        log(f"  - 语法得分: {summary['avg_syntax_score']:.2f}/5.0")
        log(f"  - 等价得分: {summary['avg_equivalence_score']:.2f}/3.0")
        log(f"  - 正确率: {summary['accuracy'] * 100:.1f}%")

    log('')
    log('==========================================')
    safe_exit(0)


if __name__ == '__main__':
    main()
