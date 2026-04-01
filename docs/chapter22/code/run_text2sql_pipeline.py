#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

LAZYLLM_PATH = '/path/to/your/lazyllm'
PIPELINE_MODEL = '/path/to/pipeline/model'
SFT_BASE_MODEL = '/path/to/sft/base/model'
JUDGE_MODEL = '/path/to/judge/model'
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
    local_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
    if local_path not in sys.path:
        sys.path.insert(0, local_path)


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
        prompt = JUDGE_PROMPT.format(
            question=question,
            gold_sql=gold_sql,
            pred_sql=pred_sql,
        )
        try:
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
{
    "semantic_score": 5,
    "syntax_score": 5,
    "equivalence_score": 3,
    "overall_score": 5.0,
    "is_correct": true,
    "reason": "SQL 完全正确，正确理解了用户意图"
}
```'''


def step1_prepare_data(step_label='[1/5]'):
    log_step(f'{step_label} 下载 Text2SQL 数据集...')

    train_path = DATA_DIR / 'train_text2sql.json'
    test_path = DATA_DIR / 'test_text2sql.jsonl'

    if train_path.exists() and test_path.exists():
        log('  数据已存在，跳过下载')
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        log_error('请先安装 datasets: pip install datasets')
        return False

    log('  正在从 Hugging Face 加载数据集 rirqing/text2sql...')
    dataset = load_dataset('rirqing/text2sql', trust_remote_code=True)

    def normalize_item(idx, item):
        return {
            'db_id': item.get('db_id', f'db_{idx}'),
            'question': item.get('question', ''),
            'schema': item.get('schema', ''),
            'gold_sql': item.get('gold_sql', ''),
            'prompt': item.get('prompt', ''),
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
            'output': item.get('output', ''),
        }

    train_data = [
        normalize_item(idx, item) for idx, item in enumerate(dataset['train'])
    ]
    test_data = [
        normalize_item(idx, item) for idx, item in enumerate(dataset['test'])
    ]

    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, ensure_ascii=False, indent=2)

    with open(test_path, 'w', encoding='utf-8') as file:
        for item in test_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

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

    input_file = DATA_DIR / 'train_text2sql.json'
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

    test_file = DATA_DIR / 'test_text2sql.jsonl'
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

    if not Path(config['lazyllm_path']).exists():
        log_error(f"LAZYLLM_PATH 不存在: {config['lazyllm_path']}")
        log('请修改脚本中的 LAZYLLM_PATH 配置')
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
