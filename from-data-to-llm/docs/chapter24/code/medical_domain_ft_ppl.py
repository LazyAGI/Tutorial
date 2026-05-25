import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

import lazyllm
from lazyllm import LOG, finetune, launchers, pipeline
from lazyllm.tools.data.pipelines.domain_finetune_pipelines import (
    build_domain_finetune_pipeline,
    build_train_test_split_pipeline,
)

# 确保 LazyLLM 在路径中
_LAZYLLM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../LazyLLM')
)
if _LAZYLLM_ROOT not in sys.path:
    sys.path.insert(0, _LAZYLLM_ROOT)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

HUATUO_DATASET_NAME = 'FreedomIntelligence/HuatuoGPT-sft-data-v1'

MEDICAL_INSTRUCTION_ZH = (
    '你是一位专业的医疗信息助手。'
    '请注意：提供的信息仅供参考，'
    '不构成医疗建议、诊断或治疗。'
    '如有医疗问题，请咨询专业医生。'
)

MEDICAL_FILTERS = [
    {'type': 'char_count', 'min_chars': 200, 'max_chars': 20000},
    {'type': 'null_content'},
]

OUTPUT_KEY = 'formatted_text'


def load_huatuo(
    dataset_name: str = HUATUO_DATASET_NAME,
    split: str = 'train',
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if load_dataset is None:
        raise ImportError('请先安装 datasets：pip install datasets')

    LOG.info(f'正在从 HuggingFace 加载 {dataset_name} [{split}]...')
    try:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f'加载 {dataset_name} 失败: {e}\n'
            '可能需要设置 HF_ENDPOINT 或确认数据集名称正确。'
        )

    raw_n = len(ds)
    if max_samples and raw_n > max_samples:
        ds = ds.select(range(max_samples))
        LOG.info(f'截取前 {max_samples} 条原始对话'
                 f'（原始 {raw_n} 条）')

    items = [dict(row) for row in ds]
    LOG.info(f'加载完成：{len(items)} 条原始对话记录')
    return items


def build_medical_dataset(
    raw_items: List[Dict[str, Any]],
    output_format: str = 'alpaca',
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = 'dataset/huatuo_ft',
    llm=None,
) -> Dict[str, Any]:
    if not raw_items:
        raise ValueError('raw_items 为空，请先加载数据集')

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'medical_formatted.jsonl')

    _sep = '=' * 60
    print(f'\n{_sep}')
    print('HuatuoGPT-sft-data-v1  →  微调数据集构建')
    print(_sep)
    print(f'  输入条数  : {len(raw_items)}')
    print(f'  输出格式  : {output_format}')
    print(f'  输出目录  : {output_dir}')
    print(_sep)

    core_pipeline = build_domain_finetune_pipeline(
        domain='medical',
        input_key='content',
        output_key=OUTPUT_KEY,
        enabled={
            'conversation_expand': True,
            'normalization': True,
            'deduplication': True,
            'output_quality_filter': True,
            'llm_cleaning': False,
        },
        options={
            'llm': llm,
            'dedup_method': 'minhash',
            'min_output_chars': 80,
            'min_output_input_ratio': 0.3,
            'expand_list_key': 'data',
            'expand_q_prefix': '问：',
            'expand_a_prefix': '答：',
            'expand_min_q_chars': 8,
            'expand_min_a_chars': 50,
        },
        output_format=output_format,
        language='zh',
        filters_config=MEDICAL_FILTERS,
        normalization_instruction=MEDICAL_INSTRUCTION_ZH,
    )
    split_pipeline = build_train_test_split_pipeline(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    def _save(data_dict: dict) -> dict:
        with open(output_file, 'w', encoding='utf-8') as f:
            for split_name, items in data_dict.items():
                for item in items:
                    record = {
                        'split': split_name,
                        **(
                            {OUTPUT_KEY: item[OUTPUT_KEY]}
                            if OUTPUT_KEY in item else item
                        ),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        LOG.info(f'全量结果已保存: {output_file}')
        return data_dict

    with pipeline() as ppl:
        ppl.process = core_pipeline
        ppl.split = split_pipeline
        ppl.save = _save

    result = ppl(raw_items)

    train_cnt = len(result.get('train', []))
    val_cnt = len(result.get('validation', []))
    test_cnt = len(result.get('test', []))
    total = train_cnt + val_cnt + test_cnt

    print(f'\n  过滤后保留 : {total} 条'
          f'（过滤/去重 {len(raw_items) - total} 条）')
    print(f'  划分       : train={train_cnt}, '
          f'validation={val_cnt}, test={test_cnt}')

    train_file = os.path.join(output_dir, 'medical_train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in result.get('train', []):
            ft = item.get(OUTPUT_KEY)
            if isinstance(ft, dict):
                record = {k: ft.get(k, '') for k in (
                    'instruction', 'input', 'output'
                )}
            elif isinstance(ft, str):
                record = {'text': ft}
            else:
                record = item
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    LOG.info(f'训练集已保存: {train_file}（{train_cnt} 条）')

    test_file = os.path.join(output_dir, 'medical_test.jsonl')
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in result.get('test', []):
            ft = item.get(OUTPUT_KEY)
            if isinstance(ft, dict):
                record = {k: ft.get(k, '') for k in (
                    'instruction', 'input', 'output'
                )}
            elif isinstance(ft, str):
                record = {'text': ft}
            else:
                record = item
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    LOG.info(f'测试集已保存: {test_file}（{test_cnt} 条）')

    print('\n--- 样例输出（前 3 条训练集）---')
    samples = result.get('train', [])[:3]
    if samples:
        for i, item in enumerate(samples, 1):
            ft = item.get(OUTPUT_KEY, item)
            print(f'\n[样例 {i}]')
            print(json.dumps(ft, ensure_ascii=False, indent=2))
    else:
        print('（无训练样例，请检查过滤条件是否过严）')
    print('---\n')

    return {
        'train_file': train_file,
        'test_file': test_file,
        'output_file': output_file,
        'output_dir': output_dir,
        'counts': {
            'train': train_cnt, 'validation': val_cnt, 'test': test_cnt
        },
    }


def cosine(x: List[float], y: List[float]) -> float:
    if np is None:
        return 0.0
    product = np.dot(x, y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    raw = product / norm if norm != 0 else 0.0
    return max(0.0, min(float(raw), 1.0))


def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(test_file):
        return []
    items = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    items.append({
                        'instruction': '', 'input': '', 'output': obj['text']
                    })
                else:
                    items.append({
                        'instruction': obj.get('instruction', ''),
                        'input': obj.get('input', ''),
                        'output': obj.get('output', ''),
                    })
            except json.JSONDecodeError:
                continue
    return items


def build_eval_data_from_test(
    test_file: str,
    max_samples: Optional[int] = None,
    default_instruction: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    items = load_test_data(test_file)
    if max_samples:
        items = items[:max_samples]
    default_instruction = default_instruction or ''
    eval_data = []
    eval_set = []
    for item in items:
        instr = item.get('instruction', '') or default_instruction
        inp = item.get('input', '')
        prompt = instr + '\n\n' + inp if instr and inp else (instr or inp)
        eval_data.append(prompt)
        eval_set.append({
            'instruction': instr,
            'input': inp,
            'answers': item.get('output', '')
        })
    return eval_data, eval_set


def _normalize_for_embed(s: str) -> str:
    if not s:
        return ''
    return ' '.join(s.strip().split())


def calculate_score(
    results: List[Dict[str, Any]],
    embedding_model: str = 'BAAI/bge-large-zh-v1.5',
    detail_path: Optional[str] = None,
    summary_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    assert results, 'results 为空'
    infer_set = [
        _normalize_for_embed(r.get('predicted', '') or '') for r in results
    ]
    eval_set = [
        {'answers': _normalize_for_embed(r.get('reference', '') or '')}
        for r in results
    ]
    n = len(eval_set)
    embed_texts = []
    for i in range(n):
        embed_texts.append(infer_set[i])
        embed_texts.append(eval_set[i]['answers'] or '')

    all_vecs = []
    if np is not None and embed_texts:
        try:
            embed_module = lazyllm.TrainableModule(embedding_model)
            embed_module.start()
            out = embed_module(embed_texts)
            all_vecs = json.loads(out) if isinstance(out, str) else (out or [])
        except Exception as e:
            LOG.warning(f'Embedding 失败，Cosine 置 0: {e}')

    accu_cosine = 0.0
    detail_list = []
    for i in range(n):
        cosine_score = 0.0
        if len(all_vecs) >= 2 * (i + 1):
            try:
                cosine_score = cosine(all_vecs[2 * i], all_vecs[2 * i + 1])
            except Exception:
                pass
        accu_cosine += cosine_score
        detail_list.append({
            'true': eval_set[i]['answers'],
            'infer': infer_set[i],
            'cosine_score': cosine_score,
        })

    total = n
    cosine_ratio = accu_cosine / total if total else 0.0
    metrics = {'cosine_score': cosine_ratio, 'num_samples': total}

    if detail_path:
        os.makedirs(os.path.dirname(detail_path) or '.', exist_ok=True)
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(detail_list, f, ensure_ascii=False, indent=2)
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or '.', exist_ok=True)
        with open(summary_path, 'a', encoding='utf-8') as f:
            json.dump({
                'cosine_score': [cosine_ratio, round(cosine_ratio, 4) * 100]
            }, f, ensure_ascii=False)
            f.write('\n')

    score_str = (f'Cosine Score: {accu_cosine}/{total}, '
                 f'{round(cosine_ratio, 4) * 100}%\n')
    return metrics, score_str


def _pred_to_str(x: Any) -> str:
    if x is None:
        return ''
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        return (
            x.get('content') or x.get('text') or x.get('output') or ''
        ).strip()
    return str(x).strip()


def run_model_eval(model_path: str, eval_data: List[str]) -> List[str]:
    model = lazyllm.TrainableModule(model_path)
    model.evalset(eval_data)
    model.start()
    model.eval()
    raw = model.eval_result or []
    return [_pred_to_str(r) for r in raw]


def evaluate_llm_effect(
    test_file: str,
    base_model: str,
    finetuned_model: Any = None,
    max_eval_samples: Optional[int] = 100,
    output_dir: Optional[str] = None,
    embedding_model: str = 'BAAI/bge-large-zh-v1.5',
    default_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    eval_data, eval_set = build_eval_data_from_test(
        test_file, max_eval_samples, default_instruction=default_instruction
    )
    if not eval_data:
        LOG.warning(f'测试集为空或文件不存在: {test_file}')
        return {}

    n_use = len(eval_data)
    print(f'\n开始效果测试（测试集 {n_use} 条）')
    print(f'  基座模型: {base_model}')
    if finetuned_model:
        print(f'  微调模型: {finetuned_model}' if isinstance(
            finetuned_model, str) else '  微调模型: 已传入实例')

    print('\n>>> 微调前：基座模型推理...')
    infer_before = run_model_eval(base_model, eval_data)
    results_before = [
        {
            'predicted': infer_before[i] if i < len(infer_before) else '',
            'reference': eval_set[i]['answers'],
            **eval_set[i]
        }
        for i in range(n_use)
    ]

    detail_before = os.path.join(
        output_dir, 'medical_eval_detail_before.json'
    ) if output_dir else None
    summary_path = os.path.join(
        output_dir, 'medical_eval_scores.jsonl'
    ) if output_dir else None
    metrics_before, score_str_before = calculate_score(
        results_before, embedding_model=embedding_model,
        detail_path=detail_before, summary_path=summary_path,
    )

    results_after = None
    metrics_after = None
    score_str_after = ''
    if finetuned_model:
        print('\n>>> 微调后：微调模型推理...')
        eval_result_pre = getattr(
            finetuned_model, 'eval_result', None
        ) if not isinstance(finetuned_model, str) else None
        if eval_result_pre is not None and len(eval_result_pre) == n_use:
            infer_after = eval_result_pre
        elif isinstance(finetuned_model, str):
            infer_after = run_model_eval(finetuned_model, eval_data)
        else:
            finetuned_model.evalset(eval_data)
            finetuned_model.eval()
            infer_after = finetuned_model.eval_result or []
        infer_after_str = [_pred_to_str(x) for x in infer_after]
        results_after = [
            {
                'predicted': (
                    infer_after_str[i] if i < len(infer_after_str) else ''
                ),
                'reference': eval_set[i]['answers'],
                **eval_set[i]
            }
            for i in range(n_use)
        ]
        detail_after = os.path.join(
            output_dir, 'medical_eval_detail_after.json'
        ) if output_dir else None
        metrics_after, score_str_after = calculate_score(
            results_after, embedding_model=embedding_model,
            detail_path=detail_after, summary_path=summary_path,
        )

    print('\n' + '=' * 60)
    print(' ' * 18 + '效果测试结果')
    print('=' * 60)
    print(f'  样本数: {n_use}')
    print('  --- 微调前（基座）---')
    print(score_str_before)
    if metrics_after is not None:
        print('  --- 微调后 ---')
        print(score_str_after)
    print('=' * 60 + '\n')

    out = {
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'n_eval': n_use
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, 'medical_eval_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def run_finetune(
    base_model: str,
    train_data_path: str,
    output_dir: str = 'medical_finetune_output',
    num_epochs: int = 3,
    per_device_batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4,
    cutoff_len: int = 2048,
    warmup_ratio: float = 0.05,
    ngpus: int = 1,
    eval_data: Optional[List[str]] = None,
):
    print(f'\n{"=" * 60}')
    print('开始 LLM 微调（医疗领域）')
    print(f'  基座模型: {base_model}')
    print(f'  训练数据: {train_data_path}')
    print(f'  输出目录: {output_dir}')
    print(f'  num_epochs={num_epochs}, lr={learning_rate}, '
          f'batch={per_device_batch_size}, '
          f'grad_accum={gradient_accumulation_steps}, '
          f'cutoff_len={cutoff_len}, warmup_ratio={warmup_ratio}')
    if eval_data:
        print(f'  evalset: {len(eval_data)} 条')
    print(f'{"=" * 60}\n')

    model = lazyllm.TrainableModule(base_model) \
        .mode('finetune') \
        .trainset(train_data_path) \
        .finetune_method((
            finetune.auto,
            {
                'launcher': launchers.empty(ngpus=ngpus),
                'num_train_epochs': num_epochs,
                'per_device_train_batch_size': per_device_batch_size,
                'learning_rate': learning_rate,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'cutoff_len': cutoff_len,
                'warmup_ratio': warmup_ratio,
            }
        ))
    if eval_data:
        model.evalset(eval_data)
    model.update()
    print('微调完成！')
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='HuatuoGPT 医疗领域微调 Pipeline（LazyLLM）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python medical_domain_ft_ppl.py --build_dataset --max_samples 500   # 试跑
  python medical_domain_ft_ppl.py --build_dataset                   # 全量
  python medical_domain_ft_ppl.py --build_dataset --train_flag      #
    构建+微调
  python medical_domain_ft_ppl.py --eval_test --max_eval_samples 50 #
    效果测试
        """,
    )

    data = parser.add_argument_group('数据参数')
    data.add_argument('--build_dataset', action='store_true',
                      help='构建数据集')
    data.add_argument('--dataset_name', type=str,
                      default=HUATUO_DATASET_NAME,
                      help='HuggingFace 数据集名称')
    data.add_argument('--split', type=str, default='train')
    data.add_argument('--max_samples', type=int, default=None,
                      help='最多加载条数，None=全量')
    data.add_argument('--output_dir', type=str,
                      default='./dataset/huatuo_ft')

    ppl = parser.add_argument_group('Pipeline 参数')
    ppl.add_argument('--output_format', type=str, default='alpaca',
                     choices=['alpaca', 'sharegpt', 'chatml', 'raw'])
    ppl.add_argument('--train_ratio', type=float, default=0.8)
    ppl.add_argument('--validation_ratio', type=float, default=0.1)
    ppl.add_argument('--test_ratio', type=float, default=0.1)

    model = parser.add_argument_group('模型参数')
    model.add_argument(
        '--base_model', type=str,
        default='Qwen/Qwen2.5-14B-Instruct'
    )
    model.add_argument('--train_flag', action='store_true',
                       help='执行微调')

    eval_group = parser.add_argument_group('效果测试参数')
    eval_group.add_argument('--eval_test', action='store_true')
    eval_group.add_argument('--max_eval_samples', type=int, default=100)
    eval_group.add_argument('--finetuned_model_path', type=str, default=None)

    train = parser.add_argument_group('训练参数')
    train.add_argument('--num_epochs', type=int, default=3)
    train.add_argument('--per_device_batch_size', type=int, default=4)
    train.add_argument('--learning_rate', type=float, default=2e-5)
    train.add_argument('--gradient_accumulation_steps', type=int, default=4)
    train.add_argument('--cutoff_len', type=int, default=2048)
    train.add_argument('--warmup_ratio', type=float, default=0.05)
    train.add_argument('--ngpus', type=int, default=1)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    print(f'\n{"=" * 80}')
    print(' ' * 10 + 'HuatuoGPT-sft-data-v1  '
          '医疗领域微调 Pipeline（LazyLLM）')
    print(f'{"=" * 80}\n')
    llm = lazyllm.TrainableModule(args.base_model)
    train_file = os.path.join(args.output_dir, 'medical_train.jsonl')
    test_file = os.path.join(args.output_dir, 'medical_test.jsonl')

    if args.build_dataset:
        print('>>> 步骤 1：从 HuggingFace 加载 HuatuoGPT-sft-data-v1')
        raw_items = load_huatuo(
            dataset_name=args.dataset_name,
            split=args.split,
            max_samples=args.max_samples,
        )
        print('\n>>> 步骤 2：Pipeline 处理'
              '（归一化 → 过滤 → 去重 → 格式化 → 划分）')
        paths = build_medical_dataset(
            raw_items=raw_items,
            output_format=args.output_format,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            output_dir=args.output_dir,
            llm=llm,
        )
        train_file = paths['train_file']
        test_file = paths.get('test_file', test_file)
        counts = paths['counts']
        print(f'数据集构建完成！train={counts["train"]}, '
              f'validation={counts["validation"]}, test={counts["test"]}')

    finetuned_model = None
    if args.train_flag:
        if not os.path.exists(train_file):
            print('\n错误：训练数据不存在，'
                  '请先运行 --build_dataset')
            return
        eval_data = None
        if args.eval_test and os.path.isfile(test_file):
            eval_data, _ = build_eval_data_from_test(
                test_file, args.max_eval_samples,
                default_instruction=MEDICAL_INSTRUCTION_ZH
            )
        print('\n>>> 步骤 3：执行 LLM 微调')
        finetuned_model = run_finetune(
            base_model=args.base_model,
            train_data_path=train_file,
            output_dir=os.path.join(args.output_dir, 'finetuned_model'),
            num_epochs=args.num_epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cutoff_len=args.cutoff_len,
            warmup_ratio=args.warmup_ratio,
            ngpus=args.ngpus,
            eval_data=eval_data,
        )

    if args.eval_test:
        if not os.path.isfile(test_file):
            print(f'\n错误：测试集不存在，'
                  f'请先运行 --build_dataset 生成 {test_file}')
            return
        finetuned_for_eval = finetuned_model
        if finetuned_for_eval is None and args.finetuned_model_path:
            finetuned_for_eval = args.finetuned_model_path
        if finetuned_for_eval is None:
            p = os.path.join(args.output_dir, 'finetuned_model')
            if os.path.exists(p):
                finetuned_for_eval = p
        print('\n>>> 效果测试：微调前 vs 微调后'
              '（Cosine 语义相似度）')
        evaluate_llm_effect(
            test_file=test_file,
            base_model=args.base_model,
            finetuned_model=finetuned_for_eval,
            max_eval_samples=args.max_eval_samples,
            output_dir=args.output_dir,
            default_instruction=MEDICAL_INSTRUCTION_ZH,
        )

    if not args.build_dataset and not args.train_flag and not args.eval_test:
        print('请指定 --build_dataset / --train_flag '
              '/ --eval_test 至少其一')
        print('示例：python medical_domain_ft_ppl.py '
              '--build_dataset --max_samples 500')


if __name__ == '__main__':
    main(parse_args())
