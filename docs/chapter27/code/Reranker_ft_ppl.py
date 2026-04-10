"""
Reranker 微调完整 Pipeline - 基于 LazyLLM
结合 lazyllm.tools.data.pipelines.reranker_pipelines 与
operators.reranker_synthesis，支持两种输入方式并实现自动化流程。

功能：
1. 两种输入：① 用户输入数据（本地 JSON/JSONL） ② HuggingFace 数据集名（自动下载）
2. 数据准备与 Pipeline：划分、难负样本挖掘、格式化
3. Reranker 微调与排序/检索+重排评估
"""

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import lazyllm
from lazyllm import LOG, Document, Retriever, Reranker, finetune, launchers
from lazyllm.tools.eval import NonLLMContextRecall, ContextRelevance

from lazyllm.tools.data.pipelines.reranker_pipelines import (
    build_reranker_dataformatter_pipeline,
    build_reranker_hard_neg_pipeline,
    build_convert_from_embed_pipeline,
)
from lazyllm.tools.data.operators.reranker_synthesis import (
    RerankerTrainTestSplitter,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_json(file_path: str, line_by_line: bool = True) -> list:
    """加载 JSON/JSONL 文件。"""
    if line_by_line:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_data_path(dir_name: str, file_name: str) -> str:
    """构建数据路径并创建目录。"""
    data_root = os.path.join(os.getcwd(), dir_name)
    os.makedirs(data_root, exist_ok=True)
    return os.path.join(data_root, file_name)


# ---------------------------------------------------------------------------
# FiQA 原始数据提取（Pipeline 未用 HuggingFace/用户数据时使用）
# ---------------------------------------------------------------------------

def _extract_raw_pairs_from_fiqa(
    queries_file: str,
    corpus_file: str,
    train_file: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """从 FiQA 文件提取 queries / corpus / train_pairs。"""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            queries[item['_id']] = item['text']

    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus[item['_id']] = item['text']

    train_pairs: Dict[str, List[str]] = {}
    with open(train_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, cid = parts[0], parts[1]
                if qid not in train_pairs:
                    train_pairs[qid] = []
                train_pairs[qid].append(cid)

    return queries, corpus, train_pairs


# ---------------------------------------------------------------------------
# 两种输入：用户数据 与 HuggingFace 数据集名
# ---------------------------------------------------------------------------

def load_from_user_data(
    data_path: str,
    query_key: str = 'query',
    pos_key: str = 'pos',
    line_by_line: bool = True,
) -> Tuple[List[dict], List[str]]:
    """从本地文件加载 (query, pos) 及语料列表。"""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f'数据文件不存在: {data_path}')

    raw = load_json(str(path), line_by_line=line_by_line)
    if not raw:
        raise ValueError(f'文件为空或格式错误: {data_path}')

    raw_items = []
    corpus_texts = []
    for item in raw:
        q = item.get(query_key) or item.get('query')
        p = item.get(pos_key) or item.get('pos') or item.get('context')
        if p is not None and not isinstance(p, list):
            p = [p]
        if not q or not p:
            continue
        raw_items.append({'query': q, 'pos': p})
        for t in p:
            if t and t not in corpus_texts:
                corpus_texts.append(t)

    msg = (
        f'从用户数据加载: {data_path} -> {len(raw_items)} 条语料 '
        f'{len(corpus_texts)} 篇'
    )
    LOG.info(msg)
    return raw_items, corpus_texts


def load_from_huggingface(
    dataset_name: str,
    split: str = 'train',
    query_column: str = 'question',
    pos_column: str = 'context',
    config: Optional[str] = None,
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[dict], List[str]]:
    """从 HuggingFace 下载并转为 (query, pos) 列表及语料。"""
    if load_dataset is None:
        raise ImportError('请安装 datasets: pip install datasets')

    kwargs = {'path': dataset_name, 'split': split}
    if config:
        kwargs['config'] = config
    if subset:
        kwargs['name'] = subset

    ds = load_dataset(**kwargs)
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    cols = ds.column_names
    if query_column in cols:
        q_col = query_column
    elif 'query' in cols:
        q_col = 'query'
    else:
        q_col = 'question'
    if pos_column in cols:
        p_col = pos_column
    elif 'pos' in cols:
        p_col = 'pos'
    else:
        p_col = 'context'

    raw_items = []
    corpus_texts = []
    for item in ds:
        q = item.get(q_col) or item.get('query')
        p = item.get(p_col) or item.get('pos') or item.get('context')
        if p is not None and not isinstance(p, list):
            p = [p]
        if not q or not p:
            continue
        raw_items.append({'query': q, 'pos': p})
        for t in p:
            if t and t not in corpus_texts:
                corpus_texts.append(t)

    msg = f'从 HuggingFace 加载: {dataset_name} -> {len(raw_items)} 条语料'
    LOG.info(msg)
    return raw_items, corpus_texts


def load_from_fiqa(
    queries_file: str,
    corpus_file: str,
    train_file: str,
) -> Tuple[List[dict], List[str]]:
    """从 FiQA 文件加载并提取 (query, pos) 及语料。仅做提取。"""
    queries, corpus, train_pairs = _extract_raw_pairs_from_fiqa(
        queries_file, corpus_file, train_file
    )
    raw_items = []
    for qid, pos_ids in train_pairs.items():
        if qid not in queries:
            continue
        pos_texts = [corpus[cid] for cid in pos_ids if cid in corpus]
        if pos_texts:
            raw_items.append({'query': queries[qid], 'pos': pos_texts[:1]})
    corpus_texts = list(corpus.values())
    LOG.info(f'从 FiQA 加载: {len(raw_items)} 条，语料 {len(corpus_texts)} 篇')
    return raw_items, corpus_texts


# ---------------------------------------------------------------------------
# 数据构建：仅 Pipeline（划分 → 难负样本 → 格式化 → 保存），不包含数据加载
# ---------------------------------------------------------------------------

def build_dataset_with_pipelines(
    raw_items: List[dict],
    corpus_texts: Optional[List[str]] = None,
    neg_num: int = 7,
    test_size: float = 0.1,
    seed: int = 1314,
    mining_strategy: str = 'random',
    output_format: str = 'flagreranker',
    train_group_size: int = 8,
    embedding_serving=None,
    bm25_ratio: float = 0.5,
    output_subdir: str = 'dataset',
) -> Tuple[str, str, str]:
    if not raw_items:
        msg = 'raw_items 为空，请先通过 load_from_fiqa / '
        msg += 'load_from_user_data / load_from_huggingface 加载'
        raise ValueError(msg)
    if corpus_texts is None:
        corpus_texts = list({
            p for item in raw_items for p in (item.get('pos') or [])
        })

    print('\n' + '=' * 60)
    print('Reranker Pipeline: 划分 → 难负样本 → 格式化 → 保存')
    print('=' * 60)
    print(f'\n>>> 输入: {len(raw_items)} 条语料 {len(corpus_texts)} 篇')

    # ----- Step 1: 划分 train / test -----
    print('\n>>> Step 1: 划分 train / test (RerankerTrainTestSplitter)')

    splitter = RerankerTrainTestSplitter(test_size=test_size, seed=seed)
    mixed = splitter(raw_items)

    train_items = [x for x in mixed if x.get('split') == 'train']
    test_items = [x for x in mixed if x.get('split') == 'test']
    print(f'训练 {len(train_items)} 条，测试 {len(test_items)} 条')

    # ----- Step 2: 难负样本挖掘 -----
    msg = f'\n>>> Step 2: 难负样本挖掘 (策略: {mining_strategy})'
    print(msg)

    test_corpus = list(set(
        item['pos'][0] for item in test_items if item.get('pos')
    ))
    use_corpus = corpus_texts

    hard_neg_fn = build_reranker_hard_neg_pipeline(
        input_query_key='query',
        input_pos_key='pos',
        output_neg_key='neg',
        corpus=use_corpus,
        mining_strategy=mining_strategy,
        num_negatives=neg_num,
        embedding_serving=embedding_serving,
        bm25_ratio=bm25_ratio,
        seed=seed,
    )
    train_items_with_neg = hard_neg_fn(train_items)
    print(f'难负样本挖掘完成: {len(train_items_with_neg)} 条')

    print(f'\n>>> Step 3: 数据格式化 (格式: {output_format})')

    def _flatten_and_write(formatter_result: List, out_path: str) -> int:
        flat = []
        for x in formatter_result:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in flat:
                if isinstance(item, dict) and item:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return len(flat)

    formatter_ppl = build_reranker_dataformatter_pipeline(
        input_query_key='query',
        input_pos_key='pos',
        input_neg_key='neg',
        output_format=output_format,
        train_group_size=train_group_size,
    )
    train_data_path = build_data_path(output_subdir, 'rerank_train.jsonl')
    formatted_train = formatter_ppl(train_items_with_neg)
    n_train = _flatten_and_write(formatted_train, train_data_path)
    print(f'训练数据: {n_train} 条 → {train_data_path}')

    import random
    random.seed(seed)
    eval_data_path = build_data_path(output_subdir, 'rerank_eval.jsonl')
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        for item in test_items:
            pos_set = set(item.get('pos', []))
            candidates = [doc for doc in test_corpus if doc not in pos_set]
            neg = random.sample(
                candidates, min(neg_num, len(candidates))
            ) if candidates else []
            f.write(json.dumps({
                'query': item.get('query', ''),
                'corpus': item.get('pos', []),
                'neg': neg,
            }, ensure_ascii=False) + '\n')
    print(f'评估数据: {len(test_items)} 条 → {eval_data_path}')

    kb_path = build_data_path('KB', 'rerank_kb.txt')
    with open(kb_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_corpus))
    print(f'知识库: {len(test_corpus)} 篇 → {kb_path}')
    print('\n数据集构建完成！')
    return train_data_path, eval_data_path, os.path.dirname(kb_path)


# ---------------------------------------------------------------------------
# 从 Embedding 训练数据转为 Reranker（可选）
# ---------------------------------------------------------------------------

def build_rerank_dataset_from_embed_data(
    embed_train_path: str,
    embed_eval_path: Optional[str] = None,
    neg_num: int = 7,
    output_subdir: str = 'dataset',
) -> Tuple[str, str]:
    """从 Embedding 格式训练数据转为 Reranker 格式。"""
    convert_ppl = build_convert_from_embed_pipeline(
        input_query_key='query',
        input_pos_key='pos',
        input_neg_key='neg',
        adjust_neg_count=neg_num,
        seed=1314,
    )
    raw = load_json(embed_train_path, line_by_line=True)
    converted = convert_ppl(raw)
    flat = [
        x for item in converted
        for x in (item if isinstance(item, list) else [item])
    ]

    train_path = build_data_path(output_subdir, 'rerank_train.jsonl')
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in flat:
            if isinstance(item, dict) and item:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    eval_path = None
    if embed_eval_path and os.path.exists(embed_eval_path):
        eval_data = load_json(embed_eval_path, line_by_line=True)
        eval_path = build_data_path(output_subdir, 'rerank_eval.jsonl')
        with open(eval_path, 'w', encoding='utf-8') as f:
            for item in eval_data:
                f.write(json.dumps({
                    'query': item.get('query', ''),
                    'corpus': item.get('corpus', item.get('pos', [])),
                    'neg': item.get('neg', [])[:neg_num],
                }, ensure_ascii=False) + '\n')
    return train_path, eval_path or train_path


# ---------------------------------------------------------------------------
# 模型部署与评估（参考 work/reranker_finetune.py）
# ---------------------------------------------------------------------------

def deploy_reranker_model(
    rerank_path: str,
    train_data_path: str,
    train_flag: bool = True,
    per_device_batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 6e-5,
    ngpus: int = 1,
    train_group_size: int = 8,
) -> lazyllm.TrainableModule:
    """部署带微调能力的 Reranker。"""
    sep = '=' * 60
    print(f'\n{sep}')
    status = '开启' if train_flag else '关闭'
    print(f'配置 Reranker: {rerank_path}，微调: {status}')
    print(f'{sep}\n')

    if train_flag:
        model = lazyllm.TrainableModule(rerank_path) \
            .mode('finetune') \
            .trainset(train_data_path) \
            .finetune_method((
                finetune.auto,
                {
                    'launcher': launchers.sco(ngpus=ngpus),
                    'per_device_train_batch_size': per_device_batch_size,
                    'num_train_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'train_group_size': train_group_size,
                    'query_max_len': 256,
                    'passage_max_len': 256,
                }
            ))
        model.update()
        print('微调完成。')
    else:
        model = lazyllm.TrainableModule(rerank_path)
        model.start()
    return model


def evaluate_reranker_direct(
    reranker_model: lazyllm.TrainableModule,
    eval_data_path: str,
    topk: int = 1,
) -> Tuple[float, float]:
    """直接评估 Reranker 排序：MRR、Hit@k。"""
    eval_data = load_json(eval_data_path, line_by_line=True)
    total_mrr = 0.0
    total_hits = 0
    for item in tqdm(eval_data, desc='评估'):
        query = item.get('query', '')
        pos_docs = item.get('corpus', item.get('pos', []))
        neg_docs = item.get('neg', [])
        all_docs = list(pos_docs) + list(neg_docs)
        if not all_docs:
            continue
        results = reranker_model(
            query, documents=all_docs, top_n=len(all_docs)
        )
        for rank, (idx, _) in enumerate(results, 1):
            if idx < len(pos_docs):
                total_mrr += 1.0 / rank
                if rank <= topk:
                    total_hits += 1
                break
    n = len(eval_data) or 1
    return total_mrr / n, total_hits / n


def evaluate_reranker_with_retriever(
    retriever: Retriever,
    reranker_obj: Reranker,
    eval_data_path: str,
) -> Tuple[List[Dict], float, float]:
    """检索 + 重排流程评估。"""
    eval_data = load_json(eval_data_path, line_by_line=True)
    results = []
    for item in tqdm(eval_data, desc='检索+重排'):
        query = item.get('query', '')
        retrieved = retriever(query=query)
        reranked = reranker_obj(retrieved, query=query)
        results.append({
            'question': query,
            'context_retrieved': [t.get_text() for t in reranked],
            'context_reference': item.get('corpus') or item.get('pos', []),
        })
    recall_eval = NonLLMContextRecall(binary=False)
    relevance_eval = ContextRelevance()
    return results, recall_eval(results), relevance_eval(results)


def save_results(
    score_dict: Dict[str, Any],
    train_flag: bool,
    file_path: str,
) -> None:
    """追加保存评估结果。"""
    prefix = 'ft_' if train_flag else 'origin_'
    with open(file_path, 'a', encoding='utf-8') as f:
        result = {f'{prefix}{k}': v for k, v in score_dict.items()}
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Reranker 微调完整 Pipeline（LazyLLM reranker_pipelines + 双输入）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  输入方式一 - 用户数据:
    python Reranker_ft_ppl.py --build_dataset --data_path /path/to/data.jsonl
  一键启动（构建 + 微调/部署 + 评估）:
    python Reranker_ft_ppl.py --one_click --train_flag --data_path /path/to/data.jsonl
  输入方式二 - HuggingFace:
    python Reranker_ft_ppl.py --build_dataset \
    --dataset_name virattt/financial-qa-10K
  从 Embedding 数据转换:
    python Reranker_ft_ppl.py --build_dataset --from_embed \
    --embed_train_path dataset/embed_train.json
  微调:
    python Reranker_ft_ppl.py --train_flag --rerank_path
    BAAI/bge-reranker-base --num_epochs 1
    """,
    )

    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument(
        '--one_click',
        action='store_true',
        help='一键执行全流程：构建数据集 + 部署/微调 + 评估',
    )
    data_group.add_argument(
        '--build_dataset', action='store_true', help='构建数据集'
    )
    data_group.add_argument(
        '--data_path', type=str, default=None, help='输入方式一：本地 JSON/JSONL 路径'
    )
    data_group.add_argument('--query_key', type=str, default='query')
    data_group.add_argument('--pos_key', type=str, default='pos')
    data_group.add_argument(
        '--use_fiqa', action='store_true', help='使用本地 FiQA'
    )
    data_group.add_argument(
        '--dataset_name',
        type=str,
        default='virattt/financial-qa-10K',
        help='输入方式二：HuggingFace 数据集名',
    )
    data_group.add_argument('--dataset_split', type=str, default='train')
    data_group.add_argument('--query_column', type=str, default='question')
    data_group.add_argument('--pos_column', type=str, default='context')
    data_group.add_argument('--max_samples', type=int, default=None)
    data_group.add_argument(
        '--from_embed', action='store_true', help='从 Embedding 训练数据转换'
    )
    data_group.add_argument('--embed_train_path', type=str, default=None)
    data_group.add_argument('--embed_eval_path', type=str, default=None)
    data_group.add_argument(
        '--queries_file', type=str, default='dataset/fiqa/queries.jsonl'
    )
    data_group.add_argument(
        '--corpus_file', type=str, default='dataset/fiqa/cpt/corpus.jsonl'
    )
    data_group.add_argument(
        '--train_file', type=str, default='dataset/fiqa/train.tsv'
    )
    data_group.add_argument('--neg_num', type=int, default=7)
    data_group.add_argument('--test_size', type=float, default=0.1)
    data_group.add_argument('--seed', type=int, default=1314)
    data_group.add_argument(
        '--output_subdir',
        type=str,
        default='/home/mnt/zhangkejun/work/dataset/reranker_ft',
    )

    ppl_group = parser.add_argument_group('Pipeline 参数')
    ppl_group.add_argument(
        '--mining_strategy',
        type=str,
        default='random',
        choices=['random', 'bm25', 'semantic', 'mixed'],
    )
    ppl_group.add_argument(
        '--output_format',
        type=str,
        default='flagreranker',
        choices=['flagreranker', 'cross_encoder', 'pairwise'],
    )
    ppl_group.add_argument('--train_group_size', type=int, default=8)

    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument(
        '--rerank_path', type=str, default='BAAI/bge-reranker-base'
    )
    model_group.add_argument(
        '--embed_path', type=str, default='BAAI/bge-small-zh-v1.5'
    )
    model_group.add_argument('--train_flag', action='store_true')

    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--per_device_batch_size', type=int, default=2)
    train_group.add_argument('--num_epochs', type=int, default=1)
    train_group.add_argument('--learning_rate', type=float, default=6e-5)
    train_group.add_argument('--ngpus', type=int, default=1)

    eval_group = parser.add_argument_group('评估参数')
    eval_group.add_argument('--topk', type=int, default=1)
    eval_group.add_argument('--eval_with_retriever', action='store_true')
    eval_group.add_argument('--retriever_topk', type=int, default=10)
    eval_group.add_argument(
        '--output_path',
        type=str,
        default=(
            '/home/mnt/zhangkejun/work/dataset/reranker_ft/'
            'rerank_eval_results.jsonl'
        ),
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    print('\n' + '=' * 80)
    print(' ' * 20 + 'Reranker 微调系统（Pipeline 版）')
    print('=' * 80 + '\n')

    need_build_dataset = args.build_dataset or args.one_click
    if need_build_dataset:
        print(
            '\n>>> 步骤 1: 构建数据集（加载 + 提取 query/pos → '
            'Pipeline 划分/难负样本/格式化/保存）'
        )
        if args.from_embed:
            embed_train = args.embed_train_path or build_data_path(
                'dataset', 'embed_train.json'
            )
            embed_eval = args.embed_eval_path or build_data_path(
                'dataset', 'embed_eval.json'
            )
            (
                train_data_path,
                eval_data_path,
            ) = build_rerank_dataset_from_embed_data(
                embed_train,
                embed_eval,
                args.neg_num,
                args.output_subdir,
            )
        else:
            if args.data_path:
                raw_items, corpus_texts = load_from_user_data(
                    args.data_path, args.query_key, args.pos_key
                )
            elif args.use_fiqa:
                raw_items, corpus_texts = load_from_fiqa(
                    args.queries_file, args.corpus_file, args.train_file
                )
            else:
                raw_items, corpus_texts = load_from_huggingface(
                    args.dataset_name,
                    split=args.dataset_split,
                    query_column=args.query_column,
                    pos_column=args.pos_column,
                    max_samples=args.max_samples,
                )
            (
                train_data_path,
                eval_data_path,
                kb_path,
            ) = build_dataset_with_pipelines(
                raw_items=raw_items,
                corpus_texts=corpus_texts,
                neg_num=args.neg_num,
                test_size=args.test_size,
                seed=args.seed,
                mining_strategy=args.mining_strategy,
                output_format=args.output_format,
                train_group_size=args.train_group_size,
                output_subdir=args.output_subdir,
            )
        print('数据集构建完成。')
        # 兼容旧行为：仅 --build_dataset 时构建后退出；
        # --one_click 时继续后续部署与评估
        if args.build_dataset and not args.one_click:
            return

    # 使用已有数据集进行微调/评估
    train_path = os.path.join(args.output_subdir, 'rerank_train.jsonl')
    eval_path = os.path.join(args.output_subdir, 'rerank_eval.jsonl')
    # train_path = '/home/mnt/zhangkejun/work/dataset/rerank_train.jsonl'
    # eval_path = '/home/mnt/zhangkejun/work/dataset/embed_eval.json'
    # kb_path = '/home/mnt/zhangkejun/work/KB/rerank_kb.txt'

    if not os.path.exists(train_path):
        print(
            '未找到 rerank_train.jsonl，请先执行 --build_dataset '
            '[--data_path PATH | --use_fiqa | --dataset_name NAME]'
        )
        return

    eval_data_path = eval_path if os.path.exists(eval_path) else train_path
    train_data_path = train_path

    print('\n>>> 步骤 2: 部署 Reranker')
    reranker_model = deploy_reranker_model(
        args.rerank_path,
        train_data_path,
        train_flag=args.train_flag,
        per_device_batch_size=args.per_device_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        ngpus=args.ngpus,
        train_group_size=args.train_group_size,
    )

    print('\n>>> 步骤 3: 评估')
    if args.eval_with_retriever:
        kb_path = os.path.join(args.output_subdir, 'rerank_kb.txt')
        if not os.path.exists(kb_path):
            print('未找到 KB 目录，跳过检索+重排评估')
        else:
            embed = lazyllm.TrainableModule(args.embed_path)
            docs = Document(kb_path, embed=embed, manager=False)
            docs.create_node_group(
                'split_sent',
                transform=lambda s: s.split('\n'),
                metadata=lambda _: {'source': 'kb'},
            )
            retriever = Retriever(
                doc=docs,
                group_name='split_sent',
                similarity='cosine',
                topk=args.retriever_topk,
            )
            reranker_wrapper = Reranker(
                name='ModuleReranker',
                model=reranker_model,
                topk=args.topk,
            )
            retriever.start()
            (
                _,
                recall_score,
                relevance_score,
            ) = evaluate_reranker_with_retriever(
                retriever,
                reranker_wrapper,
                eval_data_path,
            )
            save_results(
                {'recall': recall_score, 'relevance': relevance_score},
                args.train_flag,
                args.output_path,
            )
            print(
                f'Context Recall: {recall_score:.4f}, '
                f'Context Relevance: {relevance_score:.4f}'
            )
    else:
        mrr, hit_rate = evaluate_reranker_direct(
            reranker_model, eval_data_path, args.topk
        )
        save_results(
            {'mrr': mrr, f'hit@{args.topk}': hit_rate},
            args.train_flag,
            args.output_path,
        )
        print(f'MRR: {mrr:.4f}, Hit@{args.topk}: {hit_rate:.4f}')

    print(f'\n结果已追加到: {args.output_path}')


if __name__ == '__main__':
    main(parse_args())
