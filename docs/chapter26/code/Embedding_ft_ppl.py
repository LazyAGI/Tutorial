"""
Embedding 微调完整 Pipeline - 基于 LazyLLM
结合 lazyllm.tools.data.operators.embedding_synthesis 与 embedding_pipelines，
支持两种输入方式并实现自动化流程。

功能：
1. 两种输入：用户输入数据（本地 JSON/JSONL）或 HuggingFace 数据集名（自动下载）
2. 数据准备与 Pipeline：划分、可选数据增强、难负样本挖掘、格式化
3. Embedding 微调（FlagEmbedding）与检索评估
"""

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import lazyllm
from lazyllm import LOG, Document, finetune, launchers
from lazyllm.tools.eval import NonLLMContextRecall, ContextRelevance

# 使用 LazyLLM 官方 embedding pipeline 与算子
from lazyllm.tools.data.pipelines.embedding_pipelines import (
    build_embedding_data_augmentation_pipeline,
    build_embedding_hard_neg_pipeline,
    build_embedding_data_formatter_pipeline,
)
from lazyllm.tools.data.operators.embedding_synthesis import (
    EmbeddingTrainTestSplitter)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_json(file_path: str, line_by_line: bool = True) -> list:
    """加载JSON数据文件

    Args:
        file_path: JSON文件路径
        line_by_line: 是否按行读取（JSONL格式）

    Returns:
        加载的数据列表
    """
    if line_by_line:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_data_path(dir_name: str, file_name: str) -> str:
    """构建数据存储路径并确保目录存在

    Args:
        dir_name: 目录名称
        file_name: 文件名

    Returns:
        完整文件路径
    """
    data_root = os.path.join(os.getcwd(), dir_name)
    os.makedirs(data_root, exist_ok=True)
    return os.path.join(data_root, file_name)


def save_json(score_recall, score_relevance, train_flag: bool,
              file_path: str) -> None:
    """保存评估分数到JSON文件"""
    with open(file_path, 'a', encoding='utf-8') as f:
        if train_flag:
            json.dump(
                {'ft_score_recall': score_recall,
                 'ft_score_relevance': score_relevance},
                f, ensure_ascii=False, indent=2)
        else:
            json.dump(
                {'origin_score_recall': score_recall,
                 'origin_score_relevance': score_relevance},
                f, ensure_ascii=False, indent=2)


def save_res(data, file_path: str) -> None:
    """保存详细结果到JSON文件"""
    with open(file_path, 'a') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# 数据构建：原始方式（保留作为 fallback）
# ---------------------------------------------------------------------------

def _extract_raw_pairs_from_fiqa(
    queries_file: str,
    corpus_file: str,
    train_file: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """从FiQA原始文件中提取 queries/corpus/train_pairs 三个映射。"""
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
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id, corpus_id = parts[0], parts[1]
                train_pairs.setdefault(query_id, []).append(corpus_id)

    return queries, corpus, train_pairs


# ---------------------------------------------------------------------------
# 数据加载与提取：仅负责「下载/加载 + 提取 query/pos」，不包含划分/负样本/保存
# ---------------------------------------------------------------------------

def load_from_fiqa(
    queries_file: str,
    corpus_file: str,
    train_file: str,
) -> Tuple[List[dict], List[str]]:
    """从 FiQA 文件加载并提取 (query, pos) 列表及语料列表。仅做提取，不做划分/负样本/保存。

    Returns:
        raw_items: [{'query': ..., 'pos': [...]}, ...]
        corpus_texts: 语料列表（用于 pipeline 难负样本池）
    """
    queries, corpus, train_pairs = _extract_raw_pairs_from_fiqa(
        queries_file, corpus_file, train_file
    )
    raw_items = []
    for query_id, pos_corpus_ids in train_pairs.items():
        if query_id not in queries:
            continue
        pos_texts = [corpus[cid] for cid in pos_corpus_ids if cid in corpus]
        if pos_texts:
            raw_items.append({
                'query': queries[query_id],
                'pos': pos_texts
            })
    corpus_texts = list(corpus.values())
    LOG.info(f'从 FiQA 加载: {len(raw_items)} 条 (query,pos)，'
             f'语料 {len(corpus_texts)} 篇')
    return raw_items, corpus_texts


# ---------------------------------------------------------------------------
# 两种输入：用户数据 与 HuggingFace 数据集名（仅加载 + 提取 query/pos）
# ---------------------------------------------------------------------------

def load_from_user_data(
    data_path: str,
    query_key: str = 'query',
    pos_key: str = 'pos',
    line_by_line: bool = True,
) -> Tuple[List[dict], List[str]]:
    """从本地文件加载用户数据，得到 (query, pos) 列表及语料列表。

    文件格式：每行一个 JSON 对象，或单个 JSON 数组；每个对象需包含 query 与 pos，
    若键名不同则通过 query_key / pos_key 指定。

    Returns:
        raw_items: [{'query': ..., 'pos': [...]}, ...]
        corpus_texts: 所有 pos 的并集（用于难负样本池）
    """
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

    LOG.info(f'从用户数据加载: {data_path} -> {len(raw_items)} 条 '
             f'(query,pos)，语料 {len(corpus_texts)} 篇')
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
    """从 HuggingFace 下载数据集并转为 (query, pos) 列表及语料列表。

    Args:
        dataset_name: 数据集名，如 'virattt/financial-qa-10K'
        split: 使用的 split，默认 'train'
        query_column: 问句列名（会映射为 query）
        pos_column: 正例文本列名（会映射为 pos，单条会转为 list）
        config: 数据集 config 名（可选）
        subset: 子集名（可选）
        max_samples: 最多使用样本数（可选，用于快速试跑）

    Returns:
        raw_items: [{'query': ..., 'pos': [...]}, ...]
        corpus_texts: 所有 pos 的并集
    """
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

    # 列名映射
    cols = ds.column_names
    q_col = query_column if query_column in cols else 'query'
    if q_col not in cols and 'question' in cols:
        q_col = 'question'
    p_col = pos_column if pos_column in cols else 'pos'
    if p_col not in cols and 'context' in cols:
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

    LOG.info(f'从 HuggingFace 加载: {dataset_name} -> {len(raw_items)} 条，'
             f'语料 {len(corpus_texts)} 篇')
    return raw_items, corpus_texts


# ---------------------------------------------------------------------------
# 数据构建：Pipeline 集成版
# ---------------------------------------------------------------------------

def build_dataset_with_pipelines(
    raw_items: List[dict],
    corpus_texts: Optional[List[str]] = None,
    instruction: str = 'Represent this sentence for searching '
                       'relevant passages: ',
    neg_num: int = 10,
    test_size: float = 0.1,
    seed: int = 1314,
    mining_strategy: str = 'random',
    augment_methods: Optional[List[str]] = None,
    num_augments: int = 2,
    augment_lang: str = 'en',
    output_format: str = 'flagembedding',
    embedding_serving=None,
    llm=None,
    mining_language: str = 'zh',
    output_path: str = 'dataset',
) -> Tuple[str, str, str]:
    """使用 embedding_pipelines 完成：划分 → [增强] → 难负样本挖掘 → 格式化 → 保存。

    调用方负责先通过 load_from_fiqa / load_from_user_data /
    load_from_huggingface 得到 raw_items（及可选 corpus_texts），
    本函数只做 pipeline 流程，不包含数据下载/提取。
    """
    if not raw_items:
        raise ValueError(
            'raw_items 为空，请先通过 load_from_fiqa / '
            'load_from_user_data / load_from_huggingface 加载数据')
    if corpus_texts is None:
        corpus_texts = list({
            p for item in raw_items
            for p in (item.get('pos') or [])
        })

    print('\n' + '=' * 60)
    print('使用 Pipeline：划分 → 难负样本 → 格式化 → 保存')
    print('=' * 60)
    msg = f'\n>>> 输入: {len(raw_items)} 条 (query,pos),语料 {len(corpus_texts)} 篇'
    print(msg)

    # ----- Step 1: 划分 train / test -----
    print('\n>>> Step 1: 划分 train / test (EmbeddingTrainTestSplitter)')

    splitter = EmbeddingTrainTestSplitter(
        test_size=test_size,
        seed=seed,
    )
    mixed_items = splitter(raw_items)

    train_items = [item for item in mixed_items
                   if item.get('split') == 'train']
    test_items = [item for item in mixed_items
                  if item.get('split') == 'test']

    print(f'训练集 {len(train_items)} 条，测试集 {len(test_items)} 条')

    # ----- Step 2: 数据增强（可选） -----
    if augment_methods:
        print(f'\n>>> Step 2: 数据增强 (方法: {augment_methods})')
        augmentation_pipeline = build_embedding_data_augmentation_pipeline(
            input_query_key='query',
            output_query_key='query',
            keep_original=True,
            llm=llm,
            augment_methods=augment_methods,
            num_augments=num_augments,
            lang=augment_lang,
        )
        train_items = augmentation_pipeline(train_items)
        print(f'增强后训练集: {len(train_items)} 条')
    else:
        print('\n>>> Step 2: 数据增强 (跳过)')

    # ----- Step 3: 难负样本挖掘 -----
    print(f'\n>>> Step 3: 难负样本挖掘 (策略: {mining_strategy})')

    test_corpus = list(set(p for item in test_items
                           for p in item.get('pos', [])))

    hard_neg_pipeline = build_embedding_hard_neg_pipeline(
        input_query_key='query',
        input_pos_key='pos',
        output_neg_key='neg',
        corpus=corpus_texts,
        mining_strategy=mining_strategy,
        num_negatives=neg_num,
        embedding_serving=embedding_serving,
        language=mining_language,
        seed=seed,
    )
    train_items_with_neg = hard_neg_pipeline(train_items)

    print(f'难负样本挖掘完成: {len(train_items_with_neg)} 条样本含负样本')

    # ----- Step 4: 格式化并保存训练数据（train 用 formatter，eval 用评估格式） -----
    print(f'\n>>> Step 4: 数据格式化 (格式: {output_format})')

    train_data_path = build_data_path(output_path, 'embed_train.json')
    formatter_pipeline = build_embedding_data_formatter_pipeline(
        input_query_key='query',
        input_pos_key='pos',
        input_neg_key='neg',
        output_format=output_format,
        instruction=instruction if output_format == 'flagembedding' else None,
        output_file=train_data_path,
    )
    formatted_train = formatter_pipeline(train_items_with_neg)
    print(f'训练数据: {len(formatted_train)} 条，已保存到 {train_data_path}')

    eval_data_path = build_data_path(output_path, 'embed_eval.json')
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        for item in test_items:
            f.write(json.dumps({
                'query': item.get('query', ''),
                'corpus': item.get('pos', []),
            }, ensure_ascii=False) + '\n')
    print(f'评估数据: {len(test_items)} 条，已保存到 {eval_data_path}')

    # ----- Step 5: 保存知识库（embedding_synthesis 无现成“写知识库 txt”算子，保留原逻辑） -----
    kb_data_path = build_data_path(output_path, 'embed_kb.txt')
    with open(kb_data_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_corpus))
    print(f'知识库: {len(test_corpus)} 篇 → {kb_data_path}')
    print('\n数据集构建完成！')

    return train_data_path, eval_data_path, os.path.dirname(kb_data_path)


# ---------------------------------------------------------------------------
# 模型部署
# ---------------------------------------------------------------------------

def deploy_embedding_model(
    kb_path: str,
    embed_path: str,
    train_data_path: str,
    train_flag: bool = True,
    per_device_batch_size: int = 16,
    num_epochs: int = 2,
    learning_rate: float = 1e-5,
    ngpus: int = 1,
) -> lazyllm.Retriever:
    """部署带微调功能的Embedding检索服务

    Args:
        kb_path: 知识库目录路径
        embed_path: Embedding模型路径/名称
        train_data_path: 训练数据路径
        train_flag: 是否执行微调
        per_device_batch_size: 每个设备的训练批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        ngpus: 使用的GPU数量

    Returns:
        配置好的Retriever实例
    """
    print(f"\n{'=' * 60}")
    print(f'配置Embedding模型: {embed_path}')
    print(f'知识库路径: {kb_path}')
    print(f'训练数据: {train_data_path}')
    print(f'微调模式: {"开启" if train_flag else "关闭"}')
    print(f"\n{'=' * 60}")

    if train_flag:
        embed = lazyllm.TrainableModule(embed_path) \
            .mode('finetune') \
            .trainset(train_data_path) \
            .finetune_method((
                finetune.flagembedding,
                {
                    'launcher': launchers.remote(nnode=1, nproc=1,
                                                 ngpus=ngpus),
                    'per_device_train_batch_size': per_device_batch_size,
                    'num_train_epochs': num_epochs,
                }
            ))
        print('\n开始微调Embedding模型...')
    else:
        embed = lazyllm.TrainableModule(embed_path)

    # 创建文档处理流程
    docs = Document(kb_path, embed=embed, manager=False)
    docs.create_node_group(
        name='split_sent',
        transform=lambda s: s.split('\n'),
        metadata=lambda _: {'source': 'knowledge_base'},
    )

    # 配置检索器
    retriever = lazyllm.Retriever(
        doc=docs,
        group_name='split_sent',
        similarity='cosine',
        topk=1,
    )

    if train_flag:
        retriever.update()
        print('微调完成！')
    else:
        print('\n启动Embedding模型服务...')
        retriever.start()

    return retriever


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def evaluate_retriever(
    retriever: lazyllm.Retriever,
    eval_data_path: str,
    use_instruction: bool = False,
    instruction: str = 'Represent this sentence for searching '
                       'relevant passages: ',
    low_recall_path: Optional[str] = None,
    low_relevance_path: Optional[str] = None,
    low_score_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """评估检索器性能

    Args:
        retriever: 检索器实例
        eval_data_path: 评估数据路径
        use_instruction: 是否在查询前添加指令
        instruction: 查询指令模板

    Returns:
        评估结果、召回率分数、相关性分数的元组
    """
    print('\n开始评估检索性能...')

    query_corpus = load_json(eval_data_path, line_by_line=True)

    def _extract_passage_from_retrieved(raw_text: str) -> List[str]:
        if not raw_text or not isinstance(raw_text, str):
            return [raw_text] if raw_text else []
        raw_text = raw_text.strip()
        if not (raw_text.startswith('{') and raw_text.endswith('}')):
            return [raw_text]
        try:
            obj = json.loads(raw_text)
            if 'corpus' in obj:
                c = obj['corpus']
                return c if isinstance(c, list) else [c]
            if 'pos' in obj:
                p = obj['pos']
                return p if isinstance(p, list) else [p]
        except (json.JSONDecodeError, TypeError):
            pass
        return [raw_text]

    results = []
    for item in tqdm(query_corpus, desc='处理查询'):
        query = item.get('query', '')
        # 兼容 formatter 输出（pos）与旧格式（corpus）
        reference = item.get('corpus', item.get('pos', []))
        inputs = f'{instruction}{query}' if use_instruction else query
        retrieved = retriever(inputs)
        raw_texts = [text.get_text() for text in retrieved]
        # 去掉 context_retrieved 中的 query，只保留段落：若为 JSON 则只取 corpus/pos
        context_retrieved = []
        for t in raw_texts:
            context_retrieved.extend(_extract_passage_from_retrieved(t))
        results.append({
            'question': query,
            'context_retrieved': context_retrieved,
            'context_reference': reference,
        })

    recall_eval = NonLLMContextRecall(binary=False)
    recall_score = recall_eval(results)
    relevance_eval = ContextRelevance(
        splitter='.',
        low_score_path=low_relevance_path,
        low_score_threshold=low_score_threshold,
    )
    relevance_score = relevance_eval(results)

    return results, recall_score, relevance_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Embedding模型微调和评估系统（集成 Pipeline 版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

输入方式一 - 用户数据文件:
   python Embedding_ft_ppl.py --build_dataset --data_path /path/to/your.jsonl

 输入方式二:
   python Embedding_ft_ppl.py --build_dataset
    python Embedding_ft_ppl.py --build_dataset \\
        --dataset_name virattt/financial-qa-10K

 本地 FiQA:
   python Embedding_ft_ppl.py --build_dataset --use_fiqa --mining_strategy bm25

 微调与评估:
   python Embedding_ft_ppl.py --train_flag --embed_path BAAI/bge-large-en-v1.5
        """,
    )

    # 数据相关参数（两种输入：--data_path 用户数据文件 | --dataset_name HuggingFace 数据集名）
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--build_dataset', action='store_true',
                            help='构建或强制重建数据集')
    data_group.add_argument('--data_path', type=str, default=None,
                            help='输入方式一：本地数据文件路径（JSON/JSONL），'
                                 "每行或整体为 {'query','pos'} 或可配置键名")
    data_group.add_argument('--query_key', type=str, default='query',
                            help='用户数据文件中「问句」字段名（与 --data_path 配合）')
    data_group.add_argument('--pos_key', type=str, default='pos',
                            help='用户数据文件中「正例」字段名（与 --data_path 配合）')
    data_group.add_argument('--use_fiqa', action='store_true',
                            help='使用本地 FiQA 数据集'
                                 '--queries_file/corpus_file/train_file 配合）')
    data_group.add_argument('--dataset_name', type=str,
                            default='virattt/financial-qa-10K',
                            help='输入方式二：HuggingFace 数据集名称，自动下载并跑 Pipeline')
    data_group.add_argument('--dataset_split', type=str, default='train',
                            help='HuggingFace 使用的 split')
    data_group.add_argument('--query_column', type=str, default='question',
                            help='HuggingFace 数据集中问句列名')
    data_group.add_argument('--pos_column', type=str, default='context',
                            help='HuggingFace 数据集中正例列名')
    data_group.add_argument('--max_samples', type=int, default=None,
                            help='HuggingFace 最多使用样本数（用于快速试跑）')
    data_group.add_argument('--queries_file', type=str,
                            default='dataset/fiqa/queries.jsonl',
                            help='FiQA queries文件路径')
    data_group.add_argument('--corpus_file', type=str,
                            default='dataset/fiqa/cpt/corpus.jsonl',
                            help='FiQA corpus文件路径')
    data_group.add_argument('--train_file', type=str,
                            default='dataset/fiqa/train.tsv',
                            help='FiQA训练文件路径')
    data_group.add_argument('--neg_num', type=int, default=20,
                            help='每个样本的负样本数量')
    data_group.add_argument('--test_size', type=float, default=0.1,
                            help='测试集比例')
    data_group.add_argument('--seed', type=int, default=1314,
                            help='随机种子')

    # Pipeline 参数（构建数据集时统一走 Pipeline：划分 / 难负样本 / 格式化 / 保存）
    ppl_group = parser.add_argument_group('Pipeline 参数')
    ppl_group.add_argument('--mining_strategy', type=str, default='random',
                           choices=['random', 'bm25', 'semantic'],
                           help='难负样本挖掘策略 (default: random)')
    ppl_group.add_argument('--augment_methods', type=str, nargs='*',
                           default=['query_rewrite'],
                           help='数据增强方法 (可选: query_rewrite, synonym_replace)')
    ppl_group.add_argument('--num_augments', type=int, default=1,
                           help='每种增强方法的增强数量')
    ppl_group.add_argument('--augment_lang', type=str, default='en',
                           help='增强语言')
    ppl_group.add_argument('--output_format', type=str,
                           default='flagembedding',
                           choices=['flagembedding', 'triplet',
                                    'sentence_transformers'],
                           help='训练数据输出格式')
    ppl_group.add_argument('--mining_language', type=str, default='en',
                           help='BM25 挖掘语言')
    ppl_group.add_argument(
        '--output_dir', type=str,
        default='/home/mnt/zhangkejun/work/dataset/embed_ft',
        help='输出目录')
    # 模型相关参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--embed_path', type=str,
                             default='BAAI/bge-large-zh-v1.5',
                             help='Embedding模型路径/名称')
    model_group.add_argument('--train_flag', action='store_true',
                             help='执行模型微调')

    # 训练相关参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--per_device_batch_size', type=int,
                             default=16,
                             help='每个设备的训练批次大小')
    train_group.add_argument('--num_epochs', type=int, default=2,
                             help='训练轮数')
    train_group.add_argument('--learning_rate', type=float, default=1e-6,
                             help='学习率')
    train_group.add_argument('--ngpus', type=int, default=1,
                             help='训练使用的GPU数量')

    # 评估相关参数
    eval_group = parser.add_argument_group('评估参数')
    eval_group.add_argument(
        '--instruction', type=str,
        default='Represent this sentence for searching '
                'relevant passages: ',
        help='查询指令模板')
    eval_group.add_argument('--use_instruction', action='store_true',
                            help='在查询前添加指令')
    eval_group.add_argument(
        '--output_path', type=str,
        default='/home/mnt/zhangkejun/work/dataset/embed_ft/'
                'embed_eval_results.jsonl',
        help='评估结果保存路径')

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """主执行流程

    全流程：数据构建 → [增强] → 难负样本挖掘 → 格式化 → 微调 → 评估
    """
    print('\n' + '=' * 80)
    print(' ' * 20 + 'Embedding模型微调系统（Pipeline 集成版）')
    print('=' * 80 + '\n')
    llm = lazyllm.TrainableModule(
        '/mnt/lustre/share_data/lazyllm/models/qwen2.5-14b-instruct')
#     llm = lazyllm.TrainableModule('Qwen2.5-VL-32B-Instruct').deploy_method(
#     lazyllm.deploy.vllm,
#     url='http://10.119.21.233:25120/v1'
# )
    embedding_serve = lazyllm.TrainableModule('BAAI/bge-large-en-v1.5')
    # ====== 1. 数据准备 ======
    if args.build_dataset:
        print('\n>>> 步骤 1: 构建数据集（加载 + 提取 query/pos → Pipeline 划分/难负样本/格式化/保存）')

        # 仅负责加载与提取 query/pos，其余由 pipeline 完成
        if args.data_path:
            raw_items, corpus_texts = load_from_user_data(
                args.data_path,
                query_key=getattr(args, 'query_key', 'query'),
                pos_key=getattr(args, 'pos_key', 'pos'),
            )
        elif args.use_fiqa:
            raw_items, corpus_texts = load_from_fiqa(
                args.queries_file,
                args.corpus_file,
                args.train_file,
            )
        else:
            raw_items, corpus_texts = load_from_huggingface(
                args.dataset_name,
                split=getattr(args, 'dataset_split', 'train'),
                query_column=getattr(args, 'query_column', 'question'),
                pos_column=getattr(args, 'pos_column', 'context'),
                max_samples=getattr(args, 'max_samples', None),
            )

        (train_data_path, eval_data_path,
         kb_path) = build_dataset_with_pipelines(
            raw_items=raw_items,
            corpus_texts=corpus_texts,
            instruction=args.instruction,
            neg_num=args.neg_num,
            test_size=args.test_size,
            seed=args.seed,
            mining_strategy=args.mining_strategy,
            augment_methods=args.augment_methods,
            num_augments=args.num_augments,
            augment_lang=args.augment_lang,
            output_format=args.output_format,
            mining_language=args.mining_language,
            llm=llm,
            embedding_serving=embedding_serve,
            output_path=args.output_dir,
        )
        print('数据集构建完成！')
        return

    train_path = os.path.join(args.output_dir, 'embed_train.json')
    eval_path = os.path.join(args.output_dir, 'embed_eval.json')
    kb_file = os.path.join(args.output_dir, 'embed_kb.txt')

    if not all(os.path.exists(f) for f in [train_path, eval_path, kb_file]):
        print('\n错误: 未找到处理好的数据集！')
        print('请先运行: python Embedding_ft_ppl.py --build_dataset '
              '[--data_path PATH | --use_fiqa | --dataset_name NAME]')
        return

    print('\n>>> 步骤 1: 加载现有数据集')
    print(f'训练数据: {train_path}')
    print(f'评估数据: {eval_path}')
    print(f'知识库:   {kb_file}')

    train_data_path = train_path
    eval_data_path = eval_path
    kb_path = os.path.dirname(kb_file)

    # ====== 2. 部署检索服务 ======
    print('\n>>> 步骤 2: 部署Embedding检索服务')
    retriever = deploy_embedding_model(
        kb_path=kb_path,
        embed_path=args.embed_path,
        train_data_path=train_data_path,
        train_flag=args.train_flag,
        per_device_batch_size=args.per_device_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        ngpus=args.ngpus,
    )

    # ====== 3. 评估性能 ======
    print('\n>>> 步骤 3: 评估检索性能')
    out_dir = (os.path.dirname(os.path.abspath(args.output_path))
               or args.output_dir)
    low_recall_path = os.path.join(out_dir, 'embed_low_recall.jsonl')
    low_relevance_path = os.path.join(out_dir, 'embed_low_relevance.jsonl')
    results, recall_score, relevance_score = evaluate_retriever(
        retriever=retriever,
        eval_data_path=eval_data_path,
        use_instruction=args.use_instruction or args.train_flag,
        instruction=args.instruction,
        low_recall_path=low_recall_path,
        low_relevance_path=low_relevance_path,
        low_score_threshold=1,
    )

    # ====== 4. 保存结果 ======
    print('\n>>> 步骤 4: 保存评估结果')
    save_res(results, 'embedding_result.json')
    save_json(recall_score, relevance_score, args.train_flag, args.output_path)
    print(f'结果已保存到: {args.output_path}')

    # ====== 5. 输出总结 ======
    print('\n' + '=' * 80)
    print(' ' * 30 + '评估结果')
    print('=' * 80)
    print(f'上下文召回率 (Context Recall):   {recall_score:.4f}')
    print(f'上下文相关性 (Context Relevance): {relevance_score:.4f}')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main(parse_args())
