"""
Reranker模型微调脚本 - 基于LazyLLM框架
参考：LazyLLM FlagEmbedding微调组件

功能：
1. 数据准备和处理（支持从原始数据构建训练集）
2. Reranker模型微调（基于FlagEmbedding）
3. 效果评测（使用MRR和NDCG指标）
4. 在RAG中使用微调后的模型
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from typing import Tuple, List, Dict, Any

import lazyllm
from lazyllm import (
    Document, Retriever, Reranker, finetune, launchers, TrainableModule)
from lazyllm.tools.eval import (
    NonLLMContextRecall, ContextRelevance)


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


def build_rerank_dataset_from_fiqa(
    queries_file: str,
    corpus_file: str,
    train_file: str,
    neg_num: int = 7,
    test_size: float = 0.1,
    seed: int = 1314
) -> Tuple[str, str, str]:
    """从FiQA数据集文件构建Reranker训练和评估数据

    Args:
        queries_file: queries.jsonl文件路径
        corpus_file: corpus.jsonl文件路径
        train_file: train.tsv文件路径
        neg_num: 每个样本的负样本数量（FlagEmbedding默认train_group_size=8，即1正+7负）
        test_size: 测试集比例
        seed: 随机种子

    Returns:
        训练数据路径、评估数据路径、知识库路径的元组
    """
    print('正在从FiQA数据集构建Reranker训练数据...')

    # 加载queries
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            queries[item['_id']] = item['text']

    # 加载corpus
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus[item['_id']] = item['text']

    # 加载训练对
    train_pairs = {}
    with open(train_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id, corpus_id = parts[0], parts[1]
                if query_id not in train_pairs:
                    train_pairs[query_id] = []
                train_pairs[query_id].append(corpus_id)

    # 构建数据集（Reranker格式：不需要instruction）
    dataset = []
    np.random.seed(seed)
    corpus_ids = list(corpus.keys())

    for query_id, pos_corpus_ids in train_pairs.items():
        if query_id not in queries:
            continue

        # 获取正样本
        pos_texts = [corpus[cid] for cid in pos_corpus_ids if cid in corpus]
        if not pos_texts:
            continue

        # 生成负样本
        neg_texts = []
        attempts = 0
        while len(neg_texts) < neg_num and attempts < neg_num * 10:
            idx_max = len(corpus_ids)
            r = np.random.randint(0, idx_max)
            neg_id = corpus_ids[r]
            if (neg_id not in pos_corpus_ids and
                    corpus[neg_id] not in neg_texts):
                neg_texts.append(corpus[neg_id])
            attempts += 1

        # Reranker数据格式：只需要query, pos, neg
        dataset.append({
            'query': queries[query_id],
            'pos': pos_texts[:1],  # 使用第一个正样本
            'neg': neg_texts
        })

    # 划分训练集和测试集
    np.random.seed(seed)
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - test_size))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    # 保存训练数据（JSONL格式，FlagEmbedding Reranker要求）
    train_data_path = build_data_path('dataset', 'rerank_train.jsonl')
    with open(train_data_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存评估数据
    eval_data = []
    for item in test_data:
        eval_data.append({
            'query': item['query'],
            'corpus': item['pos'],
            'neg': item['neg']  # 保留负样本用于评估
        })
    eval_data_path = build_data_path('dataset', 'rerank_eval.jsonl')
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 创建知识库（用于检索+重排流程测试）
    kb_data_path = build_data_path('KB', 'rerank_kb.txt')
    kb_texts = []
    for item in test_data:
        kb_texts.extend(item['pos'])

    with open(kb_data_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kb_texts))

    print(f'数据集构建完成：训练样本 {len(train_data)}，测试样本 {len(test_data)}')
    return train_data_path, eval_data_path, os.path.dirname(kb_data_path)


def build_rerank_dataset_from_huggingface(
    dataset_name: str = 'virattt/financial-qa-10K',
    neg_num: int = 7,
    test_size: float = 0.1,
    seed: int = 1314
) -> Tuple[str, str, str]:
    """从HuggingFace数据集构建Reranker训练和评估数据

    Args:
        dataset_name: HuggingFace数据集名称
        neg_num: 每个样本的负样本数量
        test_size: 测试集比例
        seed: 随机种子

    Returns:
        训练数据路径、评估数据路径、知识库路径的元组
    """
    print(f'正在从HuggingFace加载数据集: {dataset_name}')

    ds = load_dataset(dataset_name, split='train')
    ds = ds.select_columns(column_names=['question', 'context'])
    ds = ds.rename_columns({'question': 'query', 'context': 'pos'})

    def str_to_lst(data):
        data['pos'] = [data['pos']]
        return data
    ds = ds.map(str_to_lst)
    split = ds.train_test_split(test_size=test_size, shuffle=True, seed=seed)

    test_corpus_list = [item[0] for item in split['test']['pos']]

    # 为训练集生成负样本（从测试集corpus中采样）
    np.random.seed(seed)
    train_data = []
    for item in tqdm(split['train'], desc='生成负样本'):
        neg_indices = np.random.randint(0, len(test_corpus_list), size=neg_num)
        neg = [test_corpus_list[idx] for idx in neg_indices]
        pos_text = item['pos'][0]
        neg = [n for n in neg if n != pos_text][:neg_num]

        train_data.append({
            'query': item['query'],
            'pos': item['pos'],
            'neg': neg
        })

    # 保存训练数据（JSONL格式）
    train_data_path = build_data_path('dataset', 'rerank_train.jsonl')
    with open(train_data_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 处理并保存评估数据
    eval_data_path = build_data_path('dataset', 'rerank_eval.jsonl')
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        for _idx, item in enumerate(split['test']):
            neg_indices = np.random.randint(
                0, len(test_corpus_list), size=neg_num)
            neg = [test_corpus_list[i]
                   for i in neg_indices
                   if test_corpus_list[i] != item['pos'][0]][:neg_num]
            f.write(json.dumps({
                'query': item['query'],
                'corpus': item['pos'],
                'neg': neg
            }, ensure_ascii=False) + '\n')

    kb_data_path = build_data_path('KB', 'rerank_kb.txt')
    with open(kb_data_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_corpus_list))

    test_size = len(split['test'])
    print(
        '数据集构建完成：训练样本 {}，测试样本 {}，知识库语料 {}'.format(
            len(train_data), test_size, len(test_corpus_list))
    )
    return train_data_path, eval_data_path, os.path.dirname(kb_data_path)


def deploy_reranker_model(
    rerank_path: str,
    train_data_path: str,
    train_flag: bool = True,
    per_device_batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 6e-5,
    ngpus: int = 1,
    train_group_size: int = 8
) -> TrainableModule:
    """部署带微调功能的Reranker模型

    Args:
        rerank_path: Reranker模型路径/名称
        train_data_path: 训练数据路径
        train_flag: 是否执行微调
        per_device_batch_size: 每个设备的训练批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        ngpus: 使用的GPU数量
        train_group_size: 训练组大小（1个正样本 + n-1个负样本）

    Returns:
        配置好的TrainableModule实例
    """
    print('\n' + '='*60)
    print('配置Reranker模型: ' + str(rerank_path))
    print('训练数据: ' + str(train_data_path))
    print('微调模式: ' + ('开启' if train_flag else '关闭'))
    print('='*60 + '\n')

    if train_flag:
        # 配置Reranker模型微调
        reranker_model = lazyllm.TrainableModule(rerank_path) \
            .mode('finetune') \
            .trainset(train_data_path) \
            .finetune_method((
                finetune.auto,
                {
                    'launcher': launchers.empty(ngpus=ngpus),
                    'per_device_train_batch_size': per_device_batch_size,
                    'num_train_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'train_group_size': train_group_size,
                    'query_max_len': 256,
                    'passage_max_len': 256,
                }
            ))
        print('\n开始微调Reranker模型...')
        reranker_model.update()
        print('微调完成！')
    else:
        reranker_model = lazyllm.TrainableModule(rerank_path)
        print('\n启动Reranker模型服务...')
        reranker_model.start()

    return reranker_model


def evaluate_reranker_direct(
    reranker_model: TrainableModule,
    eval_data_path: str,
    topk: int = 1
) -> Tuple[float, float]:
    """直接评估Reranker模型的排序性能

    Args:
        reranker_model: Reranker模型实例
        eval_data_path: 评估数据路径
        topk: 取top-k结果计算指标

    Returns:
        MRR分数、命中率
    """
    print('\n开始评估Reranker排序性能...')

    # 加载评估数据
    eval_data = load_json(eval_data_path, line_by_line=True)

    total_mrr = 0.0
    total_hits = 0

    for item in tqdm(eval_data, desc='评估进度'):
        query = item['query']
        pos_docs = item['corpus']  # 正样本文档
        neg_docs = item.get('neg', [])  # 负样本文档

        # 合并所有候选文档
        all_docs = pos_docs + neg_docs
        if len(all_docs) == 0:
            continue

        # 使用reranker进行排序
        # reranker返回格式: [(index, score), ...]
        results = reranker_model(
            query, documents=all_docs, top_n=len(all_docs))

        # 计算MRR (Mean Reciprocal Rank)
        for rank, (idx, _score) in enumerate(results, 1):
            if idx < len(pos_docs):  # 找到正样本
                total_mrr += 1.0 / rank
                if rank <= topk:
                    total_hits += 1
                break

    mrr = total_mrr / len(eval_data) if eval_data else 0.0
    hit_rate = total_hits / len(eval_data) if eval_data else 0.0

    return mrr, hit_rate


def evaluate_reranker_with_retriever(
    retriever: Retriever,
    reranker: Reranker,
    eval_data_path: str
) -> Tuple[List[Dict[str, Any]], float, float]:
    """评估检索+重排流程的性能

    Args:
        retriever: 检索器实例
        reranker: 重排器实例
        eval_data_path: 评估数据路径

    Returns:
        评估结果、召回率分数、相关性分数的元组
    """
    print('\n开始评估检索+重排流程...')

    # 加载评估数据
    query_corpus = load_json(eval_data_path, line_by_line=True)

    # 执行检索和重排
    results = []
    for item in tqdm(query_corpus, desc='处理查询'):
        query = item['query']

        # 先检索
        retrieved = retriever(query=query)

        # 再重排
        reranked = reranker(retrieved, query=query)

        results.append({
            'question': query,
            'context_retrieved': [text.get_text() for text in reranked],
            'context_reference': item['corpus']
        })

    # 计算评估指标
    recall_eval = NonLLMContextRecall(binary=False)
    relevance_eval = ContextRelevance()
    recall_score = recall_eval(results)
    relevance_score = relevance_eval(results)

    return results, recall_score, relevance_score


def save_results(
    score_dict: Dict[str, Any],
    train_flag: bool,
    file_path: str
) -> None:
    """保存评估结果到JSON文件

    Args:
        score_dict: 评估分数字典
        train_flag: 是否为微调后的模型
        file_path: 目标文件路径
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        prefix = 'ft_' if train_flag else 'origin_'
        result = {prefix + str(k): v for k, v in score_dict.items()}
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Reranker模型微调和评估系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:


 从FiQA数据集构建训练数据:
   python reranker_finetune.py --build_dataset --use_fiqa

 一键启动（构建 + 微调/部署 + 评估）:
   python reranker_finetune.py --one_click --train_flag --use_fiqa

使用基线模型评估:
   python reranker_finetune.py --rerank_path BAAI/bge-reranker-base

  微调Reranker模型:
   python reranker_finetune.py --train_flag
   python reranker_finetune.py --num_epochs 1 --ngpus 1

  使用HuggingFace数据集:
   python reranker_finetune.py --build_dataset
   python reranker_finetune.py --dataset_name virattt/financial-qa-10K
        """
    )

    # 数据相关参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--one_click', action='store_true',
                            help='一键执行全流程：构建数据集 + 部署/微调 + 评估')
    data_group.add_argument('--build_dataset', action='store_true',
                            help='构建或强制重建数据集')
    data_group.add_argument('--use_fiqa', action='store_true',
                            help='使用本地FiQA数据集')
    data_group.add_argument(
        '--dataset_name', type=str, default='virattt/financial-qa-10K',
        help='HuggingFace数据集')
    data_group.add_argument('--queries_file', type=str,
                            default='dataset/fiqa/queries.jsonl',
                            help='FiQA queries文件路径')
    data_group.add_argument('--corpus_file', type=str,
                            default='dataset/fiqa/cpt/corpus.jsonl',
                            help='FiQA corpus文件路径')
    data_group.add_argument('--train_file', type=str,
                            default='dataset/fiqa/train.tsv',
                            help='FiQA训练文件路径')
    data_group.add_argument('--neg_num', type=int, default=7,
                            help='每个样本的负样本数量（默认7，与train_group_size=8配合）')
    data_group.add_argument('--test_size', type=float, default=0.1,
                            help='测试集比例')
    data_group.add_argument('--seed', type=int, default=1314,
                            help='随机种子')

    # 模型相关参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument(
        '--rerank_path', type=str, default='BAAI/bge-reranker-base',
        help='Reranker模型路径')
    model_group.add_argument(
        '--embed_path', type=str, default='BAAI/bge-small-zh-v1.5',
        help='Embedding模型路径')
    model_group.add_argument('--train_flag', action='store_true',
                             help='执行模型微调')

    # 训练相关参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--per_device_batch_size', type=int, default=2,
                             help='每个设备的训练批次大小')
    train_group.add_argument('--num_epochs', type=int, default=1,
                             help='训练轮数')
    train_group.add_argument('--learning_rate', type=float, default=6e-5,
                             help='学习率（reranker默认6e-5）')
    train_group.add_argument('--ngpus', type=int, default=1,
                             help='训练使用的GPU数量')
    train_group.add_argument('--train_group_size', type=int, default=8,
                             help='训练组大小（1个正样本 + n-1个负样本）')

    # 评估相关参数
    eval_group = parser.add_argument_group('评估参数')
    eval_group.add_argument('--topk', type=int, default=1,
                            help='评估时取top-k结果')
    eval_group.add_argument('--eval_with_retriever', action='store_true',
                            help='使用检索+重排流程评估')
    eval_group.add_argument(
        '--retriever_topk', type=int, default=10,
        help='检索器返回的候选数量')
    eval_group.add_argument(
        '--output_path', type=str, default='rerank_eval_results.jsonl',
        help='评估结果保存路径')

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """主执行流程"""
    print('\n' + '=' * 80)
    print(' ' * 25 + 'Reranker模型微调系统')
    print('=' * 80 + '\n')

    # 1. 数据准备
    need_build_dataset = args.build_dataset or args.one_click
    if need_build_dataset:
        print('\n>>> 步骤 1: 构建数据集')

        if args.use_fiqa:
            trd, evd, kb = build_rerank_dataset_from_fiqa(
                queries_file=args.queries_file,
                corpus_file=args.corpus_file,
                train_file=args.train_file,
                neg_num=args.neg_num,
                test_size=args.test_size,
                seed=args.seed,
            )
            train_data_path, eval_data_path, kb_path = trd, evd, kb
        else:
            trd, evd, kb = build_rerank_dataset_from_huggingface(
                dataset_name=args.dataset_name,
                neg_num=args.neg_num,
                test_size=args.test_size,
                seed=args.seed,
            )
            train_data_path, eval_data_path, kb_path = trd, evd, kb
        print('数据集构建完成！')
        # 兼容旧行为：仅 --build_dataset 时构建后退出；
        # --one_click 时继续后续部署与评估
        if args.build_dataset and not args.one_click:
            return

    # 检查数据集是否存在
    work_path = os.getcwd()
    train_path = os.path.join(work_path, 'dataset', 'rerank_train.jsonl')
    eval_path = os.path.join(work_path, 'dataset', 'rerank_eval.jsonl')

    if not all(os.path.exists(f) for f in [train_path, eval_path]):
        print('\n错误: 未找到处理好的Reranker数据集！')
        print('请先运行以下命令之一构建数据集:')
        print('python reranker_finetune.py --build_dataset --from_embed')
        print('python reranker_finetune.py --build_dataset --use_fiqa')
        print('python reranker_finetune.py --build_dataset')
        return

    print('\n>>> 步骤 1: 加载现有数据集')
    print('训练数据: ' + str(train_path))
    print('评估数据: ' + str(eval_path))

    train_data_path = train_path
    eval_data_path = eval_path

    # 2. 部署Reranker模型
    print('\n>>> 步骤 2: 部署Reranker模型')
    reranker_model = deploy_reranker_model(
        rerank_path=args.rerank_path,
        train_data_path=train_data_path,
        train_flag=args.train_flag,
        per_device_batch_size=args.per_device_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        ngpus=args.ngpus,
        train_group_size=args.train_group_size
    )

    # 3. 评估性能
    print('\n>>> 步骤 3: 评估Reranker性能')

    if args.eval_with_retriever:
        # 使用检索+重排流程评估
        kb_path = os.path.join(work_path, 'KB')
        if not os.path.exists(kb_path):
            print('警告: 未找到知识库目录，跳过检索+重排评估')
        else:
            print('初始化Embedding模型用于检索...')
            embed = lazyllm.TrainableModule(args.embed_path)
            docs = Document(kb_path, embed=embed, manager=False)
            docs.create_node_group(
                name='split_sent',
                transform=lambda s: s.split('\n'),
                metadata=lambda _: {'source': 'knowledge_base'}
            )
            retriever = Retriever(
                doc=docs,
                group_name='split_sent',
                similarity='cosine',
                topk=args.retriever_topk
            )
            reranker = Reranker(
                name='ModuleReranker',
                model=reranker_model,
                topk=args.topk
            )
            retriever.start()

            results, recall_score, relevance_score = (
                evaluate_reranker_with_retriever(
                    retriever=retriever,
                    reranker=reranker,
                    eval_data_path=eval_data_path,
                )
            )

            # 保存结果
            save_results(
                {'recall': recall_score, 'relevance': relevance_score},
                args.train_flag,
                args.output_path
            )

            print('\n' + '=' * 80)
            print(' ' * 30 + '评估结果')
            print('=' * 80)
            print('上下文召回率 (Context Recall): ' + f'{recall_score:.4f}')
            print('上下文相关性 (Context Relevance): ' + f'{relevance_score:.4f}')
    else:
        # 直接评估Reranker排序性能
        mrr, hit_rate = evaluate_reranker_direct(
            reranker_model=reranker_model,
            eval_data_path=eval_data_path,
            topk=args.topk
        )

        # 保存结果
        save_results(
            {'mrr': mrr, f'hit@{args.topk}': hit_rate},
            args.train_flag,
            args.output_path
        )

        print('\n' + '=' * 80)
        print(' ' * 30 + '评估结果')
        print('=' * 80)
        print('MRR (Mean Reciprocal Rank): ' + f'{mrr:.4f}')
        print('Hit@' + str(args.topk) + ': ' + f'{hit_rate:.4f}')

    print('=' * 80 + '\n')
    print('结果已保存到: ' + str(args.output_path))


if __name__ == '__main__':
    main(parse_args())
