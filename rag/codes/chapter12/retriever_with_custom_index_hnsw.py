import time
import hnswlib

import numpy as np
from typing import Dict, List, Callable

import lazyllm
from lazyllm import LOG

from online_models import embedding_model

from lazyllm.tools.rag import IndexBase, StoreBase, DocNode
from lazyllm.tools.rag.utils import parallel_do_embedding
from lazyllm.common import override

DOC_PATH = "/mnt/lustre/share_data/dist/cmrc2018/data_kb"


class HNSWIndex(IndexBase):
    def __init__(
            self,
            embed: Dict[str, Callable],
            store: StoreBase,
            max_elements: int = 10000, # 定义索引最大容量
            ef_construction: int = 200,    # 平衡索引构建速度和搜索准确率，越大准确率越高但是构建速度越慢
            M: int = 16,    # 表示在建表期间每个向量的边数目量，M越高，内存占用越大，准确率越高，同时构建速度越慢
            dim: int = 1024, # 向量维度
            **kwargs
        ):
        self.embed = embed
        self.store = store
        # 由于hnswlib中的向量id不能为str，故建立label用于维护向量编号
        # 创建字典以维护节点id和hnsw中向量id的关系
        self.uid_to_label = {}
        self.label_to_uid = {}
        self.next_label = 0
        #初始化hnsw_index
        self._index_init(max_elements, ef_construction, M, dim)

    def _index_init(self, max_elements, ef_construction, M, dim):
        # hnswlib支持多种距离算法：l2、IP内积和cosine
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            allow_replace_deleted=True
        )
        self.index.set_ef(100) # 设置搜索时的最大近邻数量，较高值会导致更好的准确率，但搜索速度会变慢
        self.index.set_num_threads(8) # 设置在批量搜索和构建索引过程中使用的线程数

    @override
    def update(self, nodes: List['DocNode']):
        if not nodes or nodes[0]._group != 'block':
            return
        # 节点向量化，这里仅展示使用默认的embedding model
        parallel_do_embedding(self.embed, [], nodes=nodes, group_embed_keys={'block': ["__default__"]})
        vecs = []    # 向量列表
        labels = []    # 向量id列表
        for node in nodes:
            uid = str(node._uid)
            # 记录uid和label的关系，若为新的uid，则写入next_label
            if uid in self.uid_to_label:
                label = self.uid_to_label[uid]
            else:
                label = self.next_label
                self.uid_to_label[uid] = label
                self.next_label += 1
            # 取默认embedding结果
            vec = node.embedding['__default__'] 
            vecs.append(vec)
            labels.append(label)
            self.label_to_uid[label] = uid
        
        # 根据向量建立hnsw索引
        data = np.vstack(vecs)
        ids  = np.array(labels, dtype=int)
        self.index.add_items(data, ids)

    @override
    def remove(self, uids, group_name=None):
        """
        标记删除一批 uid 对应的向量，并清理映射
        """
        if group_name != 'block':
            return
        for uid in map(str, uids):
            if uid not in self.uid_to_label:
                continue
            label = self.uid_to_label.pop(uid)
            self.index.mark_deleted(label)
            self.label_to_uid.pop(label, None)

    @override
    def query(
        self,
        query: str,
        topk: int,
        embed_keys: List[str],
        **kwargs,
    ) -> List['DocNode']:
        # 生成查询向量
        parts = [self.embed[k](query) for k in embed_keys]
        qvec = np.concatenate(parts)
        # 调用hnsw knn_query方法进行向量检索
        labels, distances = self.index.knn_query(qvec, k=topk, num_threads=self.index.num_threads)
        results = []
        #取检索topk
        for lab, dist in zip(labels[0], distances[0]):
            uid = self.label_to_uid.get(lab)
            results.append(uid)
            if len(results) >= topk:
                break
        # 从store获取对应uid的节点
        return self.store.get_nodes(group_name='block', uids=results) if len(results) > 0 else []


docs1 = lazyllm.Document(dataset_path=DOC_PATH, embed=embedding_model)

docs1.create_node_group(name='block', transform=(lambda d: d.split('\n')))
docs1.register_index("hnsw", HNSWIndex, docs1.get_embed(), docs1.get_store())

retriever1 = lazyllm.Retriever(docs1, group_name="block", similarity="cosine", topk=3)
retriever2 = lazyllm.Retriever(docs1, group_name="block", index="hnsw", topk=3)
retriever1.start()
retriever2.start()

q = "证券监管？"

st = time.time()
res = retriever1(q)
et = time.time()
context_str = "\n---------\n".join([r.text for r in res])
LOG.info(f"query: {q}, default time: {et - st}, default res:\n {context_str}")

st = time.time()
res = retriever2(q)
et = time.time()
context_str = "\n---------\n".join([r.text for r in res])
LOG.info(f"query: {q}, HNSW time: {et - st}, HNSW res: \n{context_str}")
