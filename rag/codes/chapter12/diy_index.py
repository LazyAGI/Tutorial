import time
from lazyllm.tools.rag import IndexBase, StoreBase, DocNode
from lazyllm.common import override
from typing import List, Union
import lazyllm
from lazyllm.tools.rag import Document

chroma_store_conf = {
  'type': 'chroma',
  'kwargs': {
    'dir': 'diyIdxDB',
   }
}


class KeywordIndex(IndexBase):
    def __init__(self, store: StoreBase, **kwargs):
        self.store = store
        self.kv_index = {}
        self.update_index('keyword')

    @override
    def update(self, nodes: List[DocNode]) -> None:
        pass

    @override
    def remove(self, group_name: str, uids: List[str]) -> None:
        pass

    def update_index(self, group_name) -> None:
        """根据节点组名称创建关键词索引字典"""
        """形如：{“猴面包树”：[<Node 123>, <Node 214>], "亚硝酸盐"：[<Node 124345>]}"""
        nodes = self.store.get_nodes(group_name)
        for node in nodes:
            if node._group != 'keyword':
                continue
            if self.kv_index.get(node.text):
                self.kv_index[node.text].append(node.parent)
            else:
                self.kv_index[node.text] = [node.parent]

    @override
    def query(self, query: str, **kwargs) -> List[DocNode]:
        if isinstance(query, str):
            query = [query]
        results = []
        for keyword in query:
            if self.kv_index.get(keyword):
                results.extend(self.kv_index.get(keyword))
        results = set(results)
        return results


@lazyllm.tools.rag.register_similarity(mode='text', batch=True)
def keyword_filter(query: Union[str, List[str]], nodes: List[DocNode], **kwargs) -> float:
    scores = [0] * len(nodes)
    for idx, node in enumerate(nodes):
        node_text = node.text
        if query in node_text:
            scores[idx] = 1.0
            break

    return [(node, score) for node, score in zip(nodes, scores)]


print("正在加载模型...")
bge_embed = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
llm = lazyllm.TrainableModule('internlm2-chat-20b').deploy_method(lazyllm.deploy.Vllm).start()
print("模型加载完毕...")

document = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb",
                    embed=bge_embed,
                    store_conf=chroma_store_conf)
# 创建关键词节点组，提取每个文段的关键词
keyword_llm = lazyllm.LLMParser(llm, language="zh", task_type="keywords")  # 关键词提取LLM
document.create_node_group(name="keyword", transform=keyword_llm, parent="CoarseChunk")
# 注册自定义 Index
document.register_index("keyword_index", KeywordIndex, document.get_store())

# 在 CoarseChunk 节点组直接执行关键词筛选
retriever1 = lazyllm.Retriever(doc=document,
                 group_name="CoarseChunk",
                 similarity="keyword_filter",
                 similarity_cut_off=0.8,
                 topk=3)

# 在 keyword 节点组动过自定义关键词索引进行检索
retriever2 = lazyllm.Retriever(doc=document,
                               group_name="keyword",
                               index='keyword_index',
                               topk=3)

query = "尤金袋鼠"

# 初始化
retriever1(query)
retriever2(query)

# 记录检索时间
start1 = time.time()
nodes1 = retriever1(query)
end1 = time.time()

start2 = time.time()
nodes2 = retriever2(query)
end2 = time.time()

print("\n====== 默认索引召回结果 ======\n")
print('\n\n'.join([node.get_content() for node in nodes1]))
print("\n======关键词索引召回结果======\n")
print('\n\n'.join([node.get_content() for node in nodes2]))

print(f"\n默认索引  ：{(end1-start1):.8f}")
print(f"关键词索引：{(end2-start2):.8f}")
