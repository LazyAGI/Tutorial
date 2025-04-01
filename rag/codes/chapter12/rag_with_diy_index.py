from typing import List
import lazyllm
from lazyllm import bind
from lazyllm.common import override
from lazyllm.tools.rag import IndexBase, StoreBase, DocNode, Document

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


prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n '

print("模型加载中...")
bge_embed = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
bge_rerank = lazyllm.TrainableModule("bge-reranker-large").start()
llm = lazyllm.TrainableModule('internlm2-chat-20b', stream=True).start()
print("模型加载完毕！")

document = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb",
                    embed=bge_embed,
                    store_conf=chroma_store_conf)
# 创建关键词节点组，提取每个文段的关键词
keyword_llm = lazyllm.LLMParser(llm, language="zh", task_type="keywords")  # 关键词提取LLM
document.create_node_group(name="block", transform=lambda node: node.split('\n'))
document.create_node_group(name="keyword", transform=keyword_llm, parent="CoarseChunk")
# 注册自定义 Index
document.register_index("keyword_index", KeywordIndex, document.get_store())

with lazyllm.pipeline() as ppl:
    with lazyllm.parallel().sum as ppl.prl:
        ppl.prl.retriever1 = lazyllm.Retriever(document, group_name="keyword", index="keyword_index", topk=3)
        ppl.prl.retriever2 = lazyllm.Retriever(document, group_name="block", similarity="cosine", topk=3)
    ppl.reranker = lazyllm.Reranker(name='ModuleReranker', model=bge_rerank, topk=3) | bind(query=ppl.input)
    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="".join([node.get_content() for node in nodes]),
            query=query,
        )
    ) | bind(query=ppl.input)
    ppl.llm = llm.prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23456, stream=True).start().wait()
