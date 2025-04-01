from typing import List
import lazyllm
from lazyllm import pipeline, parallel, bind, Retriever, Reranker
from lazyllm.tools.rag import Document
from lazyllm.tools.rag import IndexBase, StoreBase, DocNode
from lazyllm.common import override

# 使用 Milvus 存储后端
chroma_store_conf = {
  'type': 'chroma',
  'kwargs': {
    'dir': 'diyIdxDB',
   },
  'indices': {
    'smart_embedding_index': {
      'backend': 'milvus',
      'kwargs': {
        'uri': "dbs/test.db",
        'index_kwargs': {
          'index_type': 'HNSW',
          'metric_type': 'COSINE',
        }
      },
    },
  },
}


# 自定义关键词索引后端
class KeywordIndex(IndexBase):
    def __init__(self, store: StoreBase, **kwargs):
        self.store = store
        self.kv_index = {}
        self.update_index('keyword')

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

    @override
    def update(self, nodes: List[DocNode]) -> None:
        pass

    @override
    def remove(self, group_name: str, uids: List[str]) -> None:
        pass


prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n '

# llm = lazyllm.TrainableModule('Qwen2-72B-Instruct-AWQ').deploy_method(lazyllm.deploy.Vllm).start()
# llm = lazyllm.TrainableModule('Qwen2-72B-Instruct-AWQ', stream=True).deploy_method(lazyllm.deploy.Vllm).start()
llm = lazyllm.OnlineChatModule(base_url="http://127.0.0.1:36858/", stream=True)

keyword_llm = lazyllm.LLMParser(llm, language="zh", task_type="keywords")  # 关键词提取LLM

# 从向量数据库加载文档
document = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb",
                    embed=lazyllm.TrainableModule('bge-large-zh-v1.5'),
                    manager=True,
                    store_conf=chroma_store_conf)
# 创建新的节点组并注册自定义索引
document.create_node_group(name="block", transform=lambda node: node.split('\n'))
document.create_node_group(name="keyword", transform=keyword_llm, parent="CoarseChunk")
document.register_index("keyword_index", KeywordIndex, document.get_store())

# 定义 RAG 主流程
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        ppl.prl.retriever1 = Retriever(document, "block", index="smart_embedding_index", topk=3)
        ppl.prl.retriever2 = Retriever(document, "keyword", index="keyword_index", topk=3)
    ppl.reranker = Reranker(name="ModuleReranker", model='bge-reranker-large', topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = llm.share(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))

# 启动 web 服务
lazyllm.WebModule(ppl, port=23456, stream=True).start().wait()
