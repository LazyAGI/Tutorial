import lazyllm
from lazyllm import bind

# 使用 chroma 存储后端
chroma_store_conf = {
  'type': 'chroma',
  'kwargs': {
    'dir': 'qa_pair_chromadb',
   },
  'indices': {
    # 使用 milvus 索引后端
    'smart_embedding_index': {
      'backend': 'milvus',
      'kwargs': {
        'uri': "qa_pair/test.db",
        'index_kwargs': {
          'index_type': 'HNSW',
          'metric_type': 'COSINE',
        }
      },
    },
  },
}

rewriter_prompt = "你是一个查询重写助手，负责给用户查询进行模板切换。\
          注意，你不需要进行回答，只需要对问题进行重写，使更容易进行检索\
          下面是一个简单的例子：\
          输入：RAG是啥？\
          输出：RAG的定义是什么？"
rag_prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
                根据以下资料回答问题：\
                {context_str} \n '


# 定义嵌入模型和重排序模型
# online_embedding = lazyllm.OnlineEmbeddingModule()
embedding_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 如果您要使用在线重排模型
# 目前LazyLLM仅支持 qwen和glm 在线重排模型，请指定相应的 API key。
# online_rerank = lazyllm.OnlineEmbeddingModule(type="rerank")
# 本地重排序模型
offline_rerank = lazyllm.TrainableModule('bge-reranker-large').start()

llm = lazyllm.OnlineChatModule(model="qwen2", base_url="http://10.119.24.121:25120/v1/", stream=True)  # 访问 Vllm 启动的模型服务

qa_parser = lazyllm.LLMParser(llm, language="zh", task_type="qa")

docs = lazyllm.Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embedding_model, store_conf=chroma_store_conf)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))
docs.create_node_group(name='qapair', transform=qa_parser)


def retrieve_and_rerank():
    with lazyllm.pipeline() as ppl:
        # lazyllm.parallel 并行执行两个检索操作
        with lazyllm.parallel().sum as ppl.prl:
            # CoarseChunk是LazyLLM默认提供的大小为1024的分块名
            ppl.prl.retriever1 = lazyllm.Retriever(doc=docs, group_name="CoarseChunk", index="smart_embedding_index", topk=3)
            ppl.prl.retriever2 = lazyllm.Retriever(doc=docs, group_name="block", similarity="bm25_chinese", topk=3)
        ppl.reranker = lazyllm.Reranker("ModuleReranker", model=offline_rerank, topk=3) | bind(query=ppl.input)
    return ppl


with lazyllm.pipeline() as ppl:
    # llm.share 表示复用一个大模型，如果这里设置为promptrag_prompt则会覆盖rewrite_prompt
    ppl.query_rewriter = llm.share(lazyllm.ChatPrompter(instruction=rewriter_prompt))
    with lazyllm.parallel().sum as ppl.prl:
        ppl.prl.retrieve_rerank = retrieve_and_rerank()
        # qa pair 节点组不与其他节点组一同参与重排序
        ppl.prl.qa_retrieve = lazyllm.Retriever(doc=docs, group_name="qapair", index="smart_embedding_index", topk=3)
    ppl.formatter = (
          lambda nodes, query: dict(
              context_str='\n'.join([node.get_content() for node in nodes]),
              query=query)
        ) | bind(query=ppl.input)
    ppl.llm = llm.share(lazyllm.ChatPrompter(instruction=rag_prompt, extro_keys=['context_str']))

lazyllm.WebModule(ppl, port=23491, stream=True).start().wait()
