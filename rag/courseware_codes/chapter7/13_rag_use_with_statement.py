import lazyllm
from lazyllm import bind

# 定义嵌入模型和重排序模型
# online_embedding = lazyllm.OnlineEmbeddingModule()
embedding_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 如果您要使用在线重排模型
# 目前LazyLLM仅支持 qwen和glm 在线重排模型，请指定相应的 API key。
# online_rerank = lazyllm.OnlineEmbeddingModule(type="rerank")
# 本地重排序模型
offline_rerank = lazyllm.TrainableModule('bge-reranker-large').start()

llm = lazyllm.TrainableModule('internlm2-chat-20b', stream=True).deploy_method(lazyllm.deploy.Vllm).start()

docs = lazyllm.Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embedding_model)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))
prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n '

with lazyllm.pipeline() as ppl:
    # 在 block 节点组执行两种相似度的检索召回，lazyllm.parallel 并行执行两个检索操作
    with lazyllm.parallel().sum as ppl.prl:
        ppl.prl.retriever1 = lazyllm.Retriever(doc=docs, group_name="CoarseChunk", similarity="cosine", topk=3)
        ppl.prl.retriever2 = lazyllm.Retriever(doc=docs, group_name="block", similarity="bm25_chinese", topk=3)

    # 对上述两个检索器召回的文档进行重排序
    ppl.reranker = lazyllm.Reranker(name='ModuleReranker', model=offline_rerank, topk=3) | bind(query=ppl.input)

    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="".join([node.get_content() for node in nodes]),
            query=query,
        )
    ) | bind(query=ppl.input)

    ppl.llm = llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

query = "2008年有哪些赛事"
print(ppl(query))
input("按回车键部署网页端...")

lazyllm.WebModule(ppl, port=23491, stream=True).start().wait()
