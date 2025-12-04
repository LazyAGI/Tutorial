import lazyllm

# 定义嵌入模型和重排序模型
# embedding_model = lazyllm.OnlineEmbeddingModule()
embedding_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 如果您要使用在线重排模型
# 目前LazyLLM仅支持 qwen和glm 在线重排模型，请指定相应的 API key。
# online_rerank = lazyllm.OnlineEmbeddingModule(type="rerank")
# 本地重排序模型
offline_rerank = lazyllm.TrainableModule('bge-reranker-large').start()

docs = lazyllm.Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embedding_model)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

# 定义检索器
retriever1 = lazyllm.Retriever(docs, group_name="CoarseChunk", similarity="cosine", topk=3)
retriever2 = lazyllm.Retriever(docs, group_name="block", similarity="bm25_chinese", topk=3)

# 定义重排器
reranker = lazyllm.Reranker('ModuleReranker', model=offline_rerank, topk=3)

# 定义大模型
llm = lazyllm.TrainableModule('internlm2-chat-20b').deploy_method(lazyllm.deploy.Vllm).start()

# prompt 设计
prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n '
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# 执行推理
query = "2008年有哪些赛事？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)
result = reranker(result1+result2, query=query)

# 将query和召回节点中的内容组成dict，作为大模型的输入
res = llm({"query": query, "context_str": "".join([node.get_content() for node in result])})

print(f'Answer: {res}')
