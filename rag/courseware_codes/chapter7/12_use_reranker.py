from lazyllm import Document, Retriever, Reranker, TrainableModule

# 定义嵌入模型和重排序模型
embedding_model = TrainableModule('bge-large-zh-v1.5').start()

# 如果您要使用在线重排模型
# 目前LazyLLM仅支持 qwen和glm 在线重排模型，请指定相应的 API key。
# online_rerank = OnlineEmbeddingModule(type="rerank")
# 本地重排序模型
offline_rerank = TrainableModule('bge-reranker-large').start()

docs = Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embedding_model)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

# 定义检索器
retriever = Retriever(docs, group_name="block", similarity="cosine", topk=3)

# 定义重排器
# 指定 reranker 的输出为字符串，并进行串联，未进行串联时输出为字符串列表
reranker = Reranker('ModuleReranker', model=offline_rerank, topk=3, output_format='content', join=True)

# 执行推理
query = "猴面包树有哪些功效？"
result1 = retriever(query=query)
result2 = reranker(result1, query=query)

print("余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result1]))
print("开源重排序模型结果：")
print(result2)
