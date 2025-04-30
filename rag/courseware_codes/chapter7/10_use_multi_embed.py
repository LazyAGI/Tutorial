# 探讨不同embedding对召回率的影响
# 请将您的在线嵌入模型api-key抛出为环境变量或使用本地模型

from lazyllm import Document, Retriever, TrainableModule

# 定义多个 embedding 模型
bge_m3_embed = TrainableModule('bge-m3').start()
bge_large_embed = TrainableModule('bge-large-zh-v1.5').start()
embeds = {'vec1': bge_m3_embed, 'vec2': bge_large_embed}

# 文档加载
docs = Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embeds)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

# 定义两个不同的检索器分别对相同节点组的不同 embedding 进行检索
retriever1 = Retriever(docs, group_name="block", embed_keys=['vec1', 'vec2'], similarity="cosine", topk=3)
retriever2 = Retriever(docs, group_name="block", embed_keys=['vec2'], similarity="bm25", topk=3)

# 执行检索
query = "都有谁参加了2008年的奥运会？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)

print("使用bge-m3和bge-large-zh-v1.5进行余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result1]))
print("使用bge-large-zh-v1.5进行余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result2]))
