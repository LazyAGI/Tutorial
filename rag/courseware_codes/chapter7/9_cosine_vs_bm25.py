# 探讨相似度对召回率的影响
# 请将您的在线嵌入模型api-key抛出为环境变量或更改为本地模型
import lazyllm
from lazyllm import Document, Retriever, TrainableModule

# 定义 embedding 模型
embed_model = TrainableModule("bge-large-zh-v1.5").start()

# 文档加载
docs = Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embed_model)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

# 定义两个不同的检索器在同一个节点组使用不同相似度方法进行检索
retriever1 = Retriever(docs, group_name="block", similarity="cosine", topk=3)
retriever2 = Retriever(docs, group_name="block", similarity="bm25_chinese", topk=3)

# 执行查询
query = "都有谁参加了2008年的奥运会？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)

print("余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result1]))
print("bm25召回结果：")
print("\n\n".join([res.get_content() for res in result2]))
