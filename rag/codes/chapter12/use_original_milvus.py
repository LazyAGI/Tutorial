from online_models import embedding_model
from pymilvus import MilvusClient

# 创建milvus客户端，传入本地数据库的存储路径，若路径不存在则创建
client = MilvusClient("dbs/origin_milvus.db")

# 初始化阶段，如果已存在同名collection，则先删除
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")

client.create_collection(
    collection_name="demo_collection",
    dimension=4096,
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
vecs =[embedding_model(doc) for doc in docs] 
data = [
    {"id": i, "vector": vecs[i], "text": docs[i], "subject": "history"}
    for i in range(len(vecs))
]
# 数据注入
res = client.insert(collection_name="demo_collection", data=data)
print(f"Inserted data into client:\n {res}")

query = "Who is Alan Turing?"
# query向量化
q_vec = embedding_model(query)
# 检索
res = client.search(
    collection_name="demo_collection",    # 指定collection
    data=[q_vec],
    limit=2,    # 指定检索数量（top_k）
    output_fields=["text", "subject"],    #指定检索结果中包含的字段
)
print(f"Query: {query} \nSearch result:\n {res}")


docs2 = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]
vecs2 =[embedding_model(doc) for doc in docs2]
data2 = [
    {"id": 3 + i, "vector": vecs2[i], "text": docs2[i], "subject": "biology"}
    for i in range(len(vecs2))
]
res = client.insert(collection_name="demo_collection", data=data2)
print(f"Inserted data into client:\n {res}")
res = client.search(
    collection_name="demo_collection",
    data=[embedding_model("tell me AI related information")],
    filter="subject == 'biology'",    # 期望过滤的字段
    limit=2,
    output_fields=["text", "subject"],
)
print(f"Filter Query: {query} \nSearch result:\n {res}")
