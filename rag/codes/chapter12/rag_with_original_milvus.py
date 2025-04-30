import lazyllm
from lazyllm.tools.rag import SimpleDirectoryReader, SentenceSplitter
from lazyllm.tools.rag.doc_node import MetadataMode

from pymilvus import MilvusClient

from online_models import custom_embedding_model as embedding_model, llm, rerank_model

DOC_PATH = "/mnt/lustre/share_data/dist/test_docs" 

# milvus client初始化
client = MilvusClient("dbs/rag_milvus.db")
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=1024,
)

# 加载本地文件，并解析 -- block切片 -- 向量化 -- 入库
docs = SimpleDirectoryReader(input_dir=DOC_PATH)()
block_transform = SentenceSplitter(chunk_size=256, chunk_overlap=25)

nodes = []
for doc in docs:
    nodes.extend(block_transform(doc))

# 切片向量化
vecs = [embedding_model(node.get_text(MetadataMode.EMBED)) for node in nodes]
data = [
    {"id": i, "vector": vecs[i], "text": nodes[i].text}
    for i in range(len(vecs))
]
#数据注入
res = client.insert(collection_name="demo_collection", data=data)


# 检索与生成
prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
        根据以下资料回答问题：\
        {context_str} \n '
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

query = "证券管理的基本规范？"
q_vec = embedding_model(query)
res = client.search(
    collection_name="demo_collection",
    data=[q_vec],
    limit=12,
    output_fields=["text"],
)

# 提取检索结果
contexts = [res[0][i].get('entity').get("text", "") for i in range(len(res[0]))]

# 重排
rerank_res = rerank_model(text=query, documents=contexts, top_n=3)
rerank_contexts = [contexts[res[0]] for res in rerank_res]

context_str = "\n-------------------\n".join(rerank_contexts)
res = llm({"query": query, "context_str": context_str})
print(res)
