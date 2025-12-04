from lazyllm.tools import Document, Retriever
from lazyllm.tools.rag import DocNode, NodeTransform
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm import OnlineEmbeddingModule

# =============================
# 1. 初始化知识库， 需要设置API key调用emb， e.g. export LAZYLLM_QWEN_API_KEY=""
# =============================

# 定义知识库路径
law_data_path = "path1"
product_data_path = "path2"
support_data_path = "path3"

# 注册全局node group
Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)

# 初始化知识库对象
law_knowledge_base = Document(law_data_path, name='法务知识库', embed=OnlineEmbeddingModule())
product_knowledge_base = Document(product_data_path, name='产品知识库', embed=OnlineEmbeddingModule())
support_knowledge_base = Document(support_data_path, name='用户支持知识库', embed=OnlineEmbeddingModule())


# =============================
# 2. 构建多知识库联合召回
# =============================

# 组合法务 + 产品知识库，处理与产品相关的法律问题
retriever_product = Retriever(
    [law_knowledge_base, product_knowledge_base],
    group_name="sentences",     
    similarity="cosine",       
    topk=1                
)

# 组合法务 + 客服知识库，处理客户合同投诉
retriever_support = Retriever(
    [product_knowledge_base, support_knowledge_base],
    group_name="sentences",
    similarity="cosine",       
    topk=1                
)

product_question = "A产品功能参数和产品合规性声明"
product_response = retriever_product(product_question)
print()
print(f"========== 🚀 query: {product_question } 🚀 ===========")
print()
print(f"========== 🚀 retrieve nodes 🚀 ===============================")
for node in product_response:
    print(node.text)
    print("="*100)

support_question = "B产品的主要成分的投诉的处理方式"
support_response = retriever_support(support_question)
print()
print(f"========== 🚀 query: {support_question } 🚀 ===========")
print()
print(f"========== 🚀 retrieve nodes 🚀 ===============================")
for node in support_response:
    print(node.text)
    print("="*100)