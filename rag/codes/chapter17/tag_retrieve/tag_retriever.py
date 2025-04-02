import time
import uuid
import os

from lazyllm.tools import Document, Retriever
from lazyllm.tools.rag import DocNode, NodeTransform
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.global_metadata import GlobalMetadataDesc as DocField
from lazyllm.tools.rag import DataType
from lazyllm import OnlineEmbeddingModule


# =============================
# 1. 定义知识库
# =============================

MILVUS_BASE_PATH = "path/to/milvus"  # 需要提供一个目录作为milvus库地址
def get_milvus_store_conf(kb_group_name: str = str(uuid.uuid4())):
    db_path = os.path.join(MILVUS_BASE_PATH, kb_group_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    milvus_store_conf = {
        'type': 'milvus',
        'kwargs': {
            'uri': os.path.join(db_path, "milvus.db"),
        'index_kwargs': [
                {
                    'embed_key': 'dense',
                    'index_type': 'IVF_FLAT',
                    'metric_type': 'COSINE',
                },
            ]
        },
    }
    return milvus_store_conf



# 定义知识库路径
data_path = "path/to/database"  # 知识库路径
# 注册全局node group
Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)

# 需要自定义doc field，注册需要过滤的tag
CUSTOM_DOC_FIELDS = {"department": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' ')}
doc = Document(data_path, name='法务知识库', doc_fields=CUSTOM_DOC_FIELDS, embed={"dense":OnlineEmbeddingModule()}, store_conf=get_milvus_store_conf('法务知识库'))


# =============================
# 2. 实现tag过滤的检索
# =============================

retriever = Retriever(
    doc,
    group_name="sentences",   
    topk=5, 
    embed_keys=['dense']
)

query = "合同问题"
nodes = retriever(query, filters={'department': ['法务一部']})
print()
print(f"========== 🚀 query: {query} 🚀 ===========")
print()
print(f"========== 🚀 retrieve nodes 🚀 ===========")
for node in nodes:
    print(node.text)
    print("="*100)
    