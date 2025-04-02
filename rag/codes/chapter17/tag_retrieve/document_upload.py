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
# 1. 初始化知识库， 需要设置API key调用emb， e.g. export LAZYLLM_QWEN_API_KEY=""
# =============================

# 需要使用milvus数据库实现tag检索
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

data_path = "path/to/database"  # 知识库路径
# 注册全局node group
Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)

# 需要自定义doc field，注册需要过滤的tag, 这里以department为例
CUSTOM_DOC_FIELDS = {"department": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' ')}
doc = Document(data_path, name='法务知识库', doc_fields=CUSTOM_DOC_FIELDS, embed={"dense":OnlineEmbeddingModule()}, manager=True, store_conf=get_milvus_store_conf('法务知识库'))


###############################################################
# 2. 上传文档， 注意metadata传入[{"department": "xxx"}]
###############################################################

doc.start()
time.sleep(3600)

###############################################################
# 3. 进行标签检索   @./tag_retriever.py
###############################################################
