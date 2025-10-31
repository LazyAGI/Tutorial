import os
import lazyllm
from lazyllm.tools.rag import DocField, DataType
from lazyllm.tools.rag.readers import MineruPDFReader
import atexit

def get_cache_path():
    return os.path.join(lazyllm.config['home'], 'rag_for_qa')

def get_image_path():
    return os.path.join(get_cache_path(), "images")

class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(lazyllm.config['home'], 'rag_for_qa'))
        self.rag_dir = os.path.join(self.root_dir, "rag_master")
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus.db")
        self.image_path = get_image_path()
        atexit.register(self.cleanup)

    def cleanup(self):
        if os.path.isfile(self.store_file):
            os.remove(self.store_file)
        for filename in os.listdir(self.image_path):
            filepath = os.path.join(self.image_path, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
tmp_dir = TmpDir()

# 本地存储
milvus_store_conf = {
    "type": "milvus",
    "kwargs": {
        'uri': tmp_dir.store_file,
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': "COSINE",
        }
    },
}

# 在线服务
# milvus_store_conf = {
#     "type": "milvus",
#     "kwargs": {
#         'uri': "http://your-milvus-server",
#         'index_kwargs': {
#             'index_type': 'HNSW',
#             'metric_type': "COSINE",
#         }
#     },
# }

doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}


embed = lazyllm.TrainableModule("bge-large-zh-v1.5")

documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir,
                             embed=embed.start(),
                             manager=False,
                             store_conf=milvus_store_conf,
                             doc_fields=doc_fields
                             )

documents.add_reader("**/*.pdf", MineruPDFReader(url="http://127.0.0.1:8888"))  # url 需替换为已启动的 MinerU 服务地址
documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

retriever = lazyllm.Retriever(documents, group_name="block", topk=3, output_format='content')

print(retriever('deepseek-r1相关论文的Abstract'))
