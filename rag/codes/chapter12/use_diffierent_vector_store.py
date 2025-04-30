import time

import lazyllm
from lazyllm import LOG

from online_models import custom_embedding_model as embedding_model    # 使用线上模型

DOC_PATH = "/mnt/lustre/share_data/dist/test_docs" 

def test_store(store_conf: dict=None):
    """接收存储配置，测试不同配置下系统启动性能"""
    st1 = time.time()

    docs = lazyllm.Document(dataset_path=DOC_PATH, embed=embedding_model, store_conf=store_conf)
    docs.create_node_group(name='sentence', parent="MediumChunk", transform=lambda x: x.split('。'))
    
    if store_conf and store_conf.get('type') == "milvus":
        # 存储类型为milvus时，无需similarity参数
        retriever1 = lazyllm.Retriever(docs, group_name="sentence", topk=3)
    else:
        # similariy=cosine，使用向量检索
        retriever1 = lazyllm.Retriever(docs, group_name="sentence", similarity='cosine', topk=3)
    
    retriever1.start()  # 启动检索器
    et1 = time.time()

    # 测试单次检索耗时
    st2 = time.time()
    res = retriever1("牛车水")
    et2 = time.time()
    nodes = "\n======\n".join([node.text for node in res])  # 输出检索结果
    msg = f"Init time: {et1 - st1}, retrieval time: {et2 - st2}s\n" # 输出系统耗时
    LOG.info(msg)
    LOG.info(nodes)
    return msg


# chroma db存储配置
chroma_store_conf = {
    'type': 'chroma', 
    'kwargs': {
        'dir': 'dbs/chroma1',
    }
}
# milvus存储配置
milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': 'dbs/milvus1.db',
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        }
    },
}

# 测试集，依次测试使用内存、chroma、milvus时的系统启动性能
test_conf = {
    "map": None,
    "chroma": chroma_store_conf,
    "milvus": milvus_store_conf
}
start_times = ""
for store_type, store_conf in test_conf.items():
    # 调用测试函数
    res = test_store(store_conf=store_conf)
    start_times += f"Store type: {store_type}: {res}"
print(start_times)
