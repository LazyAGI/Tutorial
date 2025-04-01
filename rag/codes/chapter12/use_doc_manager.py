import time
import lazyllm

milvus_store_conf = {
  'type': 'milvus',  # 指定存储后端类型
  'kwargs': {
    'uri': 'dbs/test.db',  # 存储后端地址，本例子使用的是本地文件 test.db，文件不存则创建新文件
    'index_kwargs': {  # 存储后端的索引配置
      'index_type': 'HNSW',  # 索引类型
      'metric_type': 'COSINE',  # 相似度计算方式
    }
  },
}

document = lazyllm.Document(dataset_path='/mnt/lustre/share_data/dist/cmrc2018/data_kb',
                            store_conf=milvus_store_conf,
                            manager='ui')  # manager='ui'

document.start().wait()
doc_manager_url = document._manager.url
# doc_manager_url 应为形如 http://127.0.0.1:12345/generate 的地址
# 有效部分为 http://127.0.0.1:12345

time.sleep(1000)
