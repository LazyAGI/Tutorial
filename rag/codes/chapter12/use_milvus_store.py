import time
import lazyllm

milvus_store_conf = {
  'type': 'milvus',  # 指定存储后端类型
  'kwargs': {
    'uri': 'dbs/test.db',  # 存储后端地址，本例子使用的是本地文件 test.db，文件不存则创建新文件
    'index_kwargs': {  # 存储后端的索引配置
      'index_type': 'FLAT',  # 索引类型
      'metric_type': 'COSINE',  # 相似度计算方式
    }
  },
}

bge_embed = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 不使用存储后端
document1 = lazyllm.Document(dataset_path='/mnt/lustre/share_data/dist/cmrc2018/data_kb',
                             embed=bge_embed)
document1.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')
retriever1 = lazyllm.Retriever(doc=document1,
                               group_name="block",
                               similarity="cosine",
                               topk=3)
# 使用 milvus 作为存储后端
document2 = lazyllm.Document(dataset_path='/mnt/lustre/share_data/dist/cmrc2018/data_kb',
                             embed=bge_embed,
                             store_conf=milvus_store_conf)
document2.create_node_group(name="block", transform=lambda s: s.split("\n"))
retriever2 = lazyllm.Retriever(doc=document2,
                               group_name="block",
                               topk=3)

query = "猴面包树的别名是什么？"

# 记录重启后首次检索时间
start1 = time.time()
retriever1(query)
end1 = time.time()
print("==基于内存耗时==：", end1-start1)

start2 = time.time()
retriever2(query)
end2 = time.time()
print("==Milvus存储后端耗时==：", end2-start2)
