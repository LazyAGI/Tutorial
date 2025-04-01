import time
import lazyllm

index_store_conf = {
  'type': 'map',
  'indices': {
    'smart_embedding_index': {
      'backend': 'milvus',
      'kwargs': {
        'uri': "dbs/test.db",
        'index_kwargs': {
          'index_type': 'HNSW',
          'metric_type': 'COSINE',
        }
      },
    },
  },
}

bge_embed = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 不使用索引后端
document1 = lazyllm.Document(dataset_path='/mnt/lustre/share_data/dist/cmrc2018/data_kb',
                             embed=bge_embed)
document1.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')
retriever1 = lazyllm.Retriever(doc=document1,
                               group_name="block",
                               similarity="cosine",
                               topk=3)
# 使用 milvus 索引后端
document2 = lazyllm.Document(dataset_path='/mnt/lustre/share_data/dist/cmrc2018/data_kb',
                             embed=bge_embed,
                             store_conf=index_store_conf)
document2.create_node_group(name="block", transform=lambda s: s.split("\n"))
retriever2 = lazyllm.Retriever(doc=document2,
                               group_name="block",
                               topk=3,
                               index='smart_embedding_index')

query = "猴面包树的别名是什么？"

# 执行 Lazy init
retriever1(query)
retriever2(query)

# 记录检索时间
start1 = time.time()
retriever1(query)
end1 = time.time()
print("==默认索引耗时==：", end1-start1)

start2 = time.time()
retriever2(query)
end2 = time.time()
print("==Milvus索引耗时==：", end2-start2)
