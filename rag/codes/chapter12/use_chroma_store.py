import time
import lazyllm

chroma_store_conf = {
  'type': 'chroma',
  'kwargs': {
    'dir': 'testdb',  # chromadb传入的 dir 是一个文件夹，不存在会自动创建
   }
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
# 使用 chromadb 作为存储后端
document2 = lazyllm.Document(dataset_path='/mnt/lustre/share_data/dist/cmrc2018/data_kb',
                             embed=bge_embed,
                             store_conf=chroma_store_conf)
document2.create_node_group(name="block", transform=lambda s: s.split("\n"))
retriever2 = lazyllm.Retriever(doc=document2,
                               group_name="block",
                               similarity="cosine",
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
print("==ChromaDB存储后端耗时==：", end2-start2)
