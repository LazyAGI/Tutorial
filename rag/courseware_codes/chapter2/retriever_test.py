from lazyllm import Document, Retriever

docs = Document("/home/mnt/zhaoshe/rag_code/rag_master")
separator = '\n' + '='*200 + '\n'  # 定义召回节点内容之间的分隔符（为了便于查看）

retriever = Retriever(docs, group_name="CoarseChunk", similarity="bm25_chinese", topk=2, output_format="dict")
res = retriever("何为大学")
print(res)

    