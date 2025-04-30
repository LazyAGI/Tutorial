from lazyllm import Document, Retriever, SentenceSplitter

docs = Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb")

# 使用内置 SentenceSplitter 创建新的 node group
# 此处 chunk_size 和 chunk_overlap 会透传给 SentenceSplitter
# 最终输出的分块规则为 长度为 512，重叠为 64
docs.create_node_group(name='512Chunk', transform=SentenceSplitter, chunk_size=512, chunk_overlap=64)

# 查看节点组内容，此处我们通过一个检索器召回一个节点并打印其中的内容，后续都通过这个方式实现
group_names = ["CoarseChunk", "MediumChunk", "FineChunk", "512Chunk"]
for group_name in group_names:
    retriever = Retriever(docs, group_name=group_name, similarity="bm25_chinese", topk=1)
    node = retriever("亚硫酸盐有什么作用？")
    print(f"======= {group_name} =====")
    print(node[0].get_content())
