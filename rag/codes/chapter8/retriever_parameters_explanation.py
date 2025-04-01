from lazyllm import OnlineEmbeddingModule, SentenceSplitter, Retriever
from lazyllm.tools.rag import Document

prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')

documents = Document(dataset_path="rag_master",
                     embed=OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

# default output
# ppl = Retriever(documents, group_name="sentences", similarity="cosine", similarity_cut_off=0.003, topk=3)

# output is dict
# ppl = Retriever(documents, group_name="sentences", similarity="cosine",
#                 similarity_cut_off=0.003, topk=3, output_format="dict")

# output is dict AND join is True
# ppl = Retriever(documents, group_name="sentences", similarity="cosine",
#                 similarity_cut_off=0.003, topk=3, output_format="dict", join=True)

# output is content AND join is False
# ppl = Retriever(documents, group_name="sentences", similarity="cosine",
#                 similarity_cut_off=0.003, topk=3, output_format="content")

# output is content AND join is True
# ppl = Retriever(documents, group_name="sentences", similarity="cosine",
#                 similarity_cut_off=0.003, topk=3, output_format="content", join=True)

# output is content AND join is '11111111111111111111111111111'
ppl = Retriever(documents, group_name="sentences", similarity="cosine", similarity_cut_off=0.003,
                topk=3, output_format="content", join='11111111111111111111111111111')

nodes = ppl("何为天道")
print(f"nodes: {nodes}")
