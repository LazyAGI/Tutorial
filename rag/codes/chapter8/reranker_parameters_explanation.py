import lazyllm
from lazyllm import pipeline, bind, OnlineEmbeddingModule, SentenceSplitter, Retriever, Reranker
from lazyllm.tools.rag import Document

prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')

documents = Document(dataset_path="rag_master",
                     embed=OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

with pipeline() as ppl:
    ppl.retriever = Retriever(documents, group_name="sentences", similarity="cosine", similarity_cut_off=0.003, topk=3)
    # default output
    # ppl.reranker = Reranker(name="ModuleReranker",
    #                         model=lazyllm.OnlineEmbeddingModule(type="rerank", source="glm",
    #                                                             embed_model_name="rerank"),
    #                         topk=2) | bind(query=ppl.input)

    # output is dict
    # ppl.reranker = Reranker(name="ModuleReranker",
    #                         model=lazyllm.OnlineEmbeddingModule(type="rerank", source="glm",
    #                                                             embed_model_name="rerank"),
    #                         topk=2, output_format="dict") | bind(query=ppl.input)

    # output is dict AND join is True
    # ppl.reranker = Reranker(name="ModuleReranker",
    #                         model=lazyllm.OnlineEmbeddingModule(type="rerank", source="glm",
    #                                                             embed_model_name="rerank"),
    #                         topk=2, output_format="dict", join=True) | bind(query=ppl.input)

    # output is content AND join is False
    # ppl.reranker = Reranker(name="ModuleReranker",
    #                         model=lazyllm.OnlineEmbeddingModule(type="rerank", source="glm",
    #                                                             embed_model_name="rerank"),
    #                         topk=2, output_format="content" ) | bind(query=ppl.input)

    # output is content AND join is True
    # ppl.reranker = Reranker(name="ModuleReranker",
    #                         model=lazyllm.OnlineEmbeddingModule(type="rerank", source="glm",
    #                                                             embed_model_name="rerank"),
    #                         topk=2, output_format="content", join=True) | bind(query=ppl.input)

    # output is content AND join is '11111111111111111111111111111'
    ppl.reranker = Reranker(name="ModuleReranker",
                            model=lazyllm.OnlineEmbeddingModule(type="rerank", source="glm",
                                                                embed_model_name="rerank"),
                            topk=2, output_format="content",
                            join='11111111111111111111111111111') | bind(query=ppl.input)


nodes = ppl("何为天道")
print(f"nodes: {nodes}")
