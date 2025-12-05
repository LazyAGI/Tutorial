import lazyllm
from lazyllm import (pipeline, bind, OnlineEmbeddingModule, SentenceSplitter, Reranker,
                     Document, Retriever, fc_register, FunctionCallAgent)

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, you need to provide your answer based on the given context and question.'

documents = Document(dataset_path="rag_master", embed=OnlineEmbeddingModule(), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
with pipeline() as ppl_rag:
    ppl_rag.retriever = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
    ppl_rag.reranker = Reranker("ModuleReranker", model=OnlineEmbeddingModule(type="rerank"), topk=1, output_format='content', join=True) | bind(query=ppl_rag.input)

@fc_register("tool")
def search_knowledge_base(query: str):
    '''
    Get info from knowledge base in a given query.

    Args:
        query (str): The query for search knowledge base.
    '''
    return ppl_rag(query)

tools = ["search_knowledge_base"]

with pipeline() as ppl:
    ppl.retriever = FunctionCallAgent(lazyllm.OnlineChatModule(stream=False), tools)
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

if __name__ == "__main__":
    lazyllm.WebModule(ppl, port=range(23467, 24000)).start().wait()