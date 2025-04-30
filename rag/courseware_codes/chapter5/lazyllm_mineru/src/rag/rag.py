# -*- coding: utf-8 -*-
# flake8: noqa: F821

import lazyllm
from lazyllm import LOG
from lazyllm import pipeline, parallel, bind, OnlineEmbeddingModule, SentenceSplitter, Document, Retriever, Reranker
from dotenv import load_dotenv

load_dotenv()

prompt = 'You will play the role of an AI question-answering assistant and complete a conversation task in which you need to provide your answer based on the given context and question. Please note that if the given context cannot answer the question, do not use your prior knowledge but tell the user that the given context cannot answer the question.'

documents = Document(dataset_path="", embed=OnlineEmbeddingModule(), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)


with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", embed_keys=["dense"], similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, group_name="sentences", embed_keys=["sparse"], similarity="bm25_chinese", topk=3)
    ppl.reranker = Reranker("ModuleReranker", model=OnlineEmbeddingModule(type="rerank"), topk=1, output_format='content', join=True) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))


if __name__ == "__main__":
    while True:
        print("✨  Welcome to your smart assistant ✨")
        
        query = input("\n🚀  Enter your query (type 'exit' to quit): \n> ")
        if query.lower() == "exit":
            print("\n👋  Exiting... Thank you for the using!")
            break

        print(f"\n✅  Received your query: {query}\n")

        answer = ppl(query)

        print("\n" + "=" * 50)
        print("🚀  ANSWER  🚀")
        print("=" * 50 + "\n")
        print(answer)
        print("\n" + "=" * 50 + "\n")
