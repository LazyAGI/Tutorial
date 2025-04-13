from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import heapq
from typing import List, Tuple
from lazyllm.tools.rag import DocNode
import lazyllm
import os
from lazyllm import OnlineEmbeddingModule, SentenceSplitter, Retriever
from lazyllm.tools.rag import Document


@lazyllm.tools.rag.register_similarity(mode='text', batch=True)
def tfidf_similarity(query: str, nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:
    def add_space(s):
        return ' '.join(list(s))
    corpus = [add_space(node.get_text()) for node in nodes]
    query = add_space(query)
    topk = min(len(nodes), kwargs.get("topk", sys.maxsize))
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    tfidf_matrix = cv.fit_transform(corpus + [query])
    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    similairyties = cosine_similarity(query_vec, doc_vecs).flatten()

    indexes = heapq.nlargest(topk, range(len(similairyties)), similairyties.__getitem__)
    results = [(nodes[i], similairyties[i]) for i in indexes]
    return results


prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')

documents = Document(dataset_path=os.path.join(os.getcwd(), "rag_data"),
                     embed=OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

# similarity_cut_off is 0.003
# ppl = Retriever(documents, group_name="CoarseChunk", similarity="tfidf_similarity", similarity_cut_off=0.003, topk=3)
# similarity_cut_off is 0.21
ppl = Retriever(documents, group_name="CoarseChunk", similarity="tfidf_similarity", similarity_cut_off=0.21, topk=3)

nodes = ppl("全国住房城乡建设工作会议的主要内容")
print(f"nodes: {nodes}")
for node in nodes:
    print(f"node: {node.similarity_score}")
