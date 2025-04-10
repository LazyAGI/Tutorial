from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import heapq
from typing import List, Tuple
from lazyllm.tools.rag import DocNode
import lazyllm
import os
from lazyllm import OnlineEmbeddingModule, SentenceSplitter, Retriever
from lazyllm.tools.rag import Document


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


def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)


# Comparison func
func1 = lazyllm.tools.rag.register_similarity(tfidf_similarity)
func2 = lazyllm.tools.rag.register_similarity()
print(f"func1: {func1.__name__}, func2: {func2.__name__}")

# Comparison mode
func1 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="text")
func2 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding")
query_t = "hello world."
node_t = [DocNode(text="hello lazyllm.")]
query_e = {"key": [1.0, 0.4, 2.1]}
node_e = [DocNode(embedding={"key": [4.2, 2.1, 3.9]})]

# Use func1 to calculate the similarity of text
ret = func1(query_t, node_t)
print(f"ret: {ret}")

# Calculate vector similarity using func1
ret = func1(query_e, node_e)
print(f"ret: {ret}")

# Use func2 to calculate the similarity of text
ret = func2(query_t, node_t)
print(f"ret: {ret}")

# Calculate vector similarity using func2
ret_2e = func2(query_e, node_e)
print(f"ret_2e: {ret_2e}")


# Comparison descend(Only one of the two euclidean_distance similarity functions can be selected)
# descend default value is True
@lazyllm.tools.rag.register_similarity(mode="embedding")
def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)


@lazyllm.tools.rag.register_similarity(mode="embedding", descend=False)
def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)


prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')

documents = Document(dataset_path=os.path.join(os.getcwd(), "rag_data"),
                     embed=OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
ppl = Retriever(documents, group_name="CoarseChunk", similarity="euclidean_distance", similarity_cut_off=0.003, topk=3)

nodes = ppl("全国住房城乡建设工作会议的主要内容")
for node in nodes:
    print(f"node: {node.text}")

# Comparison batch
func1 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding", batch=True)
func2 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding")
ret = func1(query_e, node_e)
print(f"ret: {ret}")
ret = func2(query_e, node_e)
print(f"ret: {ret}")

func3 = lazyllm.tools.rag.register_similarity(tfidf_similarity, mode="text", batch=True)
func4 = lazyllm.tools.rag.register_similarity(tfidf_similarity, mode="text")
ret = func3(query_t, node_t)
print(f"ret: {ret}")
ret = func4(query_t, node_t)
print(f"ret: {ret}")
