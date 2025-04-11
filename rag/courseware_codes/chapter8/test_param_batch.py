from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import lazyllm
from scipy.linalg import norm
import sys
import heapq
from typing import List, Tuple
from lazyllm.tools.rag import DocNode


def tfidf_similarity(query: str, nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:
    def add_space(s):
        return ' '.join(list(s))
    corpus_tokens = [add_space(node.get_text()) for node in nodes]
    query = add_space(query)
    topk = min(len(nodes), kwargs.get("topk", sys.maxsize))
    datasets = [[query, corpus] for corpus in corpus_tokens]
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    vectors = [cv.fit_transform(corpus).toarray() for corpus in datasets]
    scores = [np.dot(vector[0], vector[1]) / (norm(vector[0]) * norm(vector[1])) for vector in vectors]
    indexes = heapq.nlargest(topk, range(len(scores)), scores.__getitem__)
    results = [(nodes[idx], score) for idx, score in zip(indexes, scores)]
    return results


def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)

# 定义批量数据
query_t = "hello world."
node_t = [DocNode(text="hello lazyllm."), DocNode(text="hello.")]
query_e = {"key": [1.0, 0.4, 2.1]}
node_e = [DocNode(embedding={"key": [4.2, 2.1, 3.9]}), DocNode(embedding={"key": [4.2, 0.0, 3.9]})]

func1 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding", batch=True)
func2 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding")

func3 = lazyllm.tools.rag.register_similarity(tfidf_similarity, mode="text", batch=True)
func4 = lazyllm.tools.rag.register_similarity(tfidf_similarity, mode="text")

ret = func2(query_e, node_e)
print(ret)

