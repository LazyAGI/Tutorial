import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import heapq
from typing import List, Tuple
from lazyllm.tools.rag import DocNode
import os


def tfidf_similarity(query: str, nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:
    def add_space(s):
        return ' '.join(list(s))
    corpus = [add_space(node.get_text()) for node in nodes]
    query = add_space(query)
    topk = min(len(nodes), kwargs.get("topk", sys.maxsize))
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    tfidf_matrix = cv.fit_transform(corpus+[query])
    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    similairyties = cosine_similarity(query_vec, doc_vecs).flatten()

    indexes = heapq.nlargest(topk, range(len(similairyties)), similairyties.__getitem__)
    results = [(nodes[i], similairyties[i]) for i in indexes]
    return results


query = "今天天气怎么样"

candidates = [
    DocNode(text="今天阳光明媚"),
    DocNode(text="今天天气怎么样"),
    DocNode(text="今天天气非常好"),
    DocNode(text="我喜欢吃苹果"),
    DocNode(text="今天天气真糟糕")
]

results = tfidf_similarity(query, candidates, topk=2)

print("Query:", query)
print("Scores:")
for node, score in results:
    print(f"{node.get_text()} -> 相似度: {score:.4f}")

