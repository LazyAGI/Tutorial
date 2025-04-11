import os
import lazyllm
from typing import List
import numpy as np
from lazyllm.tools.rag import DocNode

def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)


query_t = "hello world."
node_t = [DocNode(text="hello lazyllm.")]
query_e = {"key": [1.0, 0.4, 2.1]}
node_e = [DocNode(embedding={"key": [4.2, 2.1, 3.9]})]


func1 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="text")
func2 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding")

ret = func1(query_e, node_e)
print(ret)
