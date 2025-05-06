import lazyllm
import os
import os
import os, sys

from lazyllm import OnlineEmbeddingModule, SentenceSplitter, Retriever
from lazyllm.tools.rag import Document

import lazyllm
from typing import List
import numpy as np

@lazyllm.tools.rag.register_similarity(mode="embedding", descend=False)
def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)