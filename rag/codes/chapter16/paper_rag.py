import os
import sys
from lazyllm.tools.rag import DocField, DataType
from lazyllm import OnlineChatModule, pipeline, _0, OnlineEmbeddingModule, bind
import lazyllm
from lazyllm.tools import IntentClassifier
from lazyllm.tools.agent import ReactAgent
from lazyrag.components.embedding import LazyEmbeddingFactory
from utils import bi_tools


class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser("./rag_data/papers")  # os.path.join(lazyllm.config['home'], 'rag_for_qa'))
        self.rag_dir = os.path.join(self.root_dir, "papers")
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus.db")
        self.image_path = "./images"


tmp_dir = TmpDir()

milvus_store_conf = {
    "type": "milvus",
    "kwargs": {
        'uri': tmp_dir.store_file,
        'index_kwargs': {
            'index_type': 'FLAT',
            'metric_type': "COSINE",
        }
    },
}


doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}





def build_paper_rag():
    prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
        ' In this task, you need to provide your answer based on the given context and question.'

    documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir,
                                 embed=OnlineEmbeddingModule(),
                                 manager=False,
                                 store_conf=milvus_store_conf,
                                 doc_fields=doc_fields)

    with lazyllm.pipeline() as rag_ppl:
        rag_ppl.retriever = lazyllm.Retriever(documents, group_name="CoarseChunk", topk=3)

        rag_ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                             model=OnlineEmbeddingModule(type='rerank'),
                                             topk=1,
                                             output_format='content',
                                             join=True) | bind(query=rag_ppl.input)

        rag_ppl.formatter = (
            lambda nodes, query: dict(context_str=nodes, query=query)
        ) | bind(query=rag_ppl.input)

        rag_ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(
            lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

    return rag_ppl

