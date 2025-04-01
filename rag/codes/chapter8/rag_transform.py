import lazyllm
from lazyllm import pipeline, parallel, bind, Retriever, Reranker
from lazyllm.tools.rag import Document
from lazyllm.tools.rag import DocNode, NodeTransform
from typing import List


class ParagraphSplitter(NodeTransform):
    def __init__(self, splitter: str = r"\n\n", num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        self.splitter = splitter

    def transform(self, node: DocNode, **kwargs) -> List[str]:
        return self.split_text(node.get_text())

    def split_text(self, text: str) -> List[str]:
        if text == '':
            return ['']
        paragraphs = text.split(self.splitter)
        return [para for para in paragraphs]


prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')
documents = Document(dataset_path="rag_master", embed=lazyllm.TrainableModule("bge-large-zh-v1.5"), manager=False)
documents.create_node_group(name="paragraphs", transform=ParagraphSplitter)

with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        ppl.prl.retriever1 = Retriever(documents, group_name="paragraphs", similarity="cosine", topk=3)
        ppl.prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)
    ppl.reranker = Reranker(name="ModuleReranker",
                            model='bge-reranker-large',
                            topk=1, output_format="content", join=True) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule("internlm2_5-7b-chat").prompt(lazyllm.ChatPrompter(prompt,
                                                                                         extra_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23456).start().wait()
