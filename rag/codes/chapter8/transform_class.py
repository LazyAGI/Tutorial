from lazyllm.tools.rag import DocNode, NodeTransform
from typing import List
from lazyllm.tools.rag import Document
import lazyllm
from lazyllm import Retriever

# output is List[str]
# class ParagraphSplitter(NodeTransform):
#     def __init__(self, splitter: str = r"\n\n", num_workers: int = 0):
#         super(__class__, self).__init__(num_workers=num_workers)
#         self.splitter = splitter
#
#     def transform(self, node: DocNode, **kwargs) -> List[str]:
#         return self.split_text(node.get_text())
#
#     def split_text(self, text: str) -> List[str]:
#         if text == '':
#             return ['']
#         paragraphs = text.split(self.splitter)
#         return [para for para in paragraphs]


# output is List[Node]
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
        return [DocNode(content=para) for para in paragraphs]


prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')
documents = Document(dataset_path="rag_master",
                     embed=lazyllm.OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)
documents.create_node_group(name="sentences", transform=ParagraphSplitter)

ppl = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
print(ppl("何为天道"))
