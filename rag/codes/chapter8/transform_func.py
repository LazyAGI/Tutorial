import lazyllm
from lazyllm import Retriever
from lazyllm.tools.rag import Document, DocNode

# output is List[str]
# def SentSplitter(text: str, splitter: str="\n"):
#     print(f"text: {type(text)}")
#     if text == '':
#         return ['']
#     paragraphs = text.split(splitter)
#     return [para for para in paragraphs]


# output is List[Node]
def SentSplitter(text: str, splitter: str = "\n"):
    print(f"text: {type(text)}")
    if text == '':
        return ['']
    paragraphs = text.split(splitter)
    return [DocNode(content=para) for para in paragraphs if para]


prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')
documents = Document(dataset_path="rag_master",
                     embed=lazyllm.OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)

# trans_node is False
documents.create_node_group(name="sentences", transform=SentSplitter)

# trans_node is True
# documents.create_node_group(name="sentences", transform=SentSplitter, trans_node=True)

ppl = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
print(ppl("何为天道"))
