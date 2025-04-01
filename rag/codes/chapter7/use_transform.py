from typing import List, Union
from lazyllm import Document, Retriever, SentenceSplitter
from lazyllm.tools.rag.doc_node import DocNode

docs = Document("/path/to/your/documents")


# 第一种：函数实现直接对字符串进行分块规则
def split_by_sentence1(node: str, **kwargs) -> List[str]:
    """函数接收字符串，返回字符串列表，输入为trans_node=False时调用"""
    return node.split('。')


docs.create_node_group(name='block1', transform=split_by_sentence1)


# 第二种：函数实现获取DocNode对应文本内容后进行分块，并构造DocNode
# 适用于返回非朴素DocNode，例如LazyLLM提供了ImageDocNode等特殊DocNode
def split_by_sentence2(node: DocNode, **kwargs) -> List[DocNode]:
    """函数接收DocNode，返回DocNode列表，输入为trans_node=False时调用"""
    content = node.get_text()
    nodes = []
    for text in content.split('。'):
        nodes.append(DocNode(text=text))
    return nodes


docs.create_node_group(name='block2', transform=split_by_sentence2, trans_node=True)


# 第三种：实现了 __call__ 函数的类
# 优点是一个类用于多种分块，例如这个例子可以通过控制实例化时的参数实现基于多种符号的分块
class SymbolSplitter:
    """实例化后传入Transform，默认情况下接收字符串，trans_node为true时接收DocNode"""
    def __init__(self, splitter="。", trans_node=False):
        self._splitter = splitter
        self._trans_node = trans_node

    def __call__(self, node: List[Union[str, DocNode]]) -> List[Union[str, DocNode]]:
        if self._trans_node:
            return node.get_text().split(self._splitter)
        return node.split(self._splitter)


sentence_splitter_1 = SymbolSplitter()
docs.create_node_group(name='block3', transform=sentence_splitter_1)

sentence_splitter_2 = SymbolSplitter(trans_node=True)
docs.create_node_group(name='block4', transform=sentence_splitter_2, trans_node=True)

paragraph_splitter = SymbolSplitter(splitter="\n")
docs.create_node_group(name='block5', transform=paragraph_splitter)

# 第四种：直接传入lambda函数，适用于简单规则情况
docs.create_node_group(name='block6', transform=lambda b: b.split('。'))

# 使用内置 SentenceSplitter
# 此处 chunk_size 和 chunk_overlap 会透传给 SentenceSplitter
# 最终输出的分块规则为 长度为 256，重叠为 25
docs.create_node_group(name='256Chunk', transform=SentenceSplitter, chunk_size=256, chunk_overlap=25)
