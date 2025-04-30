import time

from typing import Dict, List, Optional

import lazyllm
from lazyllm import LOG

from online_models import custom_embedding_model as embedding_model # 使用线上模型

from lazyllm.tools.rag import IndexBase, StoreBase, DocNode
from lazyllm.common import override

DOC_PATH = "/mnt/lustre/share_data/dist/index"


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.uids: set[str] = set()


class TrieTreeIndex(IndexBase):
    def __init__(self, store: 'StoreBase'):
        self.store = store
        self.root = TrieNode()
        self.uid_to_word: Dict[str, str] = {}

    @override
    def update(self, nodes: List['DocNode']) -> None:
        # 仅处理 words 分组
        if not nodes or nodes[0]._group != 'words':
            return
        for n in nodes:
            uid = n._uid
            word = n.text
            self.uid_to_word[uid] = word
            node = self.root
            for char in word:
                node = node.children.setdefault(char, TrieNode())
            node.is_end_of_word = True
            node.uids.add(uid)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        # 仅处理 words 分组
        if group_name != 'words':
            return
        for uid in uids:
            word = self.uid_to_word.pop(uid, None)
            if not word:
                continue
            self._remove(self.root, word, 0, uid)

    def _remove(self, node: TrieNode, word: str, index: int, uid: str) -> bool:
        if index == len(word):
            if uid not in node.uids:
                return False
            node.uids.remove(uid)
            node.is_end_of_word = bool(node.uids)
            return not node.children and not node.uids
        char = word[index]
        child = node.children.get(char)
        if not child:
            return False
        should_delete = self._remove(child, word, index + 1, uid)
        if should_delete:
            del node.children[char]
            return not node.children and not node.uids
        return False

    @override
    def query(self, query: str, group_name: str, **kwargs) -> List[str]:
        node = self.root
        for char in query:
            node = node.children.get(char)
            if node is None:
                return []
        return self.store.get_nodes(group_name=group_name, uids=list(node.uids)) if node.is_end_of_word else []


class LinearSearchIndex(IndexBase):
    def __init__(self):
        self.nodes = []

    @override
    def update(self, nodes: List['DocNode']) -> None:
        if not nodes or nodes[0]._group != 'words':
            return
        for n in nodes:
            self.nodes.append(n)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        if group_name != 'words':
            return
        for uid in uids:
            for i, n in enumerate(self.nodes):
                if n._uid == uid:
                    del self.nodes[i]
                    break

    @override
    def query(self, query: str, **kwargs) -> List[str]:
        # 假设每个单词只出现一次，只进行精准匹配
        res = []
        for n in self.nodes:
            if n.text == query:
                res.append(n)
                break
        return res
        

docs = lazyllm.Document(dataset_path=DOC_PATH, embed=embedding_model)
docs.create_node_group(name='words', transform=(lambda d: d.split('\r\n')))             # 创建节点组
docs.register_index("trie_tree", TrieTreeIndex, docs.get_store())                       # 注册索引
docs.register_index("linear_search", LinearSearchIndex)
retriever1 = lazyllm.Retriever(docs, group_name="words", index="linear_search", topk=1) # 创建检索器，指定索引类型
retriever2 = lazyllm.Retriever(docs, group_name="words", index="trie_tree", topk=1)
retriever1.start()                                                                      # 检索器初始化
retriever2.start()

for query in ["a", "lazyllm", "zwitterionic"]:                                          # 分别检索词表中靠前、中间、靠后单词
    st = time.time()
    res = retriever1(query)
    et = time.time()
    LOG.info(f"query: {query}, linear time: {et - st}, linear res: {res[0].text}")

    st = time.time()
    res = retriever2(query)
    et = time.time()
    LOG.info(f"query: {query}, trie time: {et - st}, trie res: {res[0].text}")
