from lazyllm.tools import Document, Retriever
from lazyllm.tools.rag import DocNode, NodeTransform
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm import OnlineEmbeddingModule


# =============================
# 1. 定义DFA过滤器，封装为node transform
# =============================
class DFAFilter:
    def __init__(self, sensitive_words):
        self.root = {}
        self.end_flag = "is_end"
        for word in sensitive_words:
            self.add_word(word)

    def add_word(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end_flag] = True

    def filter(self, text, replace_char="*"):
        result = []
        start = 0
        length = len(text)

        while start < length:
            node = self.root
            i = start
            while i < length and text[i] in node:
                node = node[text[i]]
                if self.end_flag in node:
                    # 匹配到敏感词，替换为指定字符
                    result.append(replace_char * (i - start + 1))
                    start = i + 1
                    break
                i += 1
            else:
                # 未匹配到敏感词，保留原字符
                result.append(text[start])
                start += 1

        return ''.join(result)
   
   
# 注册为tranform
class DFATranform(NodeTransform):
    def __init__(self, sensitive_words):
        super(__class__, self).__init__(num_workers=1)
        self.dfafilter = DFAFilter(sensitive_words)

    def transform(self, node: DocNode, **kwargs):
        return self.dfafilter.filter(node.get_text())

    def split_text(self, text: str):
        if text == '':
            return ['']
        paragraphs = text.split(self.splitter)
        return [para for para in paragraphs]

# DFATranform注册为node group
sensitive_words = ['合同']  # 需要过滤的敏感词
Document.create_node_group(name="dfa_filter", parent="sentences", transform=DFATranform(sensitive_words))

# =============================
# 2. 初始化知识库， 需要设置API key调用emb， e.g. export LAZYLLM_QWEN_API_KEY=""
# =============================

# 定义知识库路径
law_data_path = ""
product_data_path = ""
support_data_path = ""

# 再注册一个 sentences 用来对比
Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)


# 初始化知识库对象
law_knowledge_base = Document(law_data_path, name='法务知识库', embed=OnlineEmbeddingModule()) 
product_knowledge_base = Document(product_data_path, name='产品知识库', embed=OnlineEmbeddingModule())
support_knowledge_base = Document(support_data_path, name='用户支持知识库', embed=OnlineEmbeddingModule())


# =============================
# 3. 构建多知识库联合召回
# =============================

# 组合法务 + 产品知识库，处理与产品相关的法律问题
retriever_product = Retriever(
    [law_knowledge_base, product_knowledge_base],
    group_name="dfa_filter",     # 可切换 sentences 对比结果
    similarity="cosine",       
    topk=1                
)

# 组合法务 + 客服知识库，处理客户合同投诉
retriever_support = Retriever(
    [product_knowledge_base, support_knowledge_base],
    group_name="dfa_filter",
    similarity="cosine",       
    topk=1                
)

if __name__ == "__main__":
    product_question = "A产品功能参数和产品合规性声明"
    product_response = retriever_product(product_question)
    print()
    print(f"========== 🚀 query: {product_question } 🚀 ===========")
    print()
    print(f"========== 🚀 retrieve nodes 🚀 ===============================")
    for node in product_response:
        print(node.text)
        print("="*100)

    support_question = "B产品的主要成分的投诉的处理方式"
    support_response = retriever_support(support_question)
    print()
    print(f"========== 🚀 query: {product_question } 🚀 ===========")
    print()
    print(f"========== 🚀 retrieve nodes 🚀 ===============================")
    for node in support_response:
        print(node.text)
        print("="*100)
    
