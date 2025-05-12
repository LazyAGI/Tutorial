import time
import json
from io import BytesIO
import requests
import uuid
import os
import lazyllm
from lazyllm.tools import Document, Retriever
from lazyllm.tools.rag import DocNode, NodeTransform
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.global_metadata import GlobalMetadataDesc as DocField
from lazyllm.tools.rag import DataType
from lazyllm.tools import IntentClassifier
from lazyllm import OnlineEmbeddingModule, pipeline, Reranker, OnlineChatModule, bind, _0
import lazyllm
from lazyllm import globals
from pydantic import BaseModel
from typing import List, Optional

# ==================================================
# 1. 定义DFA算法并注册为 node transform
# ==================================================

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

# DFATranform注册为node group, 屏蔽敏感词 “垄断”
Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)
Document.create_node_group(name="dfa_filter", parent="sentences", transform=DFATranform(["垄断"]))


# ===================================================
# 2. 定义知识库
# ===================================================

def get_milvus_store_conf(kb_group_name: str = str(uuid.uuid4())):
    db_path = os.path.join(os.path.abspath("./data/milvus"), kb_group_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    milvus_store_conf = {
        'type': 'milvus',
        'kwargs': {
            'uri': os.path.join(db_path, "milvus.db"),
        'index_kwargs': [
                {
                    'embed_key': 'dense',
                    'index_type': 'IVF_FLAT',
                    'metric_type': 'COSINE',
                },
            ]
        },
    }
    return milvus_store_conf



# 定义知知识库并上传文件
product_path = os.path.abspath("./data/ecommerce_data/产品知识库")
law_path = os.path.abspath("./data/ecommerce_data/法务知识库")
support_path = os.path.abspath("./data/ecommerce_data/用户支持知识库")

CUSTOM_DOC_FIELDS = {"category": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '), "performance": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' ')}
doc_product = Document(product_path, name='产品知识库', doc_fields=CUSTOM_DOC_FIELDS, embed={"dense":OnlineEmbeddingModule()}, store_conf=get_milvus_store_conf('产品知识库'), manager=True)
doc_law = Document(law_path, name='法务知识库', embed={"dense":OnlineEmbeddingModule()}, store_conf=get_milvus_store_conf('法务知识库'), manager=True)
doc_support = Document(support_path, name='用户支持知识库', embed={"dense":OnlineEmbeddingModule()}, store_conf=get_milvus_store_conf('用户支持知识库'), manager=True)

def upload_files():
    with open("./data/ecommerce_data.json", 'r', encoding='utf-8') as file:
        ecommerce_data = json.load(file)  

    url_map = {}
    doc_product.start()
    url_map["产品知识库"] = doc_product._manager.url.rsplit('/', 1)[0] + "/upload_files"
    doc_law.start()
    url_map["法务知识库"] = doc_law._manager.url.rsplit('/', 1)[0] + "/upload_files"
    doc_support.start()
    url_map["用户支持知识库"] = doc_support._manager.url.rsplit('/', 1)[0] + "/upload_files"

    for group_name, docs in ecommerce_data.items():
        files = []
        metadatas = []
        
        for doc in docs:
            file_content = doc["content"].encode("utf-8")
            file_obj = BytesIO(file_content)
            files.append(
                ("files", (doc["name"]+'.txt', file_obj, "text/plain"))
            )
            metadatas.append(doc["key_words"])
        
        data = {
            "group_name": group_name,
            "metadatas": json.dumps(metadatas),  
        }
        
        response = requests.post(
            url_map[group_name],
            params={'group_name': group_name},
            files=files,
            data=data,
        )
        
        print(f"上传到 {group_name} 的响应：")
        print(response.status_code, response.json())
    time.sleep(20)


# =================================================
# 3. 定义两条交叉检索的 pipeline，以及主 pipeline
# =================================================

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
    ' In this task, you need to provide your answer based on the given context and question.'

with pipeline() as product_law_ppl:
    product_law_ppl.retriever = Retriever(
            [doc_law, doc_product],
            group_name="dfa_filter",   
            topk=5, 
            embed_keys=['dense'],
        )
    product_law_ppl.reranker = Reranker(name="ModuleReranker",
                            model=OnlineEmbeddingModule(type='rerank'),
                            topk=2, output_format="content", join=True) | bind(query=product_law_ppl.input)
    product_law_ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=product_law_ppl.input)
    product_law_ppl.llm = OnlineChatModule().prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

with pipeline() as support_law_ppl:
    support_law_ppl.retriever =Retriever(
            [doc_law, doc_support],
            group_name="dfa_filter",   
            topk=5, 
            embed_keys=['dense'],
        )
    support_law_ppl.reranker = Reranker(name="ModuleReranker",
                            model=OnlineEmbeddingModule(type='rerank'),
                            topk=2, output_format="content", join=True) | bind(query=support_law_ppl.input)
    support_law_ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=support_law_ppl.input)
    support_law_ppl.llm = OnlineChatModule().prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))


# 搭建主工作流
def build_ecommerce_assistant():
    llm = OnlineChatModule(source='qwen', stream=False)
    intent_list = [
        "产品法务问题",
        "用户支持问题",
    ]

    with pipeline() as ppl:
        ppl.classifier = IntentClassifier(llm, intent_list=intent_list)
        with lazyllm.switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
            ppl.sw.case[intent_list[0], product_law_ppl]
            ppl.sw.case[intent_list[1], support_law_ppl]
    return ppl

# ############################################
# 4. 定义会话管理相关模块
# ############################################

DEFAULT_FEW_SHOTS = [
    {"role": "user", "content": "你是谁？"},
    {"role": "assistant", "content": "我是你的智能助手。"}
]


class ChatHistory(BaseModel):
    user: str
    assistant: str

def init_session(session_id, user_history: Optional[List[ChatHistory]] = None):
    globals._init_sid(session_id)

    if "global_parameters" not in globals or "history" not in globals["global_parameters"]:
        globals["global_parameters"]["history"] = []

    if not globals["global_parameters"]["history"]:
        # 初始化为 default few-shot
        globals["global_parameters"]["history"].extend(DEFAULT_FEW_SHOTS)

    if user_history:
        for h in user_history:
            globals["global_parameters"]["history"].append({"role": "user", "content": h.user})
            globals["global_parameters"]["history"].append({"role": "assistant", "content": h.assistant})

def build_full_query(user_input: str):
    """根据 globals 里的历史，生成带历史的 full query文本"""
    history = globals["global_parameters"]["history"]
    history_text = ""
    for turn in history:
        role = "用户" if turn["role"] == "user" else "助手"
        history_text += f"{role}: {turn['content']}\n"

    full_query = f"{history_text}用户: {user_input}\n助手:"
    return full_query

class EcommerceAssistant:
    def __init__(self):
        self.main_pipeline = build_ecommerce_assistant()

    def __call__(self, session_id: str, user_input: str, user_history: Optional[List[ChatHistory]] = None):
        init_session(session_id, user_history)

        full_query = build_full_query(user_input)

        # 把带历史的 query 输入主 pipeline
        response = self.main_pipeline(full_query)

        # 更新历史到 globals
        globals["global_parameters"]["history"].append({"role": "user", "content": user_input})
        globals["global_parameters"]["history"].append({"role": "assistant", "content": response})

        return response

if __name__ == "__main__":
    # upload_files()
    assistant1 = EcommerceAssistant()
    print("==================== user1：用户支持问题 ====================")
    print("用户 user1 提问：")
    print("「用户投诉某智能手表的续航没有达到宣传效果，该怎么处理」")
    print("\n助手回复：")
    print(assistant1("user1", "用户投诉某智能手表的续航没有达到宣传效果，该怎么处理"))
    print("\n" + "="*60 + "\n")

    print("====================user2：产品法务问题（带历史对话） ====================")
    print("用户 user2 的对话历史：")
    print("1. 用户: 「你好」")
    print("   助手: 「你好呀！」")
    print("2. 用户: 「我想咨询耳机宣传内容是否合规」")
    print("   助手: 「当然，请详细描述你的宣传文案。」")
    print("\n用户 user2 新提问：")
    print("「骨传导耳机不展示专利号」")
    print("\n助手回复：")
    history = [
        ChatHistory(user="你好", assistant="你好呀！"),
        ChatHistory(user="我想咨询耳机宣传内容是否合规", assistant="当然，请详细描述你的宣传文案。")
    ]
    print(assistant1("user2", "骨传导耳机不展示专利号", user_history=history))
    print("\n" + "="*60 + "\n")

    print("====================user1 ：用户支持问题跟进 ====================")
    print("用户 user1 继续提问：")
    print("「这种处理方式有什么风险吗？」")
    print("\n助手回复：")
    print(assistant1("user1", "这种处理方式有什么风险吗？"))
    
    time.sleep(60)
    

    
    