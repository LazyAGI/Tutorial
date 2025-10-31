import os
import json
from lazyllm.tools.rag.doc_node import ImageDocNode, QADocNode

import lazyllm
from lazyllm import bind
from lazyllm.tools.rag import DocNode, DocField, DataType
from lazyllm.tools.rag.readers import MineruPDFReader

"""
vlm_prompt = "你是一个图像内容理解专家，基于给定的图像对其进行描述。注意：" \
"1.你的输出有且仅有图像的描述信息，不要添加任何其他内容" \
"2.你只能描述你确定的图像中存在的内容，不要做任何想象、发挥" \
"3.尽可能对图像中的内容进行较为详细的描述"
"""

vlm_prompt_en = """You are a professional image analysis assistant. Strictly adhere to the following requirements:

1. Output Specifications
- Provide only objective descriptions of image content
- Absolutely no explanatory, summary, or speculative content
- Descriptions must be based solely on clearly visible elements
- No imagination or supplementation of any kind is permitted
- Use concise declarative sentences only
- No metaphors, exaggerations, or other rhetorical devices

2. Description Requirements
- Describe in order of visual salience (main subject → background → details)
- Include these elements when present:
  * Primary objects: quantity, position, morphological characteristics
  * Scene context: environment type, spatial relationships
  * Notable details: text content, distinctive patterns, anomalous features
  * Color composition: dominant colors, contrasting colors

3. Prohibited Actions
- No subjective judgments (e.g., "beautiful", "important")
- No extrapolation beyond the image (e.g., time, intentions)
- No meta-descriptive terms (e.g., "contains", "shows")
- No assumptions about unclear elements

Begin analysis now:"""

vlm_prompt_zh = """你是一个专业的图像内容分析助手，请严格按照以下要求执行任务：

1. 输出规范
- 仅输出图像内容的客观描述，禁止添加任何解释性、总结性或推测性内容
- 描述必须基于图像中明确可见的内容，禁止任何形式的想象或补充
- 采用简洁的陈述句式，不使用比喻、夸张等修辞手法

2. 描述要求
- 按视觉显著性顺序描述（主体→背景→细节）
- 包含以下要素（如存在）：
  * 主要对象：数量、位置、形态特征
  * 场景环境：场景类型、空间关系
  * 显著细节：文字内容、特殊图案、异常特征
  * 色彩构成：主色调、对比色

3. 禁止行为
- 禁止任何主观判断（如"美丽""重要"等）
- 禁止推测图像外的信息（如时间、意图等）
- 禁止使用"包含""显示"等元描述词汇

请现在开始分析："""

def formatted_query(img_path: str):
    query = {
        "query": vlm_prompt_zh,
        "files": [img_path]
    }
    json_str = json.dumps(query)
    return f'<lazyllm-query>{json_str}'


def get_cache_path():
    return os.path.join(lazyllm.config['home'], 'rag_for_qa1')

def get_image_path(dir_name=None):
    return os.path.join(get_cache_path(), "images")

def func(x):
    print(">" * 50 + f"\n{x}\n")
    return x

vlm = lazyllm.TrainableModule('internvl-chat-v1-5').start()   # 初始化一个多模态大模型
def build_image_docnode(nodes):
    img_nodes = []
    for node in nodes:
        if node.metadata.get("type", None) == "image" and node.metadata.get("image_path", None):
            img_desc = vlm(formatted_query(node.image_path))                  # 利用VLM对图像内容进行解析生成图像的文本描述
            img_nodes.append(ImageDocNode(text=img_desc, image_path=node.metadata.get("image_path"), global_metadata=node.metadata)) # 构建ImageDocNode节点 
    return nodes + img_nodes

class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(lazyllm.config['home'], 'rag_for_qa1'))
        self.rag_dir = os.path.join(self.root_dir, "rag_master")
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus1.db")
        self.image_path = get_image_path()
        os.makedirs(self.image_path, exist_ok=True)
        # atexit.register(self.cleanup)

    def cleanup(self):
        if os.path.isfile(self.store_file):
            print(f"store file: {self.store_file}")
            os.remove(self.store_file)
        for filename in os.listdir(self.image_path):
            filepath = os.path.join(self.image_path, filename)
            print(f"filepath: {filepath}")
            if os.path.isfile(filepath):
                os.remove(filepath)

tmp_dir = TmpDir()

milvus_store_conf = {
    "type": "milvus",
    "kwargs": {
        'uri': tmp_dir.store_file,
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': "COSINE",
        }
    },
    'indices': {
        'smart_embedding_index': {
            'backend': 'milvus',
            'kwargs': {
                'uri': tmp_dir.store_file,
                'index_kwargs': {
                    'index_type': 'HNSW',
                    'metric_type': 'COSINE',
                }
            },
        },
    },
}

doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}

def extract_image_paths(node):
    global_metadata = node.global_metadata
    if global_metadata:
        try:
            image_path = os.path.join(get_image_path(), global_metadata['image_path'])
            return f"\nReference image path:{image_path}\n"
        except KeyError:
            return ''


def formatted_node(node_list):
    print('Func formatted_node called!')
    for node in node_list:
        if isinstance(node, QADocNode):
            node._content = node.get_content() + extract_image_paths(node)
    return node_list

if __name__ == "__main__":
    # """加入qa对"""
    prompt = (
        'You will play the role of an AI Q&A assistant and complete a dialogue task.'
        ' In this task, you need to provide your answer based on the given context'
        ' and question. When the context includes visual information that would be'
        ' better conveyed through an image, you must include the image reference in'
        ' your response using the exact Markdown format specified below:\n\n'
        'Image Inclusion Format Requirements:\n'
        '1. The context will provide image references in this format: "Reference'
        ' image path: /path/to/image_name.jpg"\n'
        '2. You must convert this to Markdown using:'
        ' ![image_description](/path/to/image_name.jpg)\n'
        '3. "image_description" should be a concise alt-text describing the image'
        ' content\n\n'
        'Example:\n'
        'Context provides: "Reference image path: /data/diagram.jpg"\n'
        'Your response should include:'
        ' ![System architecture diagram](/data/diagram.jpg)'
    )

    llm = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5-1202")
    qapair_llm = lazyllm.LLMParser(llm.start(), language="zh", task_type="qa")      # 问答对提取LLM

    documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir,
                                 embed=lazyllm.TrainableModule("bge-m3").start(),
                                 manager=False)
    documents.add_reader("*.pdf", MineruPDFReader(url="http://127.0.0.1:8888", post_func=build_image_docnode))

    documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')
    documents.create_node_group(name='qapair', transform=lambda d: qapair_llm(d), trans_node=True, parent='Image')

    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.prl:
            ppl.prl.retriever1 = lazyllm.Retriever(documents, group_name="block", similarity="cosine", topk=3)
            ppl.prl.retriever2 = lazyllm.Retriever(documents, group_name="qapair", similarity="cosine", topk=3)

        ppl.fotmatted_node = formatted_node
        ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                        model="bge-reranker-large",
                                        topk=3,
                                        output_format='content',
                                        join=True) | bind(query=ppl.input)

        ppl.formatter = (
            lambda nodes, query: dict(context_str=nodes, query=query)
        ) | bind(query=ppl.input)

        ppl.llm = llm.share(prompt=lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

    lazyllm.WebModule(ppl, port=43466, static_paths=get_image_path()).start().wait()
