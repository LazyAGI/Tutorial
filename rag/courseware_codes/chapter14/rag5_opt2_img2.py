# flake8: noqa: E501
import os
import lazyllm
from lazyllm import bind, _0
from lazyllm.tools.rag.readers import MineruPDFReader
from utils.config import tmp_dir, gen_prompt, build_vlm_prompt, get_image_path
from lazyllm.tools.rag.doc_node import ImageDocNode
import json

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


def build_paper_rag():
    embed_mltimodal = lazyllm.TrainableModule("colqwen2-v0.1")
    embed_text = lazyllm.TrainableModule("bge-m3")
    embeds = {'vec1': embed_text, 'vec2': embed_mltimodal}

    qapair_llm = lazyllm.LLMParser(lazyllm.OnlineChatModule(stream=False), language="zh", task_type="qa") 
    summary_llm = lazyllm.LLMParser(lazyllm.OnlineChatModule(stream=False), language="zh", task_type="summary") 

    documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir, embed=embeds, manager=False)
    documents.add_reader("*.pdf", MineruPDFReader(url="http://127.0.0.1:8888", post_func=build_image_docnode))   # url 需替换为已启动的 MinerU 服务地址    
    documents.create_node_group(name="summary", transform=lambda d: summary_llm(d), trans_node=True)
    documents.create_node_group(name='qapair', transform=lambda d: qapair_llm(d), trans_node=True)
    documents.create_node_group(name='qapair_img', transform=lambda d: qapair_llm(d), trans_node=True, parent='ImgDesc')

    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.prl:
            ppl.prl.retriever1 = lazyllm.Retriever(documents, group_name="summary", embed_keys=['vec1'], similarity="cosine", topk=1)
            ppl.prl.retriever2 = lazyllm.Retriever(documents, group_name="Image", embed_keys=['vec2'], similarity="maxsim", topk=2)
            ppl.prl.retriever3 = lazyllm.Retriever(documents, group_name="qapair", embed_keys=['vec1'], similarity="cosine", topk=1)
            ppl.prl.retriever4 = lazyllm.Retriever(documents, group_name="qapair_img", embed_keys=['vec1'], similarity="cosine", topk=1)

        ppl.prompt = build_vlm_prompt | bind(_0, ppl.input)
        ppl.vlm = lazyllm.OnlineChatModule(source="sensenova", model="SenseNova-V6-Turbo").prompt(lazyllm.ChatPrompter(gen_prompt))
    return ppl

if __name__ == "__main__":
    lazyllm.WebModule(build_paper_rag(), port=range(23468, 23470), static_paths=get_image_path()).start().wait()
