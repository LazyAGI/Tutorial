# flake8: noqa: E501

import os
import torch
import copy

import lazyllm
from lazyllm import bind, _0
from lazyllm.tools.rag import DocField, DataType
from lazyllm.tools.rag.doc_node import ImageDocNode
from lazyllm.components.formatter import encode_query_with_filepaths

from utils.pdf_reader import MagicPDFReader

## Prompt:
gen_prompt = (
    "## 角色说明\n"
    "你是一个多模态图文问答助手，需要综合用户提问、参考文段和图像信息进行回答。\n\n"
    
    "## 输入要素\n"
    "1. 用户提问（必选）\n"
    "2. 参考文段（可选）\n"
    "3. 图像路径（可选，可能包含多个）\n\n"
    
    "## 输出要求\n"
    "1. 必须检查所有给定的图片路径，并按以下规则处理：\n"
    "   - 每个图片路径必须转换为Markdown格式：`![描述](路径)`\n"
    "   - 若存在多个图片，需在回答中合理分布插入\n"
    "   - 图片描述应简明扼要且与上下文相关\n"
    "2. 未提供图片时不得虚构图片引用\n"
    "3. 回答需有机融合：\n"
    "   - 用户提问的核心需求\n"
    "   - 参考文段的关键信息（如有）\n"
    "   - 图片视觉内容（如有）\n\n"
    
    "## 示范案例\n"
    "【用户提问】\n"
    "鸡要怎么做好吃？\n\n"
    "【参考文档】\n"
    "小炒鸡的做法要领是多放姜片，炒熟姜片后再下鸡爆炒\n\n"
    "【图片路径】\n"
    "(1) /path/to/ji1.jpg\n"
    "(2) /path/to/ji2.jpg\n\n"
    "【标准回答】（这几个字不用显示，直接回答如下）\n"
    "推荐小炒鸡做法，关键步骤如图所示：\n"
    "1. 准备阶段：![食材准备](/path/to/ji1.jpg)\n"
    "2. 烹饪要领：多放姜片爆香后下鸡块快炒\n"
    "3. 成品展示：![小炒鸡成品](/path/to/ji2.jpg)"
)

def build_vlm_prompt(node_list, query):
    imgs = []
    text = []
    for node in node_list:
        if isinstance(node, ImageDocNode):
            # imgs.append(node.image_path)
            if hasattr(node, 'similarity_score'):
                print(node.similarity_score, node.image_path)
            if hasattr(node, 'similarity_score') and node.similarity_score>0.015:
                imgs.append(node.image_path)
        else:
            info = f"\nRef{len(text)+1}: " + node.get_content()#.get_text()
            text.append(info)
    contents = '\n'.join(text)

    prompt_parts = [f"【用户提问】\n{query}\n"]
    
    # Ref:
    if contents and contents.strip():
        prompt_parts.append(f"【参考文档】\n{contents}\n")
    
    # Image Path:
    if imgs:
        img_lines = []
        for i, img_path in enumerate(imgs, 1):
            if img_path.strip():  # 过滤空路径
                img_lines.append(f"({i}) {img_path.strip()}")
        
        if img_lines:
            prompt_parts.append("【图片路径】\n" + "\n".join(img_lines) + "\n")
    
    # Merge All:
    concate_cont = "\n".join(prompt_parts)
    print(">>>>>>>>>>> Gen-Model Inputs: \n", concate_cont)
    return encode_query_with_filepaths(concate_cont, imgs)

## Path Manager:
def get_image_path(dir_name=None):
    default_path = os.path.join(os.getcwd(), 'images', 'pdf_reader')
    os.makedirs(default_path, exist_ok=True)
    return default_path

class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(lazyllm.config['home'], 'rag_for_qa'))
        self.rag_dir = os.path.join(self.root_dir, "rag_master", "pdfs")
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus.db")
        self.image_path = get_image_path()
        os.makedirs(self.image_path, exist_ok=True)

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

## Similarity Function
@lazyllm.tools.rag.register_similarity(mode='embedding', batch=True)
def maxsim(query, nodes, **kwargs):
    batch_size = 128
    scores_list = []
    query = [torch.Tensor(query) for i in range(len(nodes))]
    nodes_embed = [torch.Tensor(node) for node in nodes]
    for i in range(0, len(query), batch_size):
        scores_batch = []
        query_batch = torch.nn.utils.rnn.pad_sequence(query[i:i + batch_size], batch_first=True, padding_value=0)
        for j in range(0, len(nodes_embed), batch_size):
            nodes_batch = torch.nn.utils.rnn.pad_sequence(
                nodes_embed[j:j + batch_size],
                batch_first=True,
                padding_value=0
            )
            scores_batch.append(torch.einsum("bnd,csd->bcns", query_batch, nodes_batch).max(dim=3)[0].sum(dim=2))
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)
    scores = scores_list[0][0].tolist()
    return scores

## Show func(for debug)
def func(x):
    print(">" * 50 + f"\n{x}\n")
    return x


## App build:
def build_paper_rag():
    embed_mltimodal = lazyllm.TrainableModule("siglip")  # ("colqwen2-v0.1")
    embed_text = lazyllm.TrainableModule("bge-m3")
    embeds = {'vec1': embed_text, 'vec2': embed_mltimodal}

    qapair_llm = lazyllm.LLMParser(lazyllm.OnlineChatModule(stream=False), language="zh", task_type="qa")
    qapair_img_llm = lazyllm.LLMParser(
        lazyllm.OnlineChatModule(source="sensenova", model="SenseNova-V6-Turbo"), language="zh", task_type="qa_img") 
    summary_llm = lazyllm.LLMParser(lazyllm.OnlineChatModule(stream=False), language="zh", task_type="summary") 

    documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir, embed=embeds, manager=False)
    documents.add_reader("*.pdf", MagicPDFReader(get_image_path()))
    documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')
    documents.create_node_group(name="summary", transform=lambda d: summary_llm(d), trans_node=True)
    documents.create_node_group(name='qapair', transform=lambda d: qapair_llm(d), trans_node=True)
    documents.create_node_group(name='qapair_img', transform=lambda d: qapair_img_llm(d), trans_node=True, parent='Image')

    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.mix:
            with lazyllm.pipeline() as ppl.mix.rank:
                with lazyllm.parallel().sum as ppl.mix.rank.short:
                    ppl.mix.rank.short.retriever1 = lazyllm.Retriever(documents, group_name="summary", embed_keys=['vec1'], similarity="cosine", topk=4)
                    ppl.mix.rank.short.retriever2 = lazyllm.Retriever(documents, group_name="qapair", embed_keys=['vec1'], similarity="cosine", topk=4)
                    ppl.mix.rank.short.retriever3 = lazyllm.Retriever(documents, group_name="qapair_img", embed_keys=['vec1'], similarity="cosine", topk=4)
                ppl.mix.rank.reranker = lazyllm.Reranker("ModuleReranker", model="bge-reranker-large", topk=3) | bind(query=ppl.mix.rank.input)
            ppl.mix.retriever4 = lazyllm.Retriever(documents, group_name="block", embed_keys=['vec1'], similarity="cosine", topk=2)
            ppl.mix.retriever5 = lazyllm.Retriever(documents, group_name="Image", embed_keys=['vec2'], similarity="cosine", topk=2)

        ppl.prompt = build_vlm_prompt | bind(_0, ppl.input)
        ppl.vlm = lazyllm.OnlineChatModule(source="sensenova", model="SenseNova-V6-Turbo").prompt(lazyllm.ChatPrompter(gen_prompt))
    return ppl

if __name__ == "__main__":
    lazyllm.WebModule(build_paper_rag(), port=range(23468, 23470), static_paths=get_image_path()).start().wait()
