import lazyllm
from lazyllm import bind
from lazyllm.tools import IntentClassifier

template = "请用下面的文段的原文来回答问题\n\n### 已知文段：{context}\n\n### 问题：{question}\n"
base_model = 'path/to/internlm2-chat-7b-chinese-math2'
base_llm = lazyllm.TrainableModule(base_model)

# 文档加载
documents = lazyllm.Document(dataset_path="path/to/cmrc2018/data_kb")

with lazyllm.pipeline() as ppl:
    # 检索组件定义
    ppl.retriever = lazyllm.Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3)
    ppl.formatter = (lambda nodes, query: template.format(
        context="".join([node.get_content() for node in nodes]), question=query)) | bind(query=ppl.input)
    # 生成组件定义
    ppl.llm = base_llm

with IntentClassifier(lazyllm.OnlineChatModule()) as ic:
    ic.case['Math', base_llm]
    ic.case['Default', ppl]

lazyllm.WebModule(ic, port=23496).start().wait()
