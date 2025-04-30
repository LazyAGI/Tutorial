import lazyllm

# 如果您要使用在线大模型，请将您要使用的在线模型 Api-key 抛出为环境变量
llm = lazyllm.OnlineChatModule(source="qwen", model="qwen-turbo")  # 调用大模型

# LLMParser 是 LazyLLM 内置的基于 LLM 进行节点组构造的类，支持 summary，keywords和qa三种
summary_llm = lazyllm.LLMParser(llm, language="zh", task_type="summary")  # 摘要提取LLM
keyword_llm = lazyllm.LLMParser(llm, language="zh", task_type="keywords")  # 关键词提取LLM
qapair_llm = lazyllm.LLMParser(llm, language="zh", task_type="qa")  # 问答对提取LLM

docs = lazyllm.Document("test_parse")

docs.create_node_group(name='summary', transform = summary_llm, trans_node=True, parent='CoarseChunk')
docs.create_node_group(name='keyword', transform = keyword_llm, trans_node=True, parent='CoarseChunk')
docs.create_node_group(name='qapair', transform = qapair_llm, trans_node=True, parent='CoarseChunk')

# 查看节点组内容，此处我们通过一个检索器召回一个节点并打印其中的内容，后续都通过这个方式实现
group_names = ["CoarseChunk", "summary", "keyword", "qapair"]
for group_name in group_names:
    print(f"======= 正在解析 {group_name}，首次解析耗时较长，请耐心等待 =====")
    retriever = lazyllm.Retriever(docs, group_name=group_name, similarity="bm25_chinese", topk=1)
    node = retriever("亚硫酸盐有什么作用？")
    print(node[0].get_content())
