import lazyllm

# 使用在线大模型
# 请在使用前抛出对应厂商API-key为环境变量
llm = lazyllm.OnlineChatModule(source='qwen', model='qwen-turbo')

# 文档加载
documents = lazyllm.Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb")
documents.create_node_group(name='sentences', transform=lambda b: b.split('。'))
documents.create_node_group(name='block', transform=lambda b: b.split('\n'))

# 检索组件定义
retriever1 = lazyllm.Retriever(doc=documents, group_name="block", similarity="bm25_chinese", topk=3)
retriever2 = lazyllm.Retriever(doc=documents, group_name="sentences", similarity="bm25_chinese", topk=3)

# prompt 设计
prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n '
robot = llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# 推理
query = "亚硫酸盐有什么作用？"
# 将Retriever组件召回的节点全部存储到列表doc_node_list中
doc_node_list1 = retriever1(query=query)
doc_node_list2 = retriever2(query=query)

# 将两个检索器的召回结果合并到一个列表
doc_node_list = doc_node_list1 + doc_node_list2

# 将query和召回节点中的内容组成dict，作为大模型的输入
res = robot({"query": query, "context_str": "".join([node.get_content() for node in doc_node_list])})

print("系统答案：", res)
