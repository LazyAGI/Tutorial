import lazyllm


def require_input(prompt_text: str) -> str:
    value = input(prompt_text).strip()
    if not value:
        raise ValueError(f'{prompt_text.strip()}不能为空，请提供有效输入。')
    return value


# 实现最基础的 RAG
# 获取用户输入
dataset_path = require_input('请输入 dataset 路径: ')
model_name = require_input('请输入模型名: ')
query = require_input('请输入 query: ')

# 文档加载
documents = lazyllm.Document(dataset_path=dataset_path)

# 检索组件定义
retriever = lazyllm.Retriever(
    doc=documents,
    group_name='CoarseChunk',
    similarity='bm25_chinese',
    topk=3,
)

# 生成组件定义
llm = lazyllm.OnlineChatModule(source='sensenova', model=model_name)

# prompt 设计
prompt = (
    'You will act as an AI question-answering assistant '
    'and complete a dialogue task. '
    'In this task, you need to provide your answers '
    'based on the given context and questions.'
)
llm.prompt(lazyllm.ChatPrompter(
    instruction=prompt,
    extra_keys=['context_str'],
))

# 推理
# 将Retriever组件召回的节点全部存储到列表doc_node_list中
doc_node_list = retriever(query=query)
# 将query和召回节点中的内容组成dict，作为大模型的输入
res = llm({
    'query': query,
    'context_str': ''.join([node.get_content() for node in doc_node_list]),
})

print(f'With RAG Answer: {res}')

# 生成组件定义
llm_without_rag = lazyllm.OnlineChatModule(
    source='sensenova',
    model=model_name,
)
res = llm_without_rag(query)
print(f'Without RAG Answer: {res}')
