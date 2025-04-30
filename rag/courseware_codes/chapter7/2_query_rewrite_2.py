import lazyllm
from lazyllm import Document, ChatPrompter, Retriever

# prompt设计
rewrite_prompt = "你是一个查询改写助手，将用户的查询改写的更加清晰。\
          注意，你不需要对问题进行回答，只需要对原问题进行改写.\
          下面是一个简单的例子：\
          输入：RAG\
          输出：为我介绍下RAG\
          用户输入为："

judge_prompt = "你是一个判别助手，用于判断某个回答能否解决对应的问题。如果回答可以解决问题，则输出True，否则输出False。 \
            注意，你的输出只能是True或者False。不要带有任何其他输出。 \
            当前回答为{context_str} \n"

robot_prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
                根据以下资料回答问题：\
                {context_str} \n '

# 加载文档库，定义检索器在线大模型，
documents = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb")
retriever = Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3)
llm = lazyllm.OnlineChatModule(source='qwen', model='qwen-turbo')

# 重写查询的LLM
rewrite_robot = llm.share(ChatPrompter(instruction=rewrite_prompt))

# 根据问题和查询结果进行回答的LLM
robot = llm.share(ChatPrompter(instruction=robot_prompt, extra_keys=['context_str']))

# 用于判断当前回复是否满足query要求的LLM
judge_robot = llm.share(ChatPrompter(instruction=judge_prompt, extra_keys=['context_str']))

# 推理
query = "MIT OpenCourseWare是啥？"

LLM_JUDGE = False
while LLM_JUDGE is not True:
    query_rewrite = rewrite_robot(query)                # 执行查询重写
    print('\n重写的查询：', query_rewrite)

    doc_node_list = retriever(query_rewrite)            # 得到重写后的查询结果
    res = robot({"query": query_rewrite, "context_str": "\n".join([node.get_content() for node in doc_node_list])})

    # 判断判断当前回复是否能满足query要求
    LLM_JUDGE = bool(judge_robot({"query": query, "context_str": res}))
    print(f"\nLLM判断结果：{LLM_JUDGE}")

# 打印结果
print('\n最终回复: ', res)
