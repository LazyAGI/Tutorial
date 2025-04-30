# 子问题查询
import lazyllm
from lazyllm import Document, ChatPrompter, Retriever

# prompt设计
rewrite_prompt = "你是一个查询重写助手，将用户查询分解为多个角度的具体问题。\
          注意，你不需要对问题进行回答，只需要根据问题的字面意思进行子问题拆分，输出不要超过 3 条.\
          下面是一个简单的例子：\
          输入：RAG是什么？\
          输出：RAG的定义是什么？\
               RAG是什么领域内的名词？\
               RAG有什么特点？\
               \
          用户输入为："
robot_prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
                根据以下资料回答问题：\
                {context_str} \n '

# 加载文档库，定义检索器在线大模型，
documents = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb")  # 请在 dataset_path 传入数据集绝对路径
retriever = Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3)  # 定义检索组件
llm = lazyllm.OnlineChatModule(source='qwen', model='qwen-turbo')  # 调用大模型

query_rewriter = llm.share(ChatPrompter(instruction=rewrite_prompt))  # 通过 llm.share 复用大模型
robot = llm.share(ChatPrompter(instruction=robot_prompt, extra_keys=['context_str']))

# 推理
query = "MIT OpenCourseWare是啥？"
queries = query_rewriter(query)  # 执行查询重写
queries_list = queries.split('\n')
retrieved_docs = set()
for q in queries_list:  # 对每条重写后的查询进行检索
    doc_node_list = retriever(q)
    retrieved_docs.update(doc_node_list)

# 将查询和召回节点中的内容组成dict，作为大模型的输入
res = robot({"query": query, "context_str": "\n".join([node.get_content() for node in retrieved_docs])})

# 打印结果
print('\n重写的查询：', queries)
print('\n系统答案: ', res)


