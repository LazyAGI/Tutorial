# 子问题查询
import lazyllm
from lazyllm import Document, ChatPrompter, Retriever

rewrite_prompt = "你是一个查询改写写助手，将用户的查询改写的更加清晰。\
                注意，你不需要对问题进行回答，只需要对原始问题进行改写。\
                下面是一个简单的例子：\
                输入：RAG\
                输出：为我介绍下RAG。\
                    \
                用户输入为："

# prompt设计
robot_prompt = "你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答察。\
                根据以下资料回答问题：\
                {context_str} \n"

# 加载文档库，定义检索器在线大模型，
documents = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb")  # 请在 dataset_path 传入数据集绝对路径
retriever = Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3)  # 定义检索组件
llm = lazyllm.OnlineChatModule(source='qwen', model='qwen-turbo')  # 调用大模型

llm.prompt(ChatPrompter(instrusction=robot_prompt, extra_keys=['context_str']))
query = "MIT OpenCourseWare是啥？"


query_rewriter = llm.share(ChatPrompter(instruction=rewrite_prompt))
query = query_rewriter(query)
print(f"改写后的查询：\n{query}")

doc_node_list = retriever(query=query)

# 将查询和召回节点中的内容组成dict，作为大模型的输入
res = llm({"query": query, "context_str": "".join([node.get_content() for node in doc_node_list])})

print('\n回答: ', res)
