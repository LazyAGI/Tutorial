from lazyllm import Document, Retriever
import lazyllm
from datasets import load_dataset
import os


# RAG 实战剖析

# 将LazyLLM路径加入环境变量
# import sys
# sys.path.append("path/to/LazyLLM")
# # 设置环境变量
# import os
# os.environ["LAZYLLM_SENSENOVA_API_KEY"] = ""
# os.environ["LAZYLLM_SENSENOVA_SECRET_KEY"] = ""

# 1.文档加载 📚
# RAG 文档读取
# 传入绝对路径
doc = Document("/path/to/rag_master/")
print(f"实际传入路径为：{doc.manager._dataset_path}")

# 传入相对路径
# 需配置环境变量：export LAZYLLM_DATA_PATH="your_path"
# doc = Document("/paper/")

# 2.检查组件 🕵
# 传入绝对路径
doc = Document("/path/to/rag_master/")

# 使用Retriever组件，传入文档doc，节点组名称这里采用内置切分策略"CoarseChunk"，相似度计算函数bm25_Chinese
retriever = Retriever(doc, group_name=Document.CoarseChunk, similarity="bm25_chinese", topk=3)

# 调用retriever组件，传入query
retriever_result = retriever("什么是道？")

# 打印结果，用get_content()方法打印具体的内容
print(retriever_result[0].get_content())

# 3.生成组件 🙋
api_key = ''
llm_prompt = "你是一只小猫，每次回答完问题都要加上喵喵喵"
llm = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5").prompt(llm_prompt)
print(llm("早上好！"))
# >>> 早上好！喵喵喵


# RAG 知识库构建

# 1.数据集简介
'''CMRC 2018（Chinese Machine Reading Comprehension 2018）数据集是一个中文阅读理解数据集，
用于中文机器阅读理解的跨度提取数据集，以增加该领域的语言多样性。
数据集由人类专家在维基百科段落上注释的近20,000个真实问题组成。'''
dataset = load_dataset('cmrc2018')  # 加载数据集
# dataset = load_dataset('cmrc2018', cache_dir='path/to/datasets') # 指定下载路径
print(dataset)

# 2.构建知识库
def create_KB(dataset):
    '''基于测试集中的context字段创建一个知识库，每10条数据为一个txt，最后不足10条的也为一个txt'''
    Context = []
    for i in dataset:
        Context.append(i['context'])
    Context = list(set(Context))  # 去重后获得256个语料

    # 计算需要的文件数
    chunk_size = 10
    total_files = (len(Context) + chunk_size - 1) // chunk_size  # 向上取整

    # 创建文件夹data_kb保存知识库语料
    os.makedirs("data_kb", exist_ok=True)

    # 按 10 条数据一组写入多个文件
    for i in range(total_files):
        chunk = Context[i * chunk_size: (i + 1) * chunk_size]  # 获取当前 10 条数据
        file_name = f"./data_kb/part_{i+1}.txt"  # 生成文件名
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk))  # 以换行符分隔写入文件

        # print(f"文件 {file_name} 写入完成！")  # 提示当前文件已写入

# 调用create_KB()创建知识库
create_KB(dataset['test'])
# 展示部分知识库中的内容
with open('data_kb/part_1.txt') as f:
    print(f.read())


# 实现最基础的 RAG
# 文档加载
documents = lazyllm.Document(dataset_path="/path/to/rag_master/")

# 检索组件定义
retriever = lazyllm.Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3)

# 生成组件定义
llm = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5")

# prompt 设计
prompt = '''You will act as an AI question-answering assistant and complete a dialogue task.
In this task, you need to provide your answers based on the given context and questions.'''
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# 推理
query = "什么是道？"
# 将Retriever组件召回的节点全部存储到列表doc_node_list中
doc_node_list = retriever(query=query)
# 将query和召回节点中的内容组成dict，作为大模型的输入
res = llm({"query": query, "context_str": "".join([node.get_content() for node in doc_node_list])})

print(f'With RAG Answer: {res}')

# 生成组件定义
llm_without_rag = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5")
query = "什么是道？"
res = llm_without_rag(query)
print(f'Without RAG Answer: {res}')
