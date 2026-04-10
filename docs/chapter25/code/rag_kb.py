# RAG 知识库构建
import os
from datasets import load_dataset

# 1.数据集简介
"""
CMRC 2018（Chinese Machine Reading Comprehension 2018）数据集是一个中文阅读理解数据集，
用于中文机器阅读理解的跨度提取数据集，以增加该领域的语言多样性。
数据集由人类专家在维基百科段落上注释的近20,000个真实问题组成。"""
dataset = load_dataset('cmrc2018')  # 加载数据集
# dataset = load_dataset('cmrc2018', cache_dir='path/to/datasets') # 指定下载路径
print(dataset)


# 2.构建知识库
def create_KB(dataset):
    """基于测试集中的context字段创建一个知识库，每10条数据为一个txt，最后不足10条的也为一个txt"""
    Context = []
    for i in dataset:
        Context.append(i['context'])
    Context = list(set(Context))

    # 计算需要的文件数
    chunk_size = 10
    total_files = (len(Context) + chunk_size - 1) // chunk_size

    # 创建文件夹data_kb保存知识库语料
    os.makedirs('data_kb', exist_ok=True)

    # 按 10 条数据一组写入多个文件
    for i in range(total_files):
        chunk = Context[i * chunk_size: (i + 1) * chunk_size]
        file_name = f'./data_kb/part_{i+1}.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(chunk))


# 调用create_KB()创建知识库
create_KB(dataset['test'])

# 展示部分知识库中的内容
with open('data_kb/part_1.txt') as f:
    print(f.read())
