from lazyllm import Retriever
from lazyllm import OnlineChatModule, Document, LLMParser
import os

# 请您在运行脚本前将您要使用的在线模型 Api-key 抛出为环境变量，或更改为本地模型
llm = OnlineChatModule(source='qwen', model='qwen-turbo')

# LLMParser 是 LazyLLM 内置的基于 LLM 进行节点组构造的类，支持 summary，keywords和qa三种
summary_llm = LLMParser(llm, language="zh", task_type="summary") # 摘要提取LLM
keyword_llm = LLMParser(llm, language="zh", task_type="keywords") # 关键词提取LLM
qapair_llm = LLMParser(llm, language="zh", task_type="qa") # 问答对提取LLM

docs = Document(os.path.join(os.getcwd(), 'test_parse'))

# 以换行符为分割符，将所有文档都拆成了一个个的段落块，每个块是1个Node，这些Node构成了名为"block"的NodeGroup
docs.create_node_group(name='block', transform=lambda d: d.split('\n'))

# 使用一个可以提取问答对的大模型把每个文档的摘要作为一个名为"qapair"的NodeGroup，内容是针对文档的问题和答案对
docs.create_node_group(name='qapair', transform=lambda d: qapair_llm(d), trans_node=True)

# 使用一个可以提取摘要的大模型把每个文档的摘要作为一个名为"doc-summary"的NodeGroup，内容是整个文档的摘要
docs.create_node_group(name='doc-summary', transform=lambda d: summary_llm(d), trans_node=True)

# 在"block"的基础上，通过关键词抽取大模型从每个段落都抽取出一些关键词，每个段落的关键词是一个个的Node，共同组成了"keyword"这个NodeGroup
docs.create_node_group(name='keyword', transform=lambda b: keyword_llm(b), parent='block', trans_node=True)

# 在"block"的基础上进一步转换，使用中文句号作为分割符而得到一个个句子，每个句子都是一个Node，共同构成了"sentence"这个NodeGroup
docs.create_node_group(name='sentence', transform=lambda d: d.split('。'), parent='block')

# 在"block"的基础上，使用可以抽取摘要的大模型对其中的每个Node做处理，从而得到的每个段落摘要的Node，组成"block-summary"
docs.create_node_group(name='block-summary', transform=lambda b: summary_llm(b), parent='block', trans_node=True)

# 您需要在此处更换不同的 group_name 检查对应节点组的效果
groupname_list = ['CoarseChunk', 'block', 'qapair', 'doc-summary', 'keyword', 'sentence', 'block-summary']
for groupname in groupname_list:
    retriever = Retriever(docs, group_name=groupname, similarity="bm25_chinese", topk=1)
    node = retriever("国际象棋超级大赛的概况")
    print(f"{groupname}节点组：\n", node[0].get_content())
    print('\n'+'='*100+'\n')
    input("按回车键继续下一个分组...")
