import os
import json
import matplotlib.pyplot as plt

from lazyllm.tools.rag import DocNode, DocField, DataType
from lazyllm import OnlineChatModule, pipeline, _0, fc_register, FunctionCall, bind
import lazyllm
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools import SqlManager, SqlCall, IntentClassifier
from lazyllm.tools.rag.readers import MineruPDFReader


def get_image_path():
    # return os.path.join(os.getcwd(), "images")
    return "./images"

class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(lazyllm.config['home'], 'rag_for_qa'))
        self.rag_dir = os.path.join(self.root_dir, "papers")
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus.db")
        self.image_path = get_image_path()


tmp_dir = TmpDir()

milvus_store_conf = {
    "type": "milvus",
    "kwargs": {
        'uri': tmp_dir.store_file,
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': "COSINE",
        }
    },
    'indices': {
        'smart_embedding_index': {
            'backend': 'milvus',
            'kwargs': {
                'uri': tmp_dir.store_file,
                'index_kwargs': {
                    'index_type': 'HNSW',
                    'metric_type': 'COSINE',
                }
            },
        },
    },
}

doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}

table_info = {
    "tables": [{
        "name": "papers",
        "comment": "论文数据",
        "columns": [
            {
                "name": "id",
                "data_type": "Integer",
                "comment": "序号",
                "is_primary_key": True,
            },
            {"name": "title", "data_type": "String", "comment": "标题"},
            {"name": "author", "data_type": "String", "comment": "作者"},
            {"name": "subject", "data_type": "String", "comment": "领域"},
        ],
    }]
}

@fc_register("tool")
def plot_bar_chart(subjects, values):
    """
    Plot a bar chart using Matplotlib.

    Args:
        subjects (List[str]): A list of subject names.
        values (List[float]): A list of values corresponding to each subject.
    """
    # 检查科目和数值的数量是否匹配
    if len(subjects) != len(values):
        print("科目和数值的数量不匹配！")
        return

    # 绘制柱状图
    plt.figure(figsize=(12, 6))  # 调整图表大小，使其更宽
    _ = plt.bar(subjects, values, color='skyblue', edgecolor='black')  # 设置柱子的颜色和边框颜色

    # 添加标题和标签
    plt.title('The value of each subject', fontsize=16, fontweight='bold')  # 图表标题
    plt.xlabel('Subject', fontsize=14)  # x 轴标签
    plt.ylabel('Value', fontsize=14)  # y 轴标签

    # 调整 y 轴的范围，以确保柱子能够完全显示
    plt.ylim(0, int(1.2 * max(values)))  # 设置 y 轴的范围，稍微高于最大值

    # 在每个柱子上方添加数值标签
    for i in range(len(values)):
        plt.text(i, int(1.2 * values[i]), f'{values[i]:,.2f}', ha='center', fontsize=12, fontweight='bold')

    # 显示图表
    name = "subject_bar_chart"
    img_path = os.path.join(get_image_path(), name + '.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.show()  # 显示图表，确保它在笔记本中可见
    return f"![{name}]({img_path})"

def build_sql_call(input):
    sql_manager = SqlManager("sqlite", None, None, None, None, db_name="papers.db", tables_info_dict=table_info)
    sql_llm = lazyllm.OnlineChatModule(source="sensenova", stream=False)
    sql_call = SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=True)
    sql_ret = SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=False)
    return sql_ret(input), sql_call(input)

def build_paper_assistant():
    llm = OnlineChatModule(source="sensenova", stream=False)
    intent_list = [
        "论文问答",
        "统计问答",
    ]
    tools = ["plot_bar_chart"]

    def is_sql_ret(input):
        flag = None
        try:
            ret = json.loads(input[0])
            if len(ret) == 0 or next(iter(ret[0].values())) == 0:
                flag = False
            else:
                flag = True
        except Exception as e:
            print(f"error: {e}")
            flag = False
        return flag

    def is_painting(args):
        if len(args) == 0:
            return False
        ret = json.loads(args[0])
        if len(ret) == 1:
            return False
        else:
            return True

    def concate_instruction(input):
        return ("This time it is a query intent, but SQL Call did not find the corresponding information,"
                f" so now it is a retriever intent. {input}")

    img_prompt = ("You are a professional drawing assistant. You can judge whether you need to call the drawing "
                  "tool based on the results retrieved from the database. If not, just output the input directly "
                  "without rewriting it.")

    fc = FunctionCall(llm, tools, _prompt=img_prompt)

    prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
        ' In this task, you need to provide your answer based on the given context and question.'\
        ' If an image can better present the information being expressed, please include the image reference'\
        ' in the text in Markdown format. The markdown format of the image must be as follows:'\
        ' ![image_name](file=image path)'

    documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir,
                                 embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                                 manager=False,
                                 store_conf=milvus_store_conf,
                                 doc_fields=doc_fields)

    documents.add_reader("*.pdf", MineruPDFReader(url="http://127.0.0.1:8888"))   # url 需替换为已启动的 MinerU 服务地址    

    documents.create_node_group(name="sentences", transform=lambda s: s.split("\n") if s else '')

    with lazyllm.pipeline() as chat_ppl:
        chat_ppl.retriever = lazyllm.Retriever(documents, group_name="sentences", topk=3)

        chat_ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                             model="bge-reranker-large",
                                             topk=1,
                                             output_format='content',
                                             join=True) | bind(query=chat_ppl.input)

        chat_ppl.formatter = (
            lambda nodes, query: dict(context_str=nodes, query=query)
        ) | bind(query=chat_ppl.input)

        chat_ppl.llm = lazyllm.OnlineChatModule(source="glm", stream=False).prompt(
            lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

    with pipeline() as no_sql_chat_ppl:
        no_sql_chat_ppl.concate = concate_instruction
        no_sql_chat_ppl.chat_ppl = chat_ppl

    with pipeline() as sql_painting:
        sql_painting.judge = lazyllm.ifs(is_painting, pipeline(lambda x: x[1], fc), pipeline(lambda x: x[1]))
        sql_painting.ifs = lazyllm.ifs(lambda x: isinstance(x, str), lambda x: x, lambda x: x[-1][0]["content"])

    with pipeline() as sql_ppl:
        sql_ppl.sql = build_sql_call
        sql_ppl.ifs = lazyllm.ifs(is_sql_ret, sql_painting, pipeline(lambda x: x[1], no_sql_chat_ppl))

    with pipeline() as ppl:
        ppl.classifier = IntentClassifier(llm, intent_list=intent_list)
        with lazyllm.switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
            ppl.sw.case[intent_list[0], chat_ppl]
            ppl.sw.case[intent_list[1], sql_ppl]

    return ppl

if __name__ == "__main__":
    main_ppl = build_paper_assistant()
    image_save_path = get_image_path()
    lazyllm.WebModule(main_ppl, port=23456, static_paths=image_save_path).start().wait()
