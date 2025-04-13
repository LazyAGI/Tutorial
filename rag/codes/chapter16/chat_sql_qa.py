import os
import json
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
import torch
import copy
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Optional
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import magic_pdf.model as model_config
from magic_pdf.libs import config_reader
import unicodedata

from lazyllm.tools.rag import DocNode, DocField, DataType
from lazyllm import OnlineChatModule, pipeline, _0, fc_register, FunctionCall, bind
import lazyllm
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools import SqlManager, SqlCall, IntentClassifier


def read_config():
    config = {
        'bucket_info': {
            'bucket-name-1': ['ak', 'sk', 'endpoint'],
            'bucket-name-2': ['ak', 'sk', 'endpoint']
        },
        'models-dir': '/home/mnt/share_server/models/PDF-Extract-Kit-1___0/models',
        'layoutreader-model-dir': '/home/mnt/share_server/models/PDF-Extract-Kit-1___0/layoutreader',
        'layout-config': {'model': 'layoutlmv3'},
        'formula-config': {'mfd_model': 'yolo_v8_mfd', 'mfr_model': 'unimernet_small', 'enable': False},
        'table-config': {'model': 'tablemaster', 'enable': True, 'max_time': 400},
        'config_version': '1.0.0'
    }
    config['device-mode'] = "cuda" if torch.cuda.is_available() else "cpu"
    return config

config_reader.read_config = read_config

mp.set_start_method('spawn', force=True)
model_config.__use_inside_model__ = True

def get_image_path():
    # return os.path.join(os.getcwd(), "images")
    return "./images"

PARAGRAPH_SEP = "\n"

class UnionPdfReader(ReaderBase):
    def __init__(self):
        super().__init__()
        self.image_save_path = get_image_path()
        self.model = None

    def _clean_content(self, content):
        if isinstance(content, str):
            content = content.encode('utf-8', 'replace').decode('utf-8')
            return unicodedata.normalize("NFKC", content)
        if isinstance(content, list):
            return [self._clean_content(t) for t in content]
        return content

    def _result_extract(self, content_list):  # noqa: C901
        blocks = []
        cur_title = ""
        cur_level = -1
        for content in content_list:
            block = {}
            if content["type"] == "text":
                content["text"] = self._clean_content(content["text"]).strip()
                if not content["text"]:
                    continue
                if "text_level" in content:
                    if cur_title and content["text_level"] > cur_level:
                        content["title"] = cur_title
                    cur_title = content["text"]
                    cur_level = content["text_level"]
                else:
                    if cur_title:
                        content["title"] = cur_title
                block = copy.deepcopy(content)
                block["page"] = content["page_idx"]
                del block["page_idx"]
                blocks.append(block)
            elif content["type"] == "image":
                if not content["img_path"]:
                    continue
                block["type"] = content["type"]
                block["page"] = content["page_idx"]
                block["image_path"] = content["img_path"]
                block['text'] = "".join(self._clean_content(content['img_caption']))
                block['img_footnote'] = self._clean_content(content['img_footnote'])
                if not block['text']:
                    continue
                if cur_title:
                    block["title"] = cur_title
                blocks.append(block)
            elif content["type"] == "table":
                block["type"] = content["type"]
                block["page"] = content["page_idx"]
                block["text"] = self._html_table_to_markdown(self._clean_content(content["table_body"])) \
                    if "table_body" in content else ""
                if cur_title:
                    block["title"] = cur_title
                blocks.append(block)
        return blocks

    def _html_table_to_markdown(self, html_table):  # noqa: C901
        try:
            # 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(html_table.strip(), 'html.parser')
            table = soup.find('table')
            if not table:
                raise ValueError("No <table> found in the HTML.")

            # 初始化存储表格内容的矩阵
            rows = []
            max_cols = 0

            # 解析所有行
            for row in table.find_all('tr'):
                cells = []
                for cell in row.find_all(['td', 'th']):
                    rowspan = int(cell.get('rowspan', 1))  # 获取 rowspan
                    colspan = int(cell.get('colspan', 1))  # 获取 colspan
                    text = cell.get_text(strip=True)  # 获取单元格内容

                    # 填充矩阵，支持跨行或跨列的单元格
                    for _ in range(colspan):
                        cells.append({'text': text, 'rowspan': rowspan})
                rows.append(cells)
                max_cols = max(max_cols, len(cells))  # 更新列数

            # 扩展矩阵，处理 rowspan 占用的单元格
            expanded_rows = []
            rowspan_tracker = [0] * max_cols  # 追踪每列的 rowspan
            for row in rows:
                expanded_row = []
                col_idx = 0
                for cell in row:
                    # 跳过因 rowspan 导致的占位列
                    while col_idx < max_cols and rowspan_tracker[col_idx] > 0:
                        expanded_row.append(None)
                        rowspan_tracker[col_idx] -= 1
                        col_idx += 1

                    # 添加当前单元格
                    expanded_row.append(cell['text'])
                    # 更新 rowspan 追踪器
                    if cell['rowspan'] > 1:
                        rowspan_tracker[col_idx] = cell['rowspan'] - 1
                    col_idx += 1

                # 补全因 rowspan 导致的剩余占位符
                while col_idx < max_cols:
                    if rowspan_tracker[col_idx] > 0:
                        expanded_row.append(None)
                        rowspan_tracker[col_idx] -= 1
                    else:
                        expanded_row.append("")
                    col_idx += 1

                expanded_rows.append(expanded_row)

            # 将第一行视为表头
            headers = expanded_rows[0]
            body_rows = expanded_rows[1:]

            # 生成 Markdown 表格
            markdown = ''
            if headers:
                markdown += '| ' + ' | '.join(h if h else '' for h in headers) + ' |\n'
                markdown += '| ' + ' | '.join(['-' * (len(h) if h else 3) for h in headers]) + ' |\n'
            for row in body_rows:
                markdown += '| ' + ' | '.join(cell if cell else '' for cell in row) + ' |\n'

            return markdown

        except Exception as e:
            print(f"Error parsing table: {e}")
            return ''

    def _pdf_parse_to_elements(self, pdf_path: Path):
        # args
        # image_dir = str(os.path.basename(self.image_save_path))

        os.makedirs(self.image_save_path, exist_ok=True)

        image_writer = FileBasedDataWriter(self.image_save_path)

        # read bytes
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_path)  # read the pdf content

        # proc
        # Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        # inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            pipe_result = infer_result.pipe_txt_mode(image_writer)

        infer_result.get_infer_res()

        content_list_content = pipe_result.get_content_list(self.image_save_path)
        return self._result_extract(content_list_content)

    def _load_data(self, file: Path, split_documents: Optional[bool] = True, extra_info=None, fs=None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)
        elements = self._pdf_parse_to_elements(file)
        docs = []
        if split_documents:
            for element in elements:
                metadata = copy.deepcopy(extra_info) or {}
                metadata["file_name"] = file.name
                for k, v in element.items():
                    if k == "text":
                        continue
                    metadata[k] = v
                docs.append(DocNode(text=element["text"] if "text" in element else "", metadata=metadata))
        else:
            metadata = extra_info or {}
            metadata["file_name"] = file.name
            text_chunks = [el["text"] for el in elements if "text" in el]
            docs.append(DocNode(text=PARAGRAPH_SEP.join(text_chunks), metadata=metadata))
        return docs

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

    documents.add_reader("*.pdf", UnionPdfReader)

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
