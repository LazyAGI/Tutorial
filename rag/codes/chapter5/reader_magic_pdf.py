from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import DocNode, Document

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.libs import config_reader
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup
import copy
import torch


def read_config():
    config = {
                'bucket_info': {
                    'bucket-name-1': ['ak', 'sk', 'endpoint'],
                    'bucket-name-2': ['ak', 'sk', 'endpoint']
                },
                'models-dir': 'Your_Model_Path/models',
                'layoutreader-model-dir': 'Your_Model_Path/layoutreader',
                'layout-config': {
                    'model': 'layoutlmv3'
                },
                'formula-config': {
                    'mfd_model': 'yolo_v8_mfd',
                    'mfr_model': 'unimernet_small',
                    'enable': False
                },
                'table-config': {
                    'model': 'tablemaster',
                    'enable': True,
                    'max_time': 400
                },
                'config_version': '1.0.0'
            }
    config['device-mode'] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


config_reader.read_config = read_config

PARAGRAPH_SEP = "\n"


class UnionPdfReader(ReaderBase):
    def __init__(self):
        super().__init__()
        self.image_save_path = "Stored Images Path"
        self.model = None

    def _result_extract(self, content_list):
        blocks = []
        cur_title = ""
        cur_level = -1
        for content in content_list:
            block = {}
            if content["type"] == "text":
                content["type"] = content["text"].strip()
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
                block['page'] = content["page_idx"]
                del block["page_idx"]
                blocks.append(block)
            elif content["type"] == "image":
                block["type"] = content["type"]
                block["page"] = content["page_idx"]
                block["img_path"] = content["img_path"]
                block["text"] = "".join(content["img_caption"])
                if cur_title:
                    block["title"] = cur_title
                blocks.append(block)
            elif content["type"] == "table":
                block["type"] = content["type"]
                block["page"] = content["page_idx"]
                block["text"] = self._html_table_to_markdown(content["table_body"]) if "table_body" in content else ""
                if cur_title:
                    block["title"] = cur_title
                blocks.append(block)
        return blocks

    def _html_table_to_markdown(self, table):
        try:
            table = BeautifulSoup(table, 'html.parser')
            header = table.find('thead')
            if header:
                header_row = header.find('tr')
                headers = [cell.get_text(strip=True) for cell in header_row.find_all('td')]
            else:
                headers = []

            body = table.find('tbody')
            rows = []
            if body:
                for row in body.find_all('tr'):
                    cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
                    rows.append(cells)
            markdown = ''
            if headers:
                markdown += '| ' + ' | '.join(headers) + ' \n'
                markdown += '| ' + ' | '.join(['-' * len(h) for h in headers]) + ' |\n'
            for row in rows:
                markdown += '| ' + ' | '.join(row) + ' |\n'
            return markdown
        except Exception as e:
            print(f"error: {e}")
            return table

    def _pdf_parse_to_elements(self, pdf_path: Path, output_dir: str = None):
        pdf_path = pdf_path.as_posix()
        try:
            pdf_bytes = open(pdf_path, "rb").read()

            if not self.model:
                model_json = []
                jso_useful_key = {"_pdf_type": "", "model_list": model_json}
                image_writer = DiskReaderWriter(self.image_save_path)
                self.model = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            else:
                self.model.pdf_bytes = pdf_bytes

            self.model.pipe_classify()
            self.model.pipe_analyze()
            self.model.pipe_parse()

            content_list = self.model.pipe_mk_uni_format(img_parent_path=self.image_save_path, drop_mode="none")

            return self._result_extract(content_list)
        except Exception as e:
            print(f"fail to parse pdf---{e}")
            assert e

    def _load_data(self, file: Path, split_documents: Optional[bool] = True, extra_info=None, fs=None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)
        elements = self._pdf_parse_to_elements(file)
        docs = []
        if split_documents:
            for element in elements:
                metadata = extra_info or {}
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


doc = Document(dataset_path="your_doc_path")
doc.add_reader("*.pdf", UnionPdfReader)
data = doc._impl._reader.load_data(input_files=["Masked Autoencoders Are Scalable Vision Learners.pdf"])
print(f"data: {data}")
for i in range(15):
    print(f"text-{i}: {data[i].text}")
