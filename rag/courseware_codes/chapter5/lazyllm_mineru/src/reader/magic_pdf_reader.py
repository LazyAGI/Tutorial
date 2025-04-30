import os
import torch
import copy
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Iterable
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import magic_pdf.model as model_config
from magic_pdf.libs import config_reader
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.libs import config_reader
from magic_pdf.dict2md import ocr_mkcontent
import unicodedata
from lazyllm.tools.rag import DocNode
from lazyllm import LOG
from src.rag.utils import get_image_path


# add patchs to magic-pdf
def read_config():
    config = {
        "bucket_info": {
            "bucket-name-1": ["ak", "sk", "endpoint"],
            "bucket-name-2": ["ak", "sk", "endpoint"]
        },
        "models-dir": "/home/mnt/share_server/models/PDF-Extract-Kit-1___0/models",
        "layoutreader-model-dir": "/home/mnt/share_server/models/PDF-Extract-Kit-1___0/layoutreader",
        "layout-config": {
            "model": "doclayout_yolo"
        },
        "formula-config": {
            "mfd_model": "yolo_v8_mfd",
            "mfr_model": "unimernet_small",
            "enable": False
        },
        "table-config": {
            "model": "rapid_table",
            "enable": True,
            "max_time": 400
        },
        "config_version": "1.0.0"
    }

    config["device-mode"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


config_reader.read_config = read_config


def parse_line_spans(para_block, page_idx):
    lines_metas = []
    page = page_idx
    if 'lines' in para_block:
        for line_info in para_block['lines']:
            if not line_info['spans']:
                continue
            line_meta = copy.deepcopy(line_info['spans'][0])
            line_meta.pop('score', None)
            if_cross_page = line_meta.pop('cross_page', None)
            line_meta['page'] = page + 1 if if_cross_page == True else page
            lines_metas.append(line_meta)
    return lines_metas


def para_to_standard_format_v2(para_block, img_buket_path, page_idx, drop_reason=None):
    para_type = para_block['type']
    para_content = {}

    lines_metas = parse_line_spans(para_block, page_idx)
    if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
        para_content = {
            'type': 'text',
            'text': ocr_mkcontent.merge_para_with_text(para_block),
        }
    elif para_type == BlockType.Title:
        para_content = {
            'type': 'text',
            'text': ocr_mkcontent.merge_para_with_text(para_block),
            'text_level': 1,
        }
    elif para_type == BlockType.InterlineEquation:
        para_content = {
            'type': 'equation',
            'text': ocr_mkcontent.merge_para_with_text(para_block),
            'text_format': 'latex',
        }
    elif para_type == BlockType.Image:
        para_content = {'type': 'image', 'img_path': '', 'img_caption': [], 'img_footnote': []}
        image_lines_metas = []
        for block in para_block['blocks']:
            if block['type'] == BlockType.ImageBody:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.Image:
                            if span.get('image_path', ''):
                                para_content['img_path'] = join_path(img_buket_path, span['image_path'])
            if block['type'] == BlockType.ImageCaption:
                image_lines_metas.extend(parse_line_spans(block, page_idx))
                para_content['img_caption'].append(ocr_mkcontent.merge_para_with_text(block))
            if block['type'] == BlockType.ImageFootnote:
                image_lines_metas.extend(parse_line_spans(block, page_idx))
                para_content['img_footnote'].append(ocr_mkcontent.merge_para_with_text(block))
        para_content['lines'] = image_lines_metas
    elif para_type == BlockType.Table:
        table_lines_metas = []
        para_content = {'type': 'table', 'img_path': '', 'table_caption': [], 'table_footnote': []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TableBody:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.Table:

                            if span.get('latex', ''):
                                para_content['table_body'] = f"\n\n$\n {span['latex']}\n$\n\n"
                            elif span.get('html', ''):
                                para_content['table_body'] = f"\n\n{span['html']}\n\n"

                            if span.get('image_path', ''):
                                para_content['img_path'] = join_path(img_buket_path, span['image_path'])

            if block['type'] == BlockType.TableCaption:
                table_lines_metas.extend(parse_line_spans(block, page_idx))
                para_content['table_caption'].append(ocr_mkcontent.merge_para_with_text(block))
            if block['type'] == BlockType.TableFootnote:
                table_lines_metas.extend(parse_line_spans(block, page_idx))
                para_content['table_footnote'].append(ocr_mkcontent.merge_para_with_text(block))
        para_content['lines'] = table_lines_metas

    para_content['page_idx'] = page_idx
    para_content['bbox'] = para_block['bbox']
    if lines_metas:
        para_content['lines'] = lines_metas + para_content.pop('lines', [])

    if drop_reason is not None:
        para_content['drop_reason'] = drop_reason

    return para_content


ocr_mkcontent.para_to_standard_format_v2 = para_to_standard_format_v2


from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import DocNode

import torch.multiprocessing as mp
import torch
import copy
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Iterable
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import magic_pdf.model as model_config
from magic_pdf.libs import config_reader
import unicodedata

def read_config():
    config = {'bucket_info': {'bucket-name-1': ['ak', 'sk', 'endpoint'], 'bucket-name-2': ['ak', 'sk', 'endpoint']}, 'models-dir': '/home/mnt/share_server/models/PDF-Extract-Kit-1___0/models', 'layoutreader-model-dir': '/home/mnt/share_server/models/PDF-Extract-Kit-1___0/layoutreader', 'layout-config': {'model': 'layoutlmv3'}, 'formula-config': {'mfd_model': 'yolo_v8_mfd', 'mfr_model': 'unimernet_small', 'enable': False}, 'table-config': {'model': 'rapid_table', 'enable': True, 'max_time': 400}, 'config_version': '1.0.0'}
    config['device-mode'] = "cuda" if torch.cuda.is_available() else "cpu"
    return config

config_reader.read_config = read_config
model_config.__use_inside_model__ = True
import os
PARAGRAPH_SEP = "\n"

class MagicPDFReader(ReaderBase):
    def __init__(self):
        super().__init__()
        self.image_save_path = "parse_images"
        self.model = None
    
    def _clean_content(self, content):
        if isinstance(content, str):
            content = content.encode('utf-8', 'replace').decode('utf-8')
            return unicodedata.normalize("NFKC", content)
        if isinstance(content, list):
            return [self._clean_content(t) for t in content]
        return content
    
    def _result_extract(self, content_list):
        blocks = []
        cur_title = ""
        cur_level = -1
        for content in content_list:
            block = {}
            block["bbox"] = content.get('bbox', '')
            block["lines"] = content["lines"] if 'lines' in content else []
            for line in block['lines']:
                line['content'] = self._clean_content(line['content'])            
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
                block["text"] = self._html_table_to_markdown(self._clean_content(content["table_body"])) if "table_body" in content else ""
                if cur_title:
                    block["title"] = cur_title
                blocks.append(block)
        return blocks
    
    def _html_table_to_markdown(self, html_table):
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
