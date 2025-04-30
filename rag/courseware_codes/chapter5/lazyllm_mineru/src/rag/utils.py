
import fitz
import os
from pathlib import Path


def draw_pdf_bbox(nodes, query, **kwargs):
    """在召回PDF文档中绘制box并标注query"""
    for node in nodes:
        bbox_list = node.metadata['bbox']
        output_path = os.path.join(get_pdf_output_path(), f"query[{query}] -- {node.metadata['file_name']}")
        draw_bboxes_on_pdf(node.metadata['file_path'], bbox_list, output_path, query)
    return "\n".join([node.get_content() for node in nodes])


def draw_bboxes_on_pdf(input_pdf_path, bbox_list, output_pdf_path, query):
    # 打开PDF文档
    pdf_document = fitz.open(input_pdf_path)

    # 遍历每一页
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num) 

        # 遍历bbox列表
        for bbox in bbox_list:
            if bbox['page'] == page_num:  # 检查bbox是否属于当前页
                rect = fitz.Rect(*bbox['bbox'])  # 创建矩形框
                page.draw_rect(rect, color=(1, 0, 0), width=2)  # 绘制红色矩形框

    # 保存带有bbox的PDF文档
    pdf_document.save(output_pdf_path)
    pdf_document.close()
 

def get_project_path():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == "LazyRAG-Mineru":
            return parent
    raise FileNotFoundError("Project 'LazyRAG-Mineru' not found in the directory tree.")


def get_pdf_output_path():
    path = os.path.join(get_project_path(), "data/draw_box_pdf")
    os.makedirs(path, exist_ok=True)
    return path


def get_image_path():
    path = os.path.join(get_project_path(), "data/images")
    os.makedirs(path, exist_ok=True)
    return path