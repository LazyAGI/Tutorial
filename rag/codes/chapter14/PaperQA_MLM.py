import torch
import lazyllm
import os
import fitz
import PIL.Image as Image
from pathlib import Path
from typing import List
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag.doc_node import ImageDocNode
import json
from lazyllm import pipeline, parallel, Retriever, bind, _0
from lazyllm.components.formatter import encode_query_with_filepaths

class Pdf2ImageReader(ReaderBase):
    def __init__(self, image_save_path="pdf_image_path"):
        super().__init__()
        self.image_save_path = image_save_path
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

    def _load_data(
            self,
            file: Path,
            extra_info=None,
    ) -> List[ImageDocNode]:
        """load data"""
        if not isinstance(file, Path): file = Path(file)
        docs = fitz.open(file)
        file_path = []
        for page_num in range(docs.page_count):
            metadata = extra_info or {}
            metadata["file_name"] = file.name
            metadata["file_split"] = page_num
            page = docs.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            save_path = f"{self.image_save_path}/{file.name[:-4]}_{page_num}.jpg"
            img.save(save_path)
            file_path.append(ImageDocNode(image_path=save_path, global_metadata=metadata))

        return file_path

@lazyllm.tools.rag.register_similarity(mode='embedding', batch=True)
def maxsim(query, nodes, **kwargs):
    batch_size = 128
    scores_list = []
    query = torch.Tensor([query for i in range(len(nodes))])
    nodes_embed = torch.Tensor(nodes)
    for i in range(0, len(query), batch_size):
        scores_batch = []
        query_batch = torch.nn.utils.rnn.pad_sequence(query[i:i + batch_size], batch_first=True, padding_value=0)
        for j in range(0, len(nodes_embed), batch_size):
            nodes_batch = torch.nn.utils.rnn.pad_sequence(
                nodes_embed[j:j + batch_size],
                batch_first=True,
                padding_value=0
            )
            scores_batch.append(torch.einsum("bnd,csd->bcns", query_batch, nodes_batch).max(dim=3)[0].sum(dim=2))
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)
    scores = scores_list[0][0].tolist()
    return scores

def format_markdown_image(text):
    """将路径转为markdown图像格式，用于WebModule显示检索到的图像"""
    json_part = text[text.index("{"):]
    data = json.loads(json_part)
    image_paths = data.get("files", [])
    return f'\n\n![]({image_paths[0]})'

image_file_path = "/home/mnt/zhaoshe/.lazyllm/rag_for_qa/pdfimages"
documents = lazyllm.Document(dataset_path="/home/mnt/zhaoshe/.lazyllm/rag_for_qa/rag_master",
                             embed=lazyllm.TrainableModule("colqwen2-v0.1"))
documents.add_reader("*.pdf", Pdf2ImageReader(image_file_path))

with pipeline() as ppl:
    ppl.retriever = Retriever(doc=documents, group_name="Image", similarity="maxsim", topk=1)
    ppl.formatter1 = lambda nodes: [node.image_path for node in nodes]
    # encode_query_with_filepaths 函数将用户输入的 query 和检索到的图像路径格式化为一个 json
    ppl.formatter2 = encode_query_with_filepaths | bind(ppl.input, _0)
    with parallel().sum as ppl.prl:
        ppl.prl.vlm = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-Vision")
        ppl.prl.post_action = format_markdown_image

lazyllm.WebModule(ppl, static_paths=image_file_path).start().wait()
