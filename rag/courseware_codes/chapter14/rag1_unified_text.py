# flake8: noqa: E501
import lazyllm
from lazyllm import bind, _0
from utils.pdf_reader import MagicPDFReader
from utils.config import tmp_dir, gen_prompt, build_vlm_prompt, get_image_path


def build_paper_rag():
    documents = lazyllm.Document(
        dataset_path=tmp_dir.rag_dir,
        embed=lazyllm.TrainableModule("bge-m3"),
        manager=False)
    documents.add_reader("*.pdf", MagicPDFReader(image_path=get_image_path(), use_vlm=True))
    documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.prl:
            ppl.prl.retriever1 = lazyllm.Retriever(documents, group_name="block", similarity="cosine", topk=1)
            ppl.prl.retriever2 = lazyllm.Retriever(documents, lazyllm.Document.ImgDesc, similarity="cosine", topk=1)
        ppl.prompt = build_vlm_prompt | bind(_0, ppl.input)
        ppl.vlm = lazyllm.OnlineChatModule()
    return ppl

if __name__ == "__main__":
    lazyllm.WebModule(build_paper_rag(), port=range(23468, 23470), static_paths=get_image_path()).start().wait()
