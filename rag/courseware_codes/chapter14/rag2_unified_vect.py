# flake8: noqa: E501
import lazyllm
from lazyllm import bind, _0
from utils.pdf_reader import MagicPDFReader
from utils.config import tmp_dir, gen_prompt, build_vlm_prompt, get_image_path


def build_paper_rag():
    embed_multimodal = lazyllm.TrainableModule("colqwen2-v0.1")
    embed_text = lazyllm.OnlineEmbeddingModule(
        source='qwen', embed_model_name='text-embedding-v1')
    embeds = {'vec1': embed_text, 'vec2': embed_multimodal}
    documents = lazyllm.Document(
        dataset_path=tmp_dir.rag_dir, embed=embeds, manager=False)
    documents.add_reader("*.pdf", MagicPDFReader)
    documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.prl:
            ppl.prl.retriever1 = lazyllm.Retriever(documents, group_name="block", embed_keys=['vec1'], similarity="cosine", topk=1)
            ppl.prl.retriever2 = lazyllm.Retriever(documents, group_name="Image", embed_keys=['vec2'], similarity="maxsim", topk=2)

        ppl.prompt = build_vlm_prompt | bind(_0, ppl.input)
        ppl.vlm = lazyllm.OnlineChatModule(source="sensenova", model="SenseNova-V6-Turbo").prompt(lazyllm.ChatPrompter(gen_prompt))
    return ppl

if __name__ == "__main__":
    lazyllm.WebModule(build_paper_rag(), port=range(23468, 23470), static_paths=get_image_path()).start().wait()
