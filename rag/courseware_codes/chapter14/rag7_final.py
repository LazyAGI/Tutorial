# flake8: noqa: E501
import lazyllm
from lazyllm import bind, _0
from utils.pdf_reader import MagicPDFReader
from utils.config import tmp_dir, gen_prompt, build_vlm_prompt, get_image_path


def build_paper_rag():
    embed_mltimodal = lazyllm.TrainableModule("colqwen2-v0.1")
    embed_text = lazyllm.TrainableModule("bge-m3")
    embeds = {'vec1': embed_text, 'vec2': embed_mltimodal}

    qapair_llm = lazyllm.LLMParser(lazyllm.OnlineChatModule(stream=False), language="zh", task_type="qa")
    qapair_img_llm = lazyllm.LLMParser(
        lazyllm.OnlineChatModule(source="sensenova", model="SenseNova-V6-Turbo"), language="zh", task_type="qa_img") 
    summary_llm = lazyllm.LLMParser(lazyllm.OnlineChatModule(stream=False), language="zh", task_type="summary") 

    documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir, embed=embeds, manager=False)
    documents.add_reader("*.pdf", MagicPDFReader)
    documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')
    documents.create_node_group(name="summary", transform=lambda d: summary_llm(d), trans_node=True)
    documents.create_node_group(name='qapair', transform=lambda d: qapair_llm(d), trans_node=True)
    documents.create_node_group(name='qapair_img', transform=lambda d: qapair_img_llm(d), trans_node=True, parent='Image')

    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.mix:
            with lazyllm.pipeline() as ppl.mix.rank:
                with lazyllm.parallel().sum as ppl.mix.rank.short:
                    ppl.mix.rank.short.retriever1 = lazyllm.Retriever(documents, group_name="summary", embed_keys=['vec1'], similarity="cosine", topk=4)
                    ppl.mix.rank.short.retriever2 = lazyllm.Retriever(documents, group_name="qapair", embed_keys=['vec1'], similarity="cosine", topk=4)
                    ppl.mix.rank.short.retriever3 = lazyllm.Retriever(documents, group_name="qapair_img", embed_keys=['vec1'], similarity="cosine", topk=4)
                ppl.mix.rank.reranker = lazyllm.Reranker("ModuleReranker", model="bge-reranker-large", topk=3) | bind(query=ppl.mix.rank.input)
            ppl.mix.retriever4 = lazyllm.Retriever(documents, group_name="block", embed_keys=['vec1'], similarity="cosine", topk=2)
            ppl.mix.retriever5 = lazyllm.Retriever(documents, group_name="Image", embed_keys=['vec2'], similarity="maxsim", topk=2)

        ppl.prompt = build_vlm_prompt | bind(_0, ppl.input)
        ppl.vlm = lazyllm.OnlineChatModule(source="sensenova", model="SenseNova-V6-Turbo").prompt(lazyllm.ChatPrompter(gen_prompt))
    return ppl

if __name__ == "__main__":
    lazyllm.WebModule(build_paper_rag(), port=range(23468, 23470), static_paths=get_image_path()).start().wait()
