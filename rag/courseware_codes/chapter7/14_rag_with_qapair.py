import lazyllm
from lazyllm import bind
import os

rewriter_prompt = "你是一个查询重写助手，负责给用户查询进行模板切换。\
          注意，你不需要进行回答，只需要对问题进行重写，使更容易进行检索\
          下面是一个简单的例子：\
          输入：RAG是啥？\
          输出：RAG的定义是什么？"
rag_prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
    ' In this task, you need to provide your answer based on the given context and question.'

# 定义嵌入模型和重排序模型
# online_embedding = lazyllm.OnlineEmbeddingModule()
embedding_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 如果您要使用在线重排模型
# 目前LazyLLM仅支持 qwen和glm 在线重排模型，请指定相应的 API key。
# online_rerank = lazyllm.OnlineEmbeddingModule(type="rerank")
# 本地重排序模型
offline_rerank = lazyllm.TrainableModule('bge-reranker-large').start()

llm = lazyllm.TrainableModule('internlm2-chat-20b').deploy_method(lazyllm.deploy.Vllm).start()

qa_parser = lazyllm.LLMParser(llm, language="zh", task_type="qa")

docs = lazyllm.Document(os.path.join(os.getcwd(), 'test'), embed=embedding_model)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))
docs.create_node_group(name='qapair', transform=qa_parser)


def retrieve_and_rerank():
    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.prl:
            # CoarseChunk是LazyLLM默认提供的大小为1024的分块名
            ppl.prl.retriever1 = lazyllm.Retriever(doc=docs, group_name="CoarseChunk", similarity="cosine", topk=3)
            ppl.prl.retriever2 = lazyllm.Retriever(doc=docs, group_name="block", similarity="bm25_chinese", topk=3)
        ppl.reranker = lazyllm.Reranker("ModuleReranker",
                                         model=offline_rerank,
                                         topk=3) | bind(query=ppl.input)
    return ppl


with lazyllm.pipeline() as ppl:
    # llm.share 表示复用一个大模型，如果这里设置为promptrag_prompt则会覆盖rewrite_prompt
    ppl.query_rewriter = llm.share(lazyllm.ChatPrompter(instruction=rewriter_prompt))
    with lazyllm.parallel().sum as ppl.prl:
        ppl.prl.retrieve_rerank = retrieve_and_rerank()
        ppl.prl.qa_retrieve = lazyllm.Retriever(doc=docs, group_name="qapair", similarity="cosine", topk=3)
    ppl.formatter = (
          lambda nodes, query: dict(
              context_str='\n'.join([node.get_content() for node in nodes]),
              query=query)
        ) | bind(query=ppl.input)
    ppl.llm = llm.share(lazyllm.ChatPrompter(instruction=rag_prompt, extra_keys=['context_str']))

lazyllm.WebModule(ppl, port=23491).start().wait()
