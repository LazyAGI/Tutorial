import lazyllm

from online_models import custom_embedding_model as embedding_model, rerank_model, llm

DOC_PATH = "/mnt/lustre/share_data/dist/test_docs"
milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': "dbs/test_rag.db",
        'index_kwargs': {
        'index_type': 'HNSW',
        'metric_type': 'COSINE',
        }
    }
}

docs = lazyllm.Document(dataset_path=DOC_PATH, embed=embedding_model, store_conf=milvus_store_conf)
docs.create_node_group(name='sentence', parent="MediumChunk", transform=(lambda d: d.split('。')))

prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
    根据以下资料回答问题：\
    {context_str} \n '

with lazyllm.pipeline() as ppl:
    with lazyllm.parallel().sum as ppl.prl:
        ppl.prl.r1 = lazyllm.Retriever(docs, group_name="MediumChunk", topk=6)
        ppl.prl.r2 = lazyllm.Retriever(docs, group_name="sentence", target="MediumChunk", topk=6)
    ppl.reranker = lazyllm.Reranker(name='ModuleReranker',model=rerank_model, topk=3) | lazyllm.bind(query=ppl.input)
    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="\n".join(node.get_content() for node in nodes),
            query=query)
    ) | lazyllm.bind(query=ppl.input)
    ppl.llm = llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

w = lazyllm.WebModule(ppl, port=23492, stream=True).start().wait()
