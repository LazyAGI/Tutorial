import os
import asyncio
import lazyllm
from lazyllm import pipeline, parallel, bind, Retriever
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.openai import openai_complete, openai_embed

class LightRAGRetriever:
    def __init__(self, working_dir, txt_path, mode="hybrid"):
        self.working_dir = working_dir
        self.txt_path = txt_path
        self.mode = mode
        self.api_key = "empty"
        self.rag = None 
        self.loop = asyncio.new_event_loop()
        setup_logger("lightrag", level="INFO")

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.loop.run_until_complete(self.initialize_rag())
    
    async def initialize_rag(self):
        self.rag = LightRAG(
            working_dir=self.working_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts=texts,
                    model="bge-large",
                    base_url="http://0.0.0.0:19001",
                    api_key=self.api_key,
                )
            ),
            llm_model_func=openai_complete,
            llm_model_name="qwen2",
            llm_model_kwargs={"base_url": "http://0.0.0.0:12345/v1", "api_key": self.api_key},
        )
        await self.rag.initialize_storages()
        await initialize_pipeline_status()

        with open(self.txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        await self.rag.ainsert(content)
    
    def __call__(self, query):
        return self.loop.run_until_complete(
            self.rag.aquery(
                query,
                param=QueryParam(mode=self.mode, top_k=3, only_need_context=True)
            )
        )
    
    def close(self):
        if self.rag:
            try:
                self.loop.run_until_complete(self.rag.finalize_storages())
            except Exception as e:
                print(f"关闭LightRAG时出错: {e}")
            finally:
                self.loop.close()
                self.rag = None

def main():
    kg_retriever = None
    try:
        documents = lazyllm.Document(dataset_path="./data")
        prompt = ('请你参考所给的信息给出问题的答案。')

        print("正在初始化知识图谱检索器...")
        kg_retriever = LightRAGRetriever(
            working_dir="./shuihu_kg",
            txt_path="./data/水浒传.txt"
        )
        print("知识图谱检索器初始化完成！")

        bm25_retriever = lazyllm.Retriever(
            doc=documents,
            group_name="CoarseChunk",
            similarity="bm25_chinese",
            topk=3
        )
        
        def bm25_pipeline(query):
            nodes = bm25_retriever(query)
            return "".join([node.get_content() for node in nodes])

        def context_combiner(*args):
            bm25_result = args[0]
            kg_result = args[1]   
            return (
                f"知识图谱召回结果:\n{kg_result}"
                f"BM25召回结果:\n{bm25_result}\n\n"
            )

        with lazyllm.pipeline() as ppl:
            with parallel() as ppl.multi_retrieval:
                ppl.multi_retrieval.bm25 = bm25_pipeline
                ppl.multi_retrieval.kg = kg_retriever
            ppl.context_combiner = context_combiner
            ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
            ppl.llm = (
                lazyllm.OnlineChatModule().prompt(
                    lazyllm.ChatPrompter(
                        instruction=prompt,
                        extra_keys=['context_str']
                    )
                )
            )

        lazyllm.WebModule(ppl, port=23466).start().wait()
        
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
