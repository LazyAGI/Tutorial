import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete, openai_embed

setup_logger("lightrag", level="INFO")
WORKING_DIR = "./shuihu_kg"
PATH_TO_TXT = "./data/水浒传.txt"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
api_key = "empty"
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
                        embedding_dim=1024,
                        max_token_size=8192,
                        func=lambda texts: openai_embed(
                            texts=texts,
                            model="bge-large",
                            base_url="http://127.0.0.1:19001", 
                            api_key=api_key,
                        )
                    ),
        llm_model_func=openai_complete,
        llm_model_name="qwen2",
        llm_model_kwargs={"base_url":"http://0.0.0.0:12345/v1", "api_key":api_key},
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def main():
    try:
        rag = await initialize_rag()
        with open(PATH_TO_TXT, "r", encoding="utf-8") as f:
            content = f.read()
        await rag.ainsert(content)
        mode = "hybrid"
        print(await rag.aquery("鲁智深打的是谁？", 
                            param=QueryParam(mode=mode, only_need_context=True), ))
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if rag:
            await rag.finalize_storages()
if __name__ == "__main__":
    asyncio.run(main())
