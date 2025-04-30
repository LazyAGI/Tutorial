import lazyllm
from lazyllm import TrainableModule

embedding_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

rerank_model = lazyllm.TrainableModule("bge-reranker-large").start()

llm = lazyllm.TrainableModule('internlm2-chat-20b', stream=True).start()
