import time
from lazyllm import TrainableModule, deploy

start_time = time.time()
llm = TrainableModule('Qwen2-72B-Instruct').deploy_method(
        (deploy.Vllm, {
            'tensor-parallel-size': 4,
        })).start()
end_time = time.time()
print("原始模型加载耗时：", end_time-start_time)

start_time = time.time()
llm_awq = TrainableModule('Qwen2-72B-Instruct-AWQ').deploy_method(
        (deploy.Vllm, {
            'tensor-parallel-size': 4,
        })).start()
end_time = time.time()
print("AWQ量化模型加载耗时：", end_time-start_time)

query = "生成一份1000字的人工智能发展相关报告"

start_time = time.time()
llm(query)
end_time = time.time()
print("原始模型耗时：", end_time-start_time)

start_time = time.time()
llm_awq(query)
end_time = time.time()
print("AWQ量化模型耗时：", end_time-start_time)
