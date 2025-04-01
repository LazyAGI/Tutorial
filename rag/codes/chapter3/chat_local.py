import lazyllm
from lazyllm import deploy

llm = lazyllm.TrainableModule('internlm2-chat-7b').\
        deploy_method((deploy.Vllm, {
            'port': 8081,
            'host': '0.0.0.0',
        })).start()
res = llm('hi')
print("大模型的输出是：", res)