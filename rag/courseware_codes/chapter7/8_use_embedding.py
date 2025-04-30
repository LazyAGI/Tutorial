# 将您的本地模型地址抛出为环境变量或使用绝对路径  export LAZYLLM_MODEL_PATH=/path/to/your/models
# 将您的在线模型 API key 抛出为环境变量，以 sensenova 为例
#     export LAZYLLM_SENSENOVA_API_KEY=
#     export LAZYLLM_SENSENOVA_SECRET_KEY=

import lazyllm
from lazyllm import deploy

online_embed = lazyllm.OnlineEmbeddingModule(source="qwen")
offline_embed = lazyllm.TrainableModule('bge-large-zh-v1.5').start()
offline_sparse_embed = lazyllm.TrainableModule('bge-m3').deploy_method((deploy.AutoDeploy, {'embed_type': 'sparse'})).start()
print("online embed: ", online_embed("hello world"))
print("offline embed: ", offline_embed("hello world"))
print("offline sparse embed: ",  offline_sparse_embed("hello world"))
