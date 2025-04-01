import lazyllm

prompt1 = "你扮演一只小猫，在每次回答问题都要加上：喵喵喵"
prompt2 = "你扮演一只小鸡，在每次回答问题都要加上：咯咯哒"

llm = lazyllm.TrainableModule("internlm2-chat-7b")
llm1 = llm.share(prompt=prompt1)
llm2 = llm.share(prompt=prompt2)

# Deploy LLM
llm.start() 

# Show:
inputs = '你好'
print('未设置Prompt的LLM: ', llm(inputs))
print('设置Prompt1 的LLM: ', llm1(inputs))
print('设置Prompt2 的LLM: ', llm2(inputs))
