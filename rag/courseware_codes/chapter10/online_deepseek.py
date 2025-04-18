import lazyllm

# chat0 = lazyllm.OnlineChatModule('deepseek-reasoner', source='deepseek')
chat1 = lazyllm.OnlineChatModule('deepseek-r1', source='qwen')
chat2 = lazyllm.OnlineChatModule('DeepSeek-R1', source='sensenova')

res1 = chat1('hi')
print("QWen: ", res1)
res2 = chat2('hi')
print("SenseNova: ", res2)
