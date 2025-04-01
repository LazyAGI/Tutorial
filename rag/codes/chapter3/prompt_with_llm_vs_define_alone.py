import lazyllm

passage = ('孙悟空，是在小说《西游记》当中唐僧的四个徒弟之一，排行第一，别名孙行者、孙猴子。'
           '自封美猴王、齐天大圣。因曾在天庭掌管御马监而又被称为弼马温，在取经完成后被如来佛祖授封为斗战胜佛')
query = '孙悟空有哪些名字？'

prompter1 = lazyllm.AlpacaPrompter({
    'system': '系统指令',
    'user': '用户指令。\n### 文段内容：{content}\n### 问题：{input}\n'
    })
content = prompter1.generate_prompt({'input':query,'content':passage})
print("\n独立Prompt(Alpaca):\n", repr(content))

prompter2 = lazyllm.ChatPrompter({
    'system': '系统指令',
    'user': '用户指令。\n### 文段内容：{content}\n### 问题：{input}\n'
    })
content = prompter2.generate_prompt({'input':query,'content':passage})
print("\n独立Prompt(Chat):\n", repr(content))


m1 = lazyllm.TrainableModule("internlm2-chat-7b").prompt(prompter1)
res = m1._prompt.generate_prompt({'input':query,'content':passage})
print("\n带LLM的Prompt(Alpaca):\n", repr(res))

m2 = lazyllm.TrainableModule("internlm2-chat-7b").prompt(prompter2)
res = m2._prompt.generate_prompt({'input':query,'content':passage})
print("\n带LLM的Prompt(Chat):\n", repr(res))