import lazyllm

prompter = lazyllm.AlpacaPrompter('请根据文段内容和问题来回答。\n### 文段内容：{content}\n### 问题：{input}\n')
content = prompter.generate_prompt({'input':'这是啥','content':'111'})
print(repr(content))

prompter = lazyllm.ChatPrompter('请根据文段内容和问题来回答。\n### 文段内容：{content}\n### 问题：{input}\n')
content = prompter.generate_prompt({'input':'这是啥','content':'111'})
print(repr(content))