import lazyllm

llm1 = lazyllm.OnlineChatModule()
llm2 = lazyllm.OnlineChatModule().prompt("你是一只小猫，在每次回答问题之后都要加上喵喵喵")

print('普通情况下模型的输出:   ', llm1('你好'))
print('自定义Prompt后模型的输出: ', llm2('你好'))