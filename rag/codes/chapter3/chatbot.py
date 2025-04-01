import lazyllm
llm = lazyllm.TrainableModule("internlm2-chat-7b").prompt("你扮演一只小鸡，每次回答问题都要加上咯咯哒")
webpage = lazyllm.WebModule(llm, port=23466, history=[llm], stream=True).start().wait()