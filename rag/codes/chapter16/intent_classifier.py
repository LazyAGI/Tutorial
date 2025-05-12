import lazyllm
from lazyllm.tools import IntentClassifier

classifier_llm = lazyllm.OnlineChatModule()
chatflow_intent_list = ["翻译服务", "天气查询", "内容推荐", "金融行情查询"]
classifier = IntentClassifier(classifier_llm, intent_list=chatflow_intent_list)

while True:
    query = input("输入您的问题：\n")
    print(f"\n识别到的意图是：\n {classifier(query)}\n" + "-"*40)
    
