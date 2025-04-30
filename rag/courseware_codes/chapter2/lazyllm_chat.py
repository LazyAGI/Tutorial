import lazyllm
lazyllm.tools.Retriever
chat = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5-1202")
lazyllm.WebModule(chat, port=range(23466, 23470)).start().wait()
