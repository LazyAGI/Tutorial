import lazyllm

online_model = lazyllm.OnlineChatModule(source="sensenova", model="DeepSeek-V3")
print(online_model("你好，你是DeepSeek吗？"))