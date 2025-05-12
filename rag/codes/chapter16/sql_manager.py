from lazyllm.tools import SqlManager

# 连接本地数据库
sql_manager = SqlManager("sqlite", None, None, None, None, db_name="papers.db",   tables_info_dict=table_info)
# 连接远程数据库
# sql_manager = SqlManager(type="PostgreSQL", user="", password="", host="",port="", name="",)

# 查询所有论文数量
res = sql_manager.execute_query("select count(*) as total_papser from papers;")
# >>> [{"total_papser": 100}]

# 查询带有 LLM 的论文标题
res = sql_manager.execute_query("select title from papers where title like '%LLM%'")
# >>> [{"title": "AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs"}, {"title": "Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs"}, {"title": "LLMs in Biomedicine: A study on clinical Named Entity Recognition"}, {"title": "VLLMs Provide Better Context for Emotion Understanding Through Common Sense Reasoning"}]