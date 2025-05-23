from lazyllm.tools import SqlManager


table_info = {
    "tables": [{
        "name": "papers",
        "comment": "论文数据",
        "columns": [
            {
                "name": "id",
                "data_type": "Integer",
                "comment": "序号",
                "is_primary_key": True,
            },
            {"name": "title", "data_type": "String", "comment": "标题"},
            {"name": "author", "data_type": "String", "comment": "作者"},
            {"name": "subject", "data_type": "String", "comment": "领域"},
        ],
    }]
}

# 连接本地数据库
sql_manager = SqlManager("sqlite", None, None, None, None, db_name="papers.db", tables_info_dict=table_info)
# 连接远程数据库
# sql_manager = SqlManager(type="PostgreSQL", user="", password="", host="",port="", name="",)

# 查询所有论文数量
res = sql_manager.execute_query("select count(*) as total_papser from papers;")
# >>> [{"total_papser": 100}]

# 查询带有 LLM 的论文标题
res = sql_manager.execute_query("select title from papers where title like '%LLM%'")
# >>> [{"title": "AmpleGCG: Learning a Universal and Transferable Generative Model of ..."}]
