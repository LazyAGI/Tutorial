from lazyllm.tools import SqlManager, SqlCall
import lazyllm


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

sql_manager = SqlManager("sqlite", None, None, None, None, db_name="papers.db", tables_info_dict=table_info)
sql_llm = lazyllm.OnlineChatModule()
sql_call = SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=False)

while True:
    query = input("请输入您的问题：")
    print("answer：")
    print(sql_call(query))
# 查库并输出库中一共多少篇论文
# 查库并输出库中最多的三个subject是什么
# 查询数据库并返回 subject 包含Computer Vision的论文题目的结果
