import threading
import contextlib
import io
import traceback

import lazyllm
from lazyllm.tools import SqlManager, SqlCall
from lazyllm import  fc_register
from lazyllm import LOG


BI_PROMPT = """
你是一位资深数据科学家，需要基于给定的问题进行必要的统计分析和绘图，来回答用户提出的统计问题。
你的工作流程如下：

1. 理解问题并进行数据查询
   - 你需要先拆解用户问题中需要查询数据库的步骤，调用对应工具得到数据。

2. 判断根据数据是否可直接得出统计问题的答案  
   - 如果数据可以直接回答统计问题，则直接输出结果，必要时补充绘制图表。
   - 如果不足以回答，则先进行必要的数据分析，再根据需要绘制图表。

3. 实现数据分析和绘图的方式是编写完整可执行的 Python 代码并调用相关工具执行并获取结果
   - 包含所有必要的 import、数据加载、分析逻辑、绘图代码、结果输出。  
   - 使用常见数据科学工具包（如 pandas、numpy、scikit-learn 等）进行数据分析。  
   - 使用可视化工具（如 matplotlib、seaborn）进行图表绘制。
   - 对所有需要查看的结果（如统计分析结果、图片路径等），需显式使用 print 函数输出。

4. 图像保存  
   - 所有生成的图像必须保存到以下路径：  
     {image_path}
   - 保存成功后，使用以下格式将图片展示在最终回答中（Answer部分）（image_name为保存的文件名，image_path为完整路径）： 
     ![image_name](image_path)

5. 错误处理  
   - 如果代码执行失败，请根据报错信息自动修改代码并重新执行，直到成功。


问题：  
{query}

"""

# 数据库配置信息，用于sql call
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


@fc_register("tool")
def run_sql_query(query: str):
    """
    Automatically generates and executes an SQL query based on a natural language request.
    Given a natural language query describing a data retrieval task, this function generates the corresponding SQL 
    statement, executes it against the database, and returns the result.

    Args:
        query (str): A natural language description of the desired database query.

    Returns:
        list[dict]: A list of records returned from the SQL query, where each record is represented as a dictionary.
    """
    sql_manager = SqlManager("sqlite", None, None, None, None, db_name="papers.db", tables_info_dict=table_info)
    sql_llm = lazyllm.OnlineChatModule(source='qwen', stream=False)
    sql_call = SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=False)
    return sql_call(query)


@fc_register("tool")
def run_code(code: str):
    """
    Run the given code in a separate thread and return the result.

    Args:
        code (str): code to run.
    
    Returns:
        dict: {
            "text": str,
            "error": str or None,
        }
    """
    LOG.info(f"got code:\n {code}")
    result = {
        "text": "",
        "error": None,
    }

    def code_thread():
        nonlocal result
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec_globals = {}
                exec(code, exec_globals)
        except Exception:
            result["error"] = traceback.format_exc()
        result["text"] = stdout.getvalue()

    thread = threading.Thread(target=code_thread)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        result["error"] = "Execution timed out."
        thread.join()  

    return result

