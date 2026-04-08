import json
import lazyllm
from lazyllm.tools import fc_register, ReactAgent
import re
import json5

llm = lazyllm.OnlineChatModule()

input_list = [
    {'query': '今天天气怎么样', 'answer': '今天下雨', 'content': '今天是晴天呢'},
    {'query': '小明今年多少岁', 'answer': '18', 'content': '小明今年十八岁'},
    {
        'query': '????',
        'answer': '?',
        'content': 'LazyLLM是一款高性能的开源人工智能框架'
    }
]

saved_rows = []


@fc_register('tool')
def regenerate_row(row: dict) -> dict:
    """
    Generate QA pairs based on the input content.

    Args:
        row (dict): A qa pair and the content.

    Returns:
        dict: a QA pair
        {
            query : str,
            answer : str,
            content : str
        }

    """
    q = f"""
    Generate a QA pair based on the input.
    The output format must be:
    ```
    json
    {{
    'query': 'generated query',
    'answer': 'generated answer'
    }}
    ```

   Input: {row['content']}
    """
    response = llm(q)
    if '</think>' in response:
        response = response.split('</think>')[-1]

    match = re.search(r'(\{[\s\S]*?\})', response)
    json_str = match.group(1) if match else response
    judge = json5.loads(json_str)
    judge['content'] = row['content']
    # print()

    saved_rows.append(judge)
    print(f'正在重写：{row}\n重写结果：{judge}')
    return judge


@fc_register('tool')
def evaluate_row(row: dict) -> bool:
    """
    Check whether the QA pair is generated based on the 'content'.

    Args:
        row (dict): A qa pair and the content.

    Returns:
        bool: True or False

    """
    q = f"""
    Check whether the QA pair is generated based on the 'content'.
    The output must be either True or False

    Target row: {row}
    """
    response = llm(q).lower()
    result = False
    if 'true' in response:
        result = True
        saved_rows.append(row)
    print(f'正在打分：{row}\n打分结果：{result}')
    return result


# =========================
# Prompt
# =========================

for row in input_list:
    row = input_list[0]
# agent_prompt = """
# 你是一个严格的数据质量校验 Agent。

# ⚠️ 工具调用格式必须【完全严格】：
# Action: 工具名
# Action Input: <单行 JSON，不能换行，不能缩进>


# 你必须立刻调用 evaluate_row。

# 格式必须完全如下（一行）：
# Action: evaluate_row
# Action Input: {"row":{"query":"...","answer":"...","content":"..."}}


# 示例（正确）：
# Action: evaluate_row
# Action Input: {"row":{"query":"Q","answer":"A","content":"C"}}

# 示例（错误）：
# Action Input:
# {
#   "row": {...}
# }

# 执行逻辑：
# 1. evaluate_row
# 2. False → regenerate_row
# 3. 直到 True
# """


agent_prompt = """
你必须立刻调用 evaluate_row。

格式必须完全如下（一行）：
Action: evaluate_row
Action Input: {"row":{"query":"...","answer":"...","content":"..."}}
"""


tools = ['regenerate_row', 'evaluate_row']
agent = ReactAgent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)


query = f"""
当前数据（必须原样使用，不要格式化）：
{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}
"""

result = agent(query)
print(result)

# print(saved_rows)
