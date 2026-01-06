from typing import Literal
import json
import lazyllm
from lazyllm.tools import fc_register, ReactAgent, ReWOOAgent
import re
import json5

llm = lazyllm.OnlineChatModule()

input_list = [
    {'query':'今天天气怎么样', 'answer':'今天下雨', 'content': '今天是晴天呢'},
    {'query':'小明今年多少岁', 'answer':'18', 'content': '小明今年十八岁'},
    {'query':'????', 'answer':'?', 'content': 'LazyLLM是一款高性能的开源人工智能框架'}
]

saved_rows = []
@fc_register("tool")
def regenerate_row(row: str) -> dict:
    '''
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

    '''
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
    print(f"正在重写：{row}\n重写结果：{judge}")
    return judge


@fc_register("tool")
def evaluate_row(row: dict) -> bool:
    '''
    Check whether the QA pair is generated based on the 'content'.
    
    Args:
        row (dict): A qa pair and the content.
        
    Returns:
        bool: True or False

    '''
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
    print(f"正在打分：{row}\n打分结果：{result}")
    return result




tools = ["regenerate_row", "evaluate_row"]
agent = ReWOOAgent(
    llm=llm,
    tools=tools,
)

# =========================
# Prompt
# =========================

# for row in input_list:

row = input_list[0]
agent_prompt = f"""
你是一个严格的数据质量校验 Agent。你的目标是确保 QA 对的内容与原始 Content 是一致的。

你可以使用的工具：
- evaluate_row(row: dict) -> bool
- regenerate_row(row: dict) -> dict

### 核心执行逻辑：
1. 调用 evaluate_row 检查当前的 QA 样本。
2. 如果 第一步结果是False： 调用 regenerate_row 生成新的 QA。
3. 对 regenerate_row 的结果再次进行校验，直到通过。

### ⚠️ 工具调用【强制格式规范】（必须严格遵守）：

当你调用工具时，**必须严格按照以下格式输出**：

Action: <工具名>
Action Input:
<一个严格合法的 JSON，对应函数参数，不允许有任何多余字符>

### 输出格式规范：
Thought: 你的逻辑思考
Action: 工具名
Action Input: 工具参数
Observation: 工具返回的结果（由系统提供）
... (重复上述步骤)
Final Answer: 最终通过校验的 QA 结果（JSON 格式）

---
### 当前待处理数据：
{json.dumps(row, ensure_ascii=False)}

请开始你的推理。
"""

result = agent(agent_prompt)
print(result)

# print(saved_rows)