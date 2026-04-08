import lazyllm
from lazyllm.tools import fc_register, ReactAgent
import re
import json5

llm = lazyllm.OnlineChatModule()


@fc_register('tool')
def data_synthesis(content: str) -> dict:
    """
    Generate QA pairs based on the input content.

    Args:
        content (str): The content from the origianl document.

    Returns:
        dict: a QA pair
        {
            query : str,
            answer : str,
        }

    """
    q = f"""
    Generate a QA pair based on the user's input.
    The output format must be:
    ```
    json
    {{
    'query': 'generated query',
    'answer': 'generated answer'
    }}
    ```

    user's input: {content}
    """
    response = llm(q)
    if '</think>' in response:
        response = response.split('</think>')[-1]

    match = re.search(r'(\{[\s\S]*?\})', response)
    json_str = match.group(1) if match else response
    judge = json5.loads(json_str)

    print()
    print('QA对生成工具被调用，生成结果为：')
    print(judge)
    return judge


@fc_register('tool')
def self_correction(qa_pair: dict, content: str) -> dict:
    """
    Check whether the QA pair is generated based on the input content.

    Args:
        qa_pair (Dict[str, str]): A qa pair generated based on the content.
        content (str): The content from the origianl document.

    Returns:
        dict: a QA pair
        {
            query : str,
            answer : str,
        }

    """
    q = f"""
    Check whether the QA pair is generated based on the input content.
    If the QA pair is valid, output the original QA pair.
    Otherwise, regenerate it.

    The output format must be:
    ```
    json
    {{
    'query': 'generated query',
    'answer': 'generated answer'
    }}
    ```
    generated_pair: {qa_pair}
    user's input: {content}
    """
    response = llm(q)
    if '</think>' in response:
        response = response.split('</think>')[-1]

    match = re.search(r'(\{[\s\S]*?\})', response)
    json_str = match.group(1) if match else response
    judge = json5.loads(json_str)

    print()
    print('自我修正function被调用：')
    if judge['query'] == qa_pair['query']:
        print('QA对未被修改')
        return qa_pair
    print(f'修改后的QA对：{judge}')
    return judge


tools = ['data_synthesis', 'self_correction']
agent = ReactAgent(
    llm=llm,
    tools=tools,
    max_retries=5
)

# =========================
# Prompt
# =========================
content = input('请输入原文本：\n')

agent_prompt = f"""
你是一个数据构建 Agent，需要完成「生成 + 自检」的数据流水线。

你的目标：
1. 根据原始文本生成一个 QA 对
2. 对生成的 QA 对进行自我修正，确保严格基于原文

规则：
- 你必须先调用 data_synthesis
- 然后必须调用 self_correction
- 最终只返回修正后的 QA 结果

原始文本：
{content}
"""

print('\n模型启动：\n')
result = agent(agent_prompt)
# print("\n最终输出：")
# print(result)

# 根据以下内容生成sft微调训练内容 "LazyLLM是一款高性能的开源人工智能框架。"
