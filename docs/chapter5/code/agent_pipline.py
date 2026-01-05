from typing import Literal
import json
import lazyllm
from lazyllm.tools import fc_register, FunctionCall, FunctionCallAgent
from lazyllm.components.prompter import ChatPrompter
import re
import json5


llm = lazyllm.OnlineChatModule()


@fc_register("tool")
def data_synthesis(content: str) -> dict:
    '''
    Generate QA pairs based on the input content.

    Args:
        content (str): The content from the origianl document.
        
    Returns:
        dict: a QA pair
        {
            query : str,
            answer : str,
        }

    '''
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
    print("QA对生成工具被调用，生成结果为：")
    print(judge)
    return judge


@fc_register("tool")
def self_correction(qa_pair: dict, content: str) -> dict:
    '''
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

    '''
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
    print("自我修正function被调用：")
    print(judge)
    return judge



llm = lazyllm.OnlineChatModule()

tools = ["data_synthesis", "self_correction"]
fc = FunctionCall(llm, tools)
query = input("请输入原文本： \n")

agent_prompt = f"""
    你需要根据客户输入进来的需求进行数据生成和自检查。
    客户需求：{query}
    你必须严格按照以下步骤完成任务：

    Step 1:
    调用 tool: data_synthesis
    生成一个 QA 对。

    Step 2:
    无论 Step 1 的结果如何，
    必须调用 tool: self_correction
    对生成的 QA 进行自我修正。

    只有在 Step 2 完成后，任务才算结束。
"""

print("模型启动：")
# ret = fc(query)
# print(f"ret: {ret}")
agent = FunctionCallAgent(llm, tools)
ret = agent(agent_prompt)
# print(f"ret: {ret}")

# 根据以下内容生成sft微调训练内容 "LazyLLM是一款高性能的开源人工智能框架。"