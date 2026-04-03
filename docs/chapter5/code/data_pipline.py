from typing import Dict, Any
import json
import lazyllm
from lazyllm.tools import fc_register
import json5
import re
import time

llm = lazyllm.OnlineChatModule()

def call_llm_with_retry(prompt: str, retries: int = 5) -> str:
    last_err = None
    for _ in range(retries):
        try:
            return llm(prompt)
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise last_err


@fc_register("tool")
def generate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a question–answer (QA) pair strictly based on the given content.

    This function calls the LLM to generate a QA pair that must be directly
    and strictly derived from the input content. The model is instructed
    to return JSON only, which is then parsed and normalized.

    Args:
        row (Dict[str, Any]): A dictionary containing at least the key "content".

    Returns:
        Dict[str, Any]: A dictionary with keys:
            - query: generated question
            - answer: generated answer
            - content: original content
    """
    q = f"""
Generate a QA pair strictly based on the following content.

Return JSON only:

```json
{{
  "query": "...",
  "answer": "..."
}}
```

Content:
{row["content"]}
"""
    resp = call_llm_with_retry(q)
    if '</think>' in resp:
        resp = resp.split('</think>')[-1]
    match = re.search(r'\{[\s\S]*?\}', resp)
    data = json5.loads(match.group(0))
    data["content"] = row["content"]
    print(f"{row} 重写为 {data}")
    return data


@fc_register("tool")
def evaluate_row(row: Dict[str, Any]) -> bool:
    """
    Evaluate whether a QA pair is strictly derived from the given content.

    This function asks the LLM to act as a binary judge and return
    True or False only, indicating whether the QA pair is fully supported
    by the content without hallucination or external knowledge.

    Args:
        row (Dict[str, Any]): A dictionary containing query, answer, and content.

    Returns:
        bool: True if the QA pair is strictly derived from the content,
              False otherwise.
    """
    q = f"""
Answer True or False ONLY.

Is the QA pair strictly derived from the content?

Row:
{json.dumps(row, ensure_ascii=False)}
"""
    resp = call_llm_with_retry(q).lower()
    print(f"{row} 评估结果：{resp}")
    return "true" in resp


@fc_register("tool")
def add_cot(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a chain-of-thought (CoT) explanation to a validated QA pair.

    This function generates a concise step-by-step reasoning that explains
    how the answer can be derived strictly from the given content. The CoT
    is appended to the original row without modifying existing fields.

    Args:
        row (Dict[str, Any]): A validated QA row containing query, answer,
                              and content.

    Returns:
        Dict[str, Any]: The input row augmented with an additional key:
            - cot: chain-of-thought explanation
    用中文回答 CoT
    """
    q = f"""
Return JSON only.

Add a concise step-by-step reasoning (CoT) explaining how the answer is derived
STRICTLY from the content.

JSON format:
```json
{{
  "cot": "..."
}}
```
Row:
{json.dumps(row, ensure_ascii=False)}
"""
    resp = call_llm_with_retry(q)
    if '</think>' in resp:
        resp = resp.split('</think>')[-1]
    match = re.search(r'\{[\s\S]*?\}', resp)
    data = json5.loads(match.group(0))
    row = dict(row)
    row["cot"] = data["cot"]
    print(f"添加 CoT: {row}")
    return row


def split_chunk(paragraph: str) -> list:
    return paragraph.split('\n')


def agent_manager(paragraph: str) -> list:
    chunks = split_chunk(paragraph)
    results = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        row = {"content": chunk}
        row = generate_row(row)
        for _ in range(3):
            ok = evaluate_row(row=row)
            if ok:
                break
            row = generate_row(row)
        row = add_cot(row=row)
        results.append(row)
    return results


paragraph = """
今天天气很好，阳光明媚。
小明今年十八岁，是一名大学生。
LazyLLM是一款高性能的开源人工智能框架。
"""

manager = agent_manager(paragraph)
print("✅ 最终结果：")
print(json.dumps(manager, ensure_ascii=False, indent=2))
