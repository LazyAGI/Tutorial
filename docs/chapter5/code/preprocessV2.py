from typing import Dict, Any, Literal
import json
import lazyllm
from lazyllm.tools import fc_register
import json5
import re

llm = lazyllm.OnlineChatModule()

# =========================
# 数据
# =========================

input_list = [
    {'query': '今天天气怎么样', 'answer': '今天下雨', 'content': '今天是晴天呢'},
    {'query': '小明今年多少岁', 'answer': '18', 'content': '小明今年十八岁'},
    {'query': '????', 'answer': '?', 'content': 'LazyLLM是一款高性能的开源人工智能框架'}
]

# =========================
# 工具（严格：纯函数）
# =========================

@fc_register("tool")
def regenerate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    q = f"""
Generate a QA pair strictly based on the following content.

Return JSON only:
{{
  "query": "...",
  "answer": "..."
}}

Content:
{row["content"]}
"""
    resp = llm(q)
    if '</think>' in resp:
        resp = resp.split('</think>')[-1]

    match = re.search(r'\{[\s\S]*?\}', resp)
    data = json5.loads(match.group(0))
    data["content"] = row["content"]
    print(f"{row} 重写为 {data}")
    return data


@fc_register("tool")
def evaluate_row(row: Dict[str, Any]) -> bool:
    q = f"""
Answer True or False ONLY.

Is the QA pair strictly derived from the content?

Row:
{json.dumps(row, ensure_ascii=False)}
"""
    resp = llm(q).lower()
    print(f"{row} 评估结果：{resp}")
    return "true" in resp


# =========================
# Agent 控制器（外部 FSM）
# =========================

SYSTEM_PROMPT = """
You are a decision agent.

You must respond in JSON only.

Allowed outputs:

1. Request a tool:
{
  "type": "tool_call",
  "name": "evaluate_row" | "regenerate_row",
  "arguments": { ... }
}

2. Finish:
{
  "type": "final",
  "row": { ... }
}

Rules:
- No explanations
- No thoughts
- No assumptions about tool results
"""

def agent_loop(row: Dict[str, Any]) -> Dict[str, Any]:
    current = row

    while True:
        prompt = SYSTEM_PROMPT + "\nCurrent row:\n" + json.dumps(current, ensure_ascii=False)
        resp = llm(prompt)

        try:
            action = json.loads(resp)
        except Exception:
            raise RuntimeError("Invalid JSON from LLM")

        if action["type"] == "tool_call":
            name = action["name"]
            args = action["arguments"]

            if name == "evaluate_row":
                ok = evaluate_row(**args)
                if ok:
                    return current
                else:
                    current = regenerate_row(row=current)

            elif name == "regenerate_row":
                current = regenerate_row(**args)

            else:
                raise RuntimeError("Unknown tool")

        elif action["type"] == "final":
            return action["row"]

        else:
            raise RuntimeError("Invalid agent output")


# =========================
# 执行
# =========================

result = agent_loop(input_list[0])
print("✅ 最终结果：")
print(json.dumps(result, ensure_ascii=False, indent=2))
