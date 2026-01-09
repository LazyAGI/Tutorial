
import json
import openai
from typing import Optional, Dict, Any

# 1. 配置 Qwen API 参数 (兼容 OpenAI 格式)
# 替换为你的阿里云 DashScope API Key
client = openai.OpenAI(
    api_key="sk-your-qwen-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def generate_preference_data_qwen(
    prompt: str, 
    response_a: str, 
    response_b: str, 
    threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    使用 Qwen (LLM-as-a-Judge) 构造 Pairwise 偏好数据。
    
    参数:
        threshold: 分差阈值。如果 abs(score_a - score_b) < threshold，则丢弃该样本。
                   建议设为 0.5 或 0 以保留更多数据。
    """

    # 0. 预检查：如果两个回答完全一样，直接跳过，节省 Token 费用
    if response_a.strip() == response_b.strip():
        print("Skipping: Responses are identical.")
        return None

    # 2. 构造裁判 Prompt (优化版)
    # 增加 Chain-of-Thought (CoT) 引导：要求先根据多维度分析，强迫模型寻找差异
    system_message = "你是一位严格且专业的AI训练判官。即使两个回答质量接近，你也必须从逻辑严密性、排版清晰度、安全性等细节中找出优劣差异。"
    
    judge_prompt = f"""
    ### 任务
    请对比以下两个模型对同一指令的回复，判断哪一个更好。

    ### 用户指令
    {prompt}

    ### 待评估回答
    回答 A:
    {response_a}
    ---
    回答 B:
    {response_b}

    ### 评判标准
    1. 指令遵循：是否完全满足了用户的需求？
    2. 逻辑性：内容是否准确、条理清晰？
    3. 冗余度：在信息量相同的情况下，更简洁者更优。

    ### 输出要求
    1. 必须以 JSON 格式输出。
    2. winner 字段只能是 "A" 或 "B"。
    3. score_a 和 score_b 分数范围为 0-10 (支持一位小数)。
    4. reason 必须详细说明 A 比 B 好（或差）的具体细节。

    JSON 示例: {{"reason": "A的回答直接给出了代码，而B主要在空谈理论，因此A更好。", "score_a": 9.5, "score_b": 6.0, "winner": "A"}}
    """

    try:
        # 3. 调用 Qwen 模型
        response = client.chat.completions.create(
            model="qwen-max", 
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": judge_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # 设置为 0 保证结果的确定性
        )

        # 4. 解析结果
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)

        # 5. 构造标准格式
        is_a_winner = result["winner"] == "A"
        score_diff = abs(result["score_a"] - result["score_b"])
        
        # 逻辑过滤
        if score_diff < threshold:
            print(f"Skipping: Quality difference too small ({score_diff} < {threshold})")
            print(f"Reason: {result.get('reason', 'No reason provided')}")
            return None

        # 成功构造
        print(f"Success! Winner: {result['winner']} (Diff: {score_diff})")
        return {
            "prompt": prompt,
            "chosen": response_a if is_a_winner else response_b,
            "rejected": response_b if is_a_winner else response_a,
            "metadata": {
                "judge_model": "qwen-max",
                "score_chosen": result["score_a"] if is_a_winner else result["score_b"],
                "score_rejected": result["score_b"] if is_a_winner else result["score_a"],
                "reason": result["reason"]
            }
        }

    except json.JSONDecodeError:
        print("Error: Failed to parse JSON from model response.")
        return None
    except Exception as e:
        print(f"Error during Qwen evaluation: {e}")
        return None

# --- 使用示例 (使用了有明显差异的文本，确保能跑通) ---

prompt_text = "用Python写一个Hello World"

# 回答A：正确且简洁
resp_a = """
```python
print("Hello, World!")
"""

#回答B：啰嗦且不仅是代码
resp_b = "你好，这很简单。你可以用print函数。比如 print('Hello World')。希望能帮到你。"

#设置阈值为 0.5，更容易通过过滤
data = generate_preference_data_qwen(prompt_text, resp_a, resp_b, threshold=0.5)

if data: print("\n--- Final Data Schema ---") 
print(json.dumps(data, indent=2, ensure_ascii=False))