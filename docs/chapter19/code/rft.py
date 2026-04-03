import lazyllm
import json
import re
questions = [
    "2 + 3 * 4 等于多少？",
    "如果 A 是 B 的导师，B 获得了奖项，那么谁的学生获得了奖项？"
]

def model_infer(model, question, k=5):
    prompt = f"""
    请一步一步思考并给出答案。
    问题：{question}
    给出{k}个CoTs

    要求：
    - 给出清晰的推理步骤（CoT）
    - 最后给出明确答案

    # 输出格式示例
```json
[
  {{
    "result": "cot1 ### answer"
  }},
  {{
    "result": "cot2 ### answer"
  }}
]
```
    """

    res = model(
    prompt,
    static_params={
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 1500,
    }
)

    # 提取 JSON
    match = re.search(r'```json\s*([\s\S]*?)\s*```', res)
    json_str = match.group(1) if match else res

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("解析失败")
        return []

    cots = [item["result"] for item in data if "result" in item]

    return cots[:k]

def model_eval(model, q, CoTs):
    scores = []
    # 让模型 给cot 打分， 
    prompt = f"""
    根据问题{q}
    给下面CoTs 打分
    问题：{CoTs}
    根据cot质量评分从1-5

    # 输出格式示例
```json
[
  {{
    "score": "score1"
  }},
  {{
    "score": "score2"
  }}
]
```
    """


    res = model(
    prompt,
    static_params={
        "temperature": 0.0,
        "max_tokens": 512,
    }
    )

    match = re.search(r'```json\s*([\s\S]*?)\s*```', res)
    json_str = match.group(1) if match else res

    try:
        scores_data = json.loads(json_str)
    except json.JSONDecodeError:
        print("⚠️ eval JSON 解析失败")
        return []

    scores = [int(item["score"]) for item in scores_data if "score" in item]
    return scores



def rft(baseline, questions, eval_model):
    train_set = []

    for q in questions:
        cots = model_infer(baseline, q)
        print(f"问题：{q}输出的CoTs为：\n{cots}")
        if not cots:
            continue

        scores = model_eval(eval_model, q, cots)
        if len(scores) != len(cots):
            continue

        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        train_set.append({
            "question": q,
            "cot": cots[best_idx],
            "score": scores[best_idx]
        })

    return train_set


# =========================
# 4. 模型初始化
# =========================
baseline_model = lazyllm.OnlineChatModule(source = "sensenova",
    model="SenseChat-Vision")
evalate_model = lazyllm.OnlineChatModule(
    source = "sensenova",
    model="SenseNova-V6-5-Pro"
)

# =========================
# 5. 运行 RFT
# =========================
train_set = rft(baseline_model, questions, evalate_model)

for item in train_set:
    print("======")
    print("Q:", item["question"])
    print("Score:", item["score"])
    print("CoT:", item["cot"])



