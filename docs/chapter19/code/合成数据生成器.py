import json
import uuid
from random import randint

def gen_simple_quadratic(n=50):
    dataset = []
    for _ in range(n):
        a, b, c = 1, randint(-10, 10), randint(-10, 10)
        problem = f"求方程 $x^2 + ({b})x + ({c}) = 0$ 的根。"

        # 构造 CoT 逻辑
        delta = b*b - 4*a*c
        cot = [
            f"1. 写出判别式：Δ = ({b})^2 - 4*{a}*{c} = {delta}。",
            "2. 根据 Δ 的符号判断根的情况。"
        ]

        if delta >= 0:
            r1 = round((-b + delta**0.5) / (2*a), 2)
            r2 = round((-b - delta**0.5) / (2*a), 2)
            answer = f"x1={r1}, x2={r2}"
            cot.append(f"3. 代入公式求得根为 {r1} 和 {r2}。")
        else:
            answer = "无实数根"
            cot.append("3. 由于 Δ < 0，该方程在实数范围内无解。")

        dataset.append({
            "id": str(uuid.uuid4()),
            "domain": "math",
            "input": problem,
            "cot": cot,
            "answer": answer,
            "verification": {"type": "numeric_check", "evidence": "auto-computed"}
        })
    return dataset

if __name__ == '__main__':
    data = gen_simple_quadratic(10)
    with open('synthetic_math_data.jsonl', 'w', encoding='utf8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")