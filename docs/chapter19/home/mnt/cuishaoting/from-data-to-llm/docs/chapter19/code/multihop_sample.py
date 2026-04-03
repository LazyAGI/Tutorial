TEMPLATES = [
    {
        "fact1": "{A} 是 {B} 的导师",
        "fact2": "{B} 获得了 {C} 奖",
        "question": "谁的学生获得了 {C} 奖？",
        "cot": [
            "根据事实1，{A} 是 {B} 的导师，说明 {B} 是 {A} 的学生",
            "根据事实2，{B} 获得了 {C} 奖",
            "因此，{A} 的学生获得了 {C} 奖"
        ],
        "answer": "{A} 的学生"
    }
]


import random

def build_multihop_sample():
    A = random.choice(["张三", "李四", "王五"])
    B = random.choice(["小明", "小红", "小刚"])
    C = random.choice(["国家奖", "科研奖", "优秀论文奖"])

    tpl = TEMPLATES[0]

    return {
        "facts": [
            tpl["fact1"].format(A=A, B=B),
            tpl["fact2"].format(B=B, C=C)
        ],
        "question": tpl["question"].format(C=C),
        "cot": [step.format(A=A, B=B, C=C) for step in tpl["cot"]],
        "answer": tpl["answer"].format(A=A)
    }

print(build_multihop_sample())
print(build_multihop_sample())
print(build_multihop_sample())