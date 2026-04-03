from datasets import load_dataset

ds = load_dataset("qwedsacf/competition_math")

# 示例：查看一条训练样本
print(ds["train"][0])
