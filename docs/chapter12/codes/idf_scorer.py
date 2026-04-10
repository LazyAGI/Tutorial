from transformers import AutoTokenizer, AutoModelForCausalLM
from lazyllm.tools.data.operators.text2qa_ops import IFDScorer

# ===== 加载模型和 tokenizer =====
model_name = 'qwen2.5-0.5b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# ===== 创建 IFDScorer 实例 =====
scorer = IFDScorer(model=model, tokenizer=tokenizer, max_length=128)

# ===== 构造测试数据列表 =====
test_samples = [
    {
        'query': '请解释量子叠加原理',
        'answer': '量子叠加原理表示量子系统可以同时处于多个状态，直到被测量时才确定具体状态。'
    },
    {
        'query': '解释牛顿第一定律',
        'answer': '牛顿第一定律，也称惯性定律，表明物体在没有外力作用时保持匀速直线运动或静止状态。'
    },
    {
        'query': '什么是光合作用',
        'answer': '光合作用是植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。'
    },
    {
        'query': '描述水循环过程',
        'answer': '水循环包括蒸发、凝结、降水和地表径流等环节，将水从地表输送到大气再返回地表。'
    },
    {
        'query': '解释相对论的时间膨胀效应',
        'answer': '相对论表明，当一个物体接近光速运动时，其时间相对于静止观察者会变慢，这称为时间膨胀。'
    }
]

# ===== 批量计算 IFD =====
for i, sample in enumerate(test_samples, 1):
    result = scorer.forward(sample)
    print(f'Sample {i} result:\n', result, '\n')
