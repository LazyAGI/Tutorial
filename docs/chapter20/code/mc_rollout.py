import lazyllm

generator = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5")
verifier = lambda pred, target: pred.strip() == target.strip()

def construct_prm_data(question, ground_truth):
    steps = [
        "第一步：设变量 x 为数量", 
        "第二步：根据公式计算得出 10", 
        "第三步：得出结论答案是 10"
    ]
    
    prm_labeled_data = []
    
    for i in range(len(steps)):
        is_correct_path = verifier(steps[-1], ground_truth) 
        
        prm_labeled_data.append({
            "step": steps[i],
            "label": 1 if is_correct_path and i < 1 else 0 
        })
        
    return {"prompt": question, "process": prm_labeled_data}

sample_data = construct_prm_data("1+1+8等于几？", "10")
print(sample_data)