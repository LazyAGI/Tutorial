"""
CMRC2018 Evaluation Script with Model Fine-tuning and Inference Capabilities
"""

import os
import json
import time
import datasets
import argparse
import numpy as np
from tqdm import tqdm

import lazyllm
from lazyllm import finetune, deploy, launchers


# Template for constructing QA prompts
template = "请用下面的文段的原文来回答问题\n\n### 已知文段：{context}\n\n### 问题：{question}\n"

def load_data(data_path):
    """Load JSON data from specified file path"""
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def save_res(data, file_path):
    """Save data to JSON file with proper formatting"""
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def build_data_path(file_name):
    """Construct data storage path and ensure directory exists"""
    data_root = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    save_path = os.path.join(data_root, file_name)
    return save_path

def build_eval_data(data):
    """Extract necessary fields for evaluation dataset"""
    extracted_data = []
    for item in data:
        extracted_item = {
            "context": item["context"],
            "question": item["question"],
            "answers": item["answers"]["text"][0]
        }
        extracted_data.append(extracted_item)
    return extracted_data

def build_train_data(data):
    """Format training data using predefined template"""
    extracted_data = []
    for item in data:
        extracted_item = {
            "instruction": template.format(context=item["context"], question=item["question"]),
            "input": "",
            "output": item["answers"]["text"][0]
        }
        extracted_data.append(extracted_item)
    return extracted_data

def get_dataset(data_name, rebuild=False):
    """Get or rebuild dataset from HuggingFace hub"""
    train_path = build_data_path('train_set.json')
    eval_path = build_data_path('eval_set.json')
    if not os.path.exists(train_path) or not os.path.exists(eval_path) or rebuild:
        dataset = datasets.load_dataset(data_name)
        save_res(build_eval_data(dataset['test']), eval_path)
        save_res(build_train_data(dataset['train']), train_path)
    return train_path, eval_path

def cosine(x, y):
    """Calculate cosine similarity between two vectors"""
    product = np.dot(x, y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    raw_cosine = product / norm if norm != 0 else 0.0
    return max(0.0, min(raw_cosine, 1.0))

def check_words_from_content(infer, content):
    """Check if all words in inference output exist in original context"""
    return 1 if all(w in content for w in infer.split()) else 0

def caculate_score(eval_set, infer_set):
    """Calculate three evaluation metrics: exact match, cosine similarity, and word containment"""
    assert len(eval_set) == len(infer_set), \
        f"The size of eval-set is {len(eval_set)}, But size of infer-res is {len(infer_set)}."

    # Initialize embedding model
    m = lazyllm.TrainableModule('bge-large-zh-v1.5')
    m.start()

    accu_exact_score = 0
    accu_cosin_score = 0
    accu_origi_score = 0
    res = []
    for index, eval_item in enumerate(eval_set):
        output = infer_set[index].strip()
        true_v = eval_item['answers']
        # Exact match scoring:
        exact_score = 1 if output == true_v else 0
        accu_exact_score += exact_score
        # Cosine similarity scoring:
        outputs = json.loads(m([output, true_v]))
        cosine_score = cosine(outputs[0], outputs[1])
        accu_cosin_score += cosine_score
        # Word containment scoring:
        origin_score = check_words_from_content(output, eval_item['context'])
        accu_origi_score += origin_score
        res.append({'context': eval_item['context'],
                    'true': true_v,
                    'infer': output,
                    'exact_score': exact_score,
                    'cosine_score': cosine_score,
                    'origin_score': origin_score})
    save_res(res, 'infer_true_cp.json')
    total_score = len(eval_set)
    return (f'Exact Score : {accu_exact_score}/{total_score}, {round(accu_exact_score/total_score,4)*100}%\n'
            f'Cosine Score: {accu_cosin_score}/{total_score}, {round(accu_cosin_score/total_score,4)*100}%\n'
            f'Origin Score: {accu_origi_score}/{total_score}, {round(accu_origi_score/total_score,4)*100}%\n')

def online_infer(model, data):
    res_list = []
    for x in tqdm(data, desc="Processing"):
        try_times = 1
        while try_times < 5:
            try:
                res = model(x)
                if res:
                    try_times = 10
                    res_list.append(res)
                else:
                    try_times += 1
            except Exception:
                try_times += 1
        if try_times != 10:
            res_list.append('')
    return res_list

def main(model_path, mode, eval_data_path, train_data_path, eval_res_path):
    """Main execution flow for different operation modes"""
    # Load evaluation data
    eval_set = load_data(eval_data_path)
    eval_data = [template.format(context=item["context"], question=item["question"])
                 for item in eval_set]

    # Online inference mode
    if mode == 'online_infer':
        model = lazyllm.OnlineChatModule(model_path)
        eval_res = online_infer(model, eval_data)
        # eval_res = [model(x) for x in tqdm(eval_data, desc="Processing")]

    # Local model operations
    if mode in ('local_infer', 'local_train'):
        model = lazyllm.TrainableModule(model_path)\
            .mode('finetune')\
            .trainset(train_data_path)\
            .finetune_method((finetune.llamafactory, {
                'learning_rate': 1e-4,
                'cutoff_len': 5120,
                'max_samples': 20000,
                'val_size': 0.01,
                'per_device_train_batch_size': 2,
                'num_train_epochs': 2.0,
                'launcher': launchers.sco(nnode=1, nproc=1, ngpus=8)
            }))\
            .prompt(dict(system='You are a helpful assistant.', drop_builtin_system=True))\
            .deploy_method(deploy.Vllm)
        model.evalset(eval_data)
        if mode == 'local_train':
            model.update()  # Auto: Start fine-tuning -> Launch inference service -> Run evaluation
        else:
            model.start()  # Start inference service
            model.eval()  # Run evaluation
        eval_res = model.eval_result
    # Score calculation mode
    if mode == 'score':
        infer_res = load_data(eval_res_path)
        eval_res = [item['infer'] for item in infer_res]

    # Calculate and display final scores
    score = caculate_score(eval_set, eval_res)
    time.sleep(5)  # Buffer for log synchronization
    print("All Done. Score is: ", score)

if __name__ == '__main__':
    # Command-line argument configuration
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Pipeline")
    parser.add_argument('--model_path', type=str, default='internlm2-chat-7b',
                        help='Path to model or model identifier')
    parser.add_argument('--dataset_name', type=str, default='cmrc2018',
                        help='Name of HuggingFace dataset')
    parser.add_argument('--train_data_path', type=str, default=None,
                        help='Custom path to training data')
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Custom path to evaluation data')
    parser.add_argument('--eval_res_path', type=str, default=None,
                        help='Path to pre-computed inference results')
    parser.add_argument('--mode', type=str, default='local_infer',
                        choices=['online_infer', 'local_infer', 'local_train', 'score'],
                        help='Operation mode selection')
    args = parser.parse_args()

    # Data path handling
    train_data_path, eval_data_path = get_dataset(args.dataset_name)
    train_data_path = args.train_data_path or train_data_path
    eval_data_path = args.train_data_path or eval_data_path

    # Execute main pipeline
    main(args.model_path, args.mode, eval_data_path, train_data_path, args.eval_res_path)


# Example Usage Patterns:
# 1. Baseline Evaluation:
#    python run_cmrc.py --mode="local_infer" --model_path="internlm2-chat-7b"
#
# 2. Fine-tuning and Evaluation:
#    python run_cmrc.py --mode="local_train"
#
# 3. Online Model Evaluation:
#    python run_cmrc.py --mode="online_infer" --model_path="DeepSeek-V3"
#
# 4. Score Calculation Only:
#    python run_cmrc.py --mode="score" --eval_res_path="path/to/results.json"
