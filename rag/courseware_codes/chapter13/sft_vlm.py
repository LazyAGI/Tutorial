"""
VQA-RAD Evaluation Script with Model Fine-tuning and Inference Capabilities
"""

import os
import json
import time
import datasets
import argparse
import numpy as np
from tqdm import tqdm

import lazyllm
from lazyllm import finetune
from lazyllm.components.formatter import encode_query_with_filepaths


def load_data(data_path):
    """Load JSON data from specified file path"""
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def save_res(data, file_path):
    """Save data to JSON file with proper formatting"""
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def transform_list(data):
    """
    Transforms the input list to a new list of dictionaries with keys 'query', 'answer', and 'img_path'.

    Args:
    data (list): The input list to transform.

    Returns:
    list: A new list of dictionaries with the desired structure.
    """
    transformed = []

    for item in data:
        messages = item.get("messages", [])
        images = item.get("images", [])

        # Extract query and answer, assuming the first message is from the user and the second from the assistant
        if len(messages) >= 2:
            query = messages[0]["content"].replace("<image>", "")
            answer = messages[1]["content"]
        else:
            query = ""
            answer = ""

        # Extract image path, assuming there's at least one image
        img_path = images[0] if images else ""

        # Append the new dictionary to the transformed list
        transformed.append({"query": query, "answer": answer, "img_path": img_path})

    return transformed

def build_data_path(file_name):
    """Construct data storage path and ensure directory exists"""
    data_root = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    save_path = os.path.join(data_root, file_name)
    return save_path

def build_eval_data(data, save_path):
    """Extract necessary fields for evaluation dataset"""
    image_dir = os.path.join(os.path.dirname(save_path), 'img_eval')
    os.makedirs(image_dir, exist_ok=True)
    extracted_data = []
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing eval"):
        image_path = os.path.join(image_dir, f"eval_image_{idx}.jpg")
        item["image"].save(image_path)
        extracted_item = {
            "query": item['question'],
            "answer": item["answer"],
            "img_path": os.path.abspath(image_path)
        }
        extracted_data.append(extracted_item)
    return extracted_data

def build_train_data(data, save_path):
    """Format training data using predefined template"""
    image_dir = os.path.join(os.path.dirname(save_path), 'img_train')
    os.makedirs(image_dir, exist_ok=True)
    extracted_data = []
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing train"):
        image_path = os.path.join(image_dir, f"train_image_{idx}.jpg")
        item["image"].save(image_path)
        conversation = {
            "messages": [
                {
                    "content": f"<image>{item['question']}",
                    "role": "user"
                },
                {
                    "content": item["answer"],
                    "role": "assistant"
                }
            ],
            "images": [os.path.abspath(image_path)]
        }
        extracted_data.append(conversation)
    return extracted_data

def get_dataset(data_name, rebuild=False):
    """Get or rebuild dataset from HuggingFace hub"""
    train_path = build_data_path('train_set.json')
    eval_path = build_data_path('eval_set.json')
    if not os.path.exists(train_path) or not os.path.exists(eval_path) or rebuild:
        dataset = datasets.load_dataset(data_name)
        save_res(build_eval_data(dataset['test'], eval_path), eval_path)
        save_res(build_train_data(dataset['train'], train_path), train_path)
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
    res = []
    for index, eval_item in enumerate(eval_set):
        output = infer_set[index].strip()
        true_v = eval_item['answer']
        # Exact match scoring:
        exact_score = 1 if output == true_v else 0
        accu_exact_score += exact_score
        # Cosine similarity scoring:
        outputs = json.loads(m([output, true_v]))
        cosine_score = cosine(outputs[0], outputs[1])
        accu_cosin_score += cosine_score

        res.append({'query': eval_item['query'],
                    'true': true_v,
                    'infer': output,
                    'exact_score': exact_score,
                    'cosine_score': cosine_score})
    save_res(res, 'infer_true_cp.json')
    total_score = len(eval_set)
    return (f'Exact Score : {accu_exact_score}/{total_score}, {round(accu_exact_score/total_score,4)*100}%\n'
            f'Cosine Score: {accu_cosin_score}/{total_score}, {round(accu_cosin_score/total_score,4)*100}%\n')

def online_infer(model, data):
    res_list = []
    for x in tqdm(data, desc="Processing Online-Infer"):
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

def main(model_name, mode, eval_data_path, train_data_path, eval_res_path, sft_model):
    """Main execution flow for different operation modes"""
    # Load evaluation data
    eval_set = load_data(eval_data_path)
    eval_data = [encode_query_with_filepaths(item["query"], item["img_path"])
                 for item in eval_set]

    # Online inference mode
    if mode == 'online_infer':
        model = lazyllm.OnlineChatModule(model_name)
        eval_res = online_infer(model, eval_data)

    # Local model operations
    if mode in ('local_infer', 'local_train'):
        model = lazyllm.TrainableModule(model_name, sft_model)\
            .mode('finetune')\
            .trainset(train_data_path)\
            .finetune_method((finetune.llamafactory, {
                'learning_rate': 1e-4,
                'cutoff_len': 5120,
                'max_samples': 20000,
                'val_size': 0.01,
                'per_device_train_batch_size': 16,
                'num_train_epochs': 2.0,
            }))
        model.evalset(eval_data)
        if mode == 'local_train':
            # Auto: Start fine-tuning -> Launch inference service -> Run evaluation
            model.update()
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
    parser.add_argument('--model_name', type=str, default='Qwen2.5-VL-3B-Instruct',
                        help='model identifier')
    parser.add_argument('--sft_model', type=str, default='',
                        help='Path to after finetuned model')
    parser.add_argument('--dataset_name', type=str, default='flaviagiammarino/vqa-rad',
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
    parser.add_argument('--build_dataset', action='store_true',
                        help='Force rebuild dataset ignoring existing files')
    args = parser.parse_args()

    # Data path handling
    train_data_path, eval_data_path = get_dataset(args.dataset_name, rebuild=args.build_dataset)
    train_data_path = args.train_data_path or train_data_path
    eval_data_path = args.eval_data_path or eval_data_path

    # Execute main pipeline
    if not args.build_dataset:
        main(args.model_name, args.mode, eval_data_path, train_data_path, args.eval_res_path, args.sft_model)


# Example Usage Patterns:
# 1. Baseline Evaluation:
#    python sft_vlm.py --mode="local_infer" --model_name="Qwen2.5-VL-3B-Instruct" --sft_model="path/to/after/sft/model"
#
# 2. Fine-tuning and Evaluation: Auto: build dataset -> finetune model -> deploy model -> infer all eval-data -> score.
#    python sft_vlm.py --mode="local_train"
#
# 3. Online Model Evaluation:
#    python sft_vlm.py --mode="online_infer" --model_name="SenseNova-V6-Turbo"
#
# 4. Score Calculation Only:
#    python sft_vlm.py --mode="score" --eval_res_path="path/to/results.json"
#
# 5. Build Dataset Only:
#    python sft_vlm.py --build_dataset"
