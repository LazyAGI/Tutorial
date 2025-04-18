import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import lazyllm
from lazyllm import Document
from lazyllm.tools.eval import NonLLMContextRecall, ContextRelevance


def build_data_path(dir_name: str, file_name: str) -> str:
    """Construct data storage path and ensure directory exists.

    Args:
        dir_name (str): Directory name to create/store files
        file_name (str): Target file name

    Returns:
        str: Full path to the target file
    """
    data_root = os.path.join(os.getcwd(), dir_name)
    os.makedirs(data_root, exist_ok=True)
    return os.path.join(data_root, file_name)


def build_dataset_corpus(instruction: str, neg_num: int = 10, test_size: float = 0.1, seed: int = 1314) -> tuple:
    """Process dataset and create training/evaluation files.

    Args:
        instruction (str): Instruction template for prompts
        neg_num (int): Number of negative samples per instance
        test_size (float): Proportion of data for test split
        seed (int): Random seed for reproducibility

    Returns:
        tuple: Paths to training data, evaluation data, and knowledge base directory
    """
    # Load and preprocess dataset
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    ds = ds.select_columns(column_names=["question", "context"])
    ds = ds.rename_columns({"question": "query", "context": "pos"})

    # Generate negative samples
    np.random.seed(seed)
    new_col = []
    for i in range(len(ds)):
        ids = np.random.randint(0, len(ds), size=neg_num)
        while i in ids:  # Ensure no self-match in negatives
            ids = np.random.randint(0, len(ds), size=neg_num)
        neg = [ds[int(i)]["pos"] for i in ids]
        new_col.append(neg)

    # Create dataset splits
    ds = ds.add_column("neg", new_col)

    def str_to_lst(data):
        data["pos"] = [data["pos"]]
        return data
    ds = ds.map(str_to_lst)  # Convert pos to list format
    ds = ds.add_column("prompt", [instruction] * len(ds))
    split = ds.train_test_split(test_size=test_size, shuffle=True, seed=seed)

    # Save training data
    train_data_path = build_data_path('dataset', 'train.json')
    split["train"].to_json(train_data_path)

    # Process and save evaluation data
    test = split["test"].select_columns(["query", "pos"]).rename_column("pos", "corpus")
    eval_data_path = build_data_path('dataset', 'eval.json')
    test.to_json(eval_data_path)

    # Create knowledge base
    kb_data_path = build_data_path('KB', 'knowledge_base.txt')
    corpus = "\n".join([''.join(item) for item in test['corpus']])
    with open(kb_data_path, 'w', encoding='utf-8') as f:
        f.write(corpus)

    return train_data_path, eval_data_path, os.path.dirname(kb_data_path)


def deploy_serve(
    kb_path: str,
    embed_path: str,
    train_data_path: str,
    train_flag: bool = True,
    per_device_batch_size: int = 16,
    num_epochs: int = 2,
    ngpus: int = 4
) -> lazyllm.Retriever:
    """Deploy the retrieval service with optional fine-tuning.

    Args:
        kb_path (str): Path to knowledge base directory
        embed_path (str): Embedding model path/name
        train_data_path (str): Path to training data
        train_flag (bool): Whether to perform fine-tuning
        per_device_batch_size (int): Training batch size per device
        num_epochs (int): Number of training epochs
        ngpus (int): Number of GPUs to use

    Returns:
        lazyllm.Retriever: Configured retriever instance
    """
    # Configure embedding model
    embed = lazyllm.TrainableModule(embed_path)\
        .mode('finetune').trainset(train_data_path)\
        .finetune_method((
            lazyllm.finetune.flagembedding,
            {
                'launcher': lazyllm.launchers.remote(nnode=1, nproc=1, ngpus=ngpus),
                'per_device_train_batch_size': per_device_batch_size,
                'num_train_epochs': num_epochs,
            }
        ))

    # Create document processing pipeline
    docs = Document(kb_path, embed=embed, manager=False)
    docs.create_node_group(name='split_sent', transform=lambda s: s.split('\n'))

    # Configure retriever
    retriever = lazyllm.Retriever(doc=docs, group_name="split_sent", similarity="cosine", topk=1)
    retriever.update() if train_flag else retriever.start()

    return retriever


def evaluate_results(data: list) -> tuple:
    """Evaluate retrieval results using multiple metrics.

    Args:
        data (list): List of retrieval results to evaluate

    Returns:
        tuple: Evaluation scores (context recall, context relevance)
    """
    recall_eval = NonLLMContextRecall(binary=False)
    relevance_eval = ContextRelevance()
    return recall_eval(data), relevance_eval(data)


def load_json(file_path: str, line_by_line: bool = True) -> list:
    """Load JSON data from file.

    Args:
        file_path (str): Path to JSON file
        line_by_line (bool): Whether file contains JSON lines

    Returns:
        list: Loaded data
    """
    if line_by_line:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: list, file_path: str) -> None:
    """Save data to JSON file with proper formatting.

    Args:
        data (list): Data to save
        file_path (str): Target file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Document Retrieval SFT and Eval System')
    parser.add_argument('--embed_path', type=str, default='bge-large-zh-v1.5',
                        help='Embedding model path/name')
    parser.add_argument('--instruction', type=str,
                        default="Represent this sentence for searching relevant passages: ",
                        help='Prompt template for queries')
    parser.add_argument('--train_flag', action='store_true',
                        help='Perform model fine-tuning')
    parser.add_argument('--use_instruction', action='store_true',
                        help='Prepend instruction to queries')
    parser.add_argument('--output_path', type=str, default='eval_res.json',
                        help='Path to save evaluation results')
    parser.add_argument('--neg_num', type=int, default=10,
                        help='Number of negative samples per instance')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of data for evaluation')
    parser.add_argument('--seed', type=int, default=1314,
                        help='Random seed for reproducibility')
    parser.add_argument('--per_device_batch_size', type=int, default=16,
                        help='Training batch size per device')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--ngpus', type=int, default=4,
                        help='Number of GPUs for training')
    parser.add_argument('--build_dataset', action='store_true',
                        help='Build or Force rebuild the dataset even if it exists')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main execution pipeline."""
    # Prepare dataset
    if args.build_dataset:
        print("Rebuilding dataset as requested...")
        train_data_path, eval_data_path, kb_path = build_dataset_corpus(
            instruction=args.instruction,
            neg_num=args.neg_num,
            test_size=args.test_size,
            seed=args.seed
        )
        return
    else:
        work_path = os.getcwd()
        train_path = os.path.join(work_path, 'dataset', 'train.json')
        eval_path = os.path.join(work_path, 'dataset', 'eval.json')
        kb_file = os.path.join(work_path, 'KB', 'knowledge_base.txt')

        if all(os.path.exists(f) for f in [train_path, eval_path, kb_file]):
            print("Using existing dataset files.")
            train_data_path = train_path
            eval_data_path = eval_path
            kb_path = os.path.dirname(kb_file)
        else:
            print("Warning: Existing processed dataset not found. Building dataset...")
            train_data_path, eval_data_path, kb_path = build_dataset_corpus(
                instruction=args.instruction,
                neg_num=args.neg_num,
                test_size=args.test_size,
                seed=args.seed
            )

    # Deploy retrieval service
    retriever = deploy_serve(
        kb_path=kb_path,
        embed_path=args.embed_path,
        train_data_path=train_data_path,
        train_flag=args.train_flag,
        per_device_batch_size=args.per_device_batch_size,
        num_epochs=args.num_epochs,
        ngpus=args.ngpus
    )

    # Run SFT or Evaluation
    results = []
    query_corpus = load_json(eval_data_path)
    for item in tqdm(query_corpus, desc="Processing queries"):
        query = item['query']
        inputs = f"{args.instruction}{query}" if args.use_instruction or args.train_flag else query
        retrieved = retriever(inputs)
        results.append({
            'question': query,
            'context_retrieved': [text.get_text() for text in retrieved],
            'context_reference': item['corpus']
        })

    # Save and report results
    save_json(results, args.output_path)
    recall_score, relevance_score = evaluate_results(results)
    print(f"Evaluation Complete!\nContext Recall: {recall_score}\nContext Relevance: {relevance_score}")


if __name__ == '__main__':
    main(parse_args())

"""
Usage Examples:

    # Build dataset with custom parameters
    python sft_embed.py --build_dataset --neg_num 10 --test_size 0.1

    # Basic evaluation with default settings
    python sft_embed.py

    # Enable fine-tuning with custom parameters
    python sft_embed.py --train_flag --embed_path bge-large-zh-v1.5 --num_epochs 2 --ngpus 4
"""
