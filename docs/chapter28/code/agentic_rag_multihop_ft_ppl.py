"""
复杂多跳问答数据增强实验 Pipeline - 基于 LazyLLM Agentic RAG

目标：
  1. 使用原始多跳 QA 数据集评测基座模型效果
  2. 基于支持文档运行 Agentic RAG 的 atomic / depth / width 流水线，生成增强样本
  3. 使用“原始样本 + 增强样本”做一次微调
  4. 在同一验证集上对比微调前后的效果

默认数据集：
  HotpotQA（HuggingFace: hotpot_qa/fullwiki）

使用示例：
  cd /home/mnt/zhangkejun/work/ppl
  python agentic_rag_multihop_ft_ppl.py --run_experiment --max_train_samples 200 --max_eval_samples 50
  python agentic_rag_multihop_ft_ppl.py --build_dataset --build_augmented_dataset --max_train_samples 500
  python agentic_rag_multihop_ft_ppl.py --eval_test --max_eval_samples 100
"""

import argparse
import json
import os
import re
import string
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


_LAZYLLM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../LazyLLM"))
if _LAZYLLM_ROOT not in sys.path:
    sys.path.insert(0, _LAZYLLM_ROOT)

import lazyllm
from lazyllm import LOG, finetune, launchers
from lazyllm.tools.data.pipelines.rag_pipelines import (
    atomic_rag_pipeline,
    depth_qa_pipeline,
    qa_evaluation_pipeline,
    width_qa_pipeline,
)


DEFAULT_DATASET_NAME = "hotpot_qa"
DEFAULT_DATASET_CONFIG = "fullwiki"
DEFAULT_OUTPUT_DIR = "/home/mnt/zhangkejun/work/dataset/agentic_rag_multihop"
DEFAULT_BASE_MODEL = "/mnt/lustre/share_data/lazyllm/models/qwen2.5-14b-instruct"

MULTIHOP_INSTRUCTION = (
    "你是一个多跳问答助手。请严格依据给定上下文回答问题；"
    "如果问题需要跨句或跨段整合证据，请先综合相关事实，再给出简洁答案。"
)

GROUNDING_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context is insufficient, answer with 'unknown'. "
    "Return a concise final answer only.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_jsonl(path: str, rows: List[Dict[str, Any]], keep_all_fields: bool = True) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if keep_all_fields:
                data = row
            else:
                data = {
                    "instruction": row.get("instruction", ""),
                    "input": row.get("input", ""),
                    "output": row.get("output", ""),
                }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pred_to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        return (x.get("content") or x.get("text") or x.get("output") or "").strip()
    if isinstance(x, list):
        return " ".join(_pred_to_str(v) for v in x if _pred_to_str(v)).strip()
    return str(x).strip()


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    text = _safe_str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _join_sentences(sentences: Any) -> str:
    if isinstance(sentences, list):
        return " ".join(_safe_str(s) for s in sentences if _safe_str(s))
    return _safe_str(sentences)


def _extract_support_titles(example: Dict[str, Any]) -> List[str]:
    supporting = example.get("supporting_facts")
    if isinstance(supporting, dict):
        titles = supporting.get("title", [])
        if isinstance(titles, list):
            return [_safe_str(t) for t in titles if _safe_str(t)]
    return []


def extract_context_docs(example: Dict[str, Any], prefer_supporting: bool = True) -> List[Dict[str, str]]:
    context = example.get("context")
    titles: List[str] = []
    sentences_group: List[Any] = []

    if isinstance(context, dict):
        titles = list(context.get("title", []) or [])
        sentences_group = list(context.get("sentences", []) or [])
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                titles.append(_safe_str(item[0]))
                sentences_group.append(item[1])

    support_titles = set(_extract_support_titles(example))
    docs = []
    for idx, title in enumerate(titles):
        sents = sentences_group[idx] if idx < len(sentences_group) else []
        text = _join_sentences(sents)
        if not text:
            continue
        if prefer_supporting and support_titles and title not in support_titles:
            continue
        docs.append({"title": _safe_str(title), "text": text})

    if docs:
        return docs

    fallback = []
    for idx, title in enumerate(titles):
        sents = sentences_group[idx] if idx < len(sentences_group) else []
        text = _join_sentences(sents)
        if text:
            fallback.append({"title": _safe_str(title), "text": text})
    return fallback


def flatten_example_context(example: Dict[str, Any], max_context_chars: int, prefer_supporting: bool = True) -> str:
    docs = extract_context_docs(example, prefer_supporting=prefer_supporting)
    chunks = []
    for doc in docs:
        title = doc.get("title", "")
        text = doc.get("text", "")
        if title:
            chunks.append(f"[{title}] {text}")
        else:
            chunks.append(text)
    return _truncate_text("\n".join(chunks), max_chars=max_context_chars)


def load_multihop_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if load_dataset is None:
        raise ImportError("请先安装 datasets：pip install datasets")

    LOG.info(f"正在加载数据集: {dataset_name} / {dataset_config or '-'} [{split}]")
    try:
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {e}")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    rows = [dict(x) for x in ds]
    LOG.info(f"加载完成: {len(rows)} 条")
    return rows


def build_prompt_input(context: str, question: str) -> str:
    return f"上下文：\n{context}\n\n问题：{question}"


def make_ft_record(
    context: str,
    question: str,
    answer: str,
    source_type: str,
    sample_id: str,
) -> Dict[str, Any]:
    return {
        "instruction": MULTIHOP_INSTRUCTION,
        "input": build_prompt_input(context=context, question=question),
        "output": _safe_str(answer),
        "source_type": source_type,
        "sample_id": sample_id,
        "question": _safe_str(question),
    }


def deduplicate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []
    seen = set()
    for row in records:
        key = (
            row.get("instruction", ""),
            row.get("input", ""),
            row.get("output", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


class GroundedAnswerHelper:
    def __init__(self, llm):
        self._serve = llm.share()
        self._serve.start()

    def answer(self, context: str, question: str, hints: Optional[List[str]] = None) -> str:
        prompt = GROUNDING_PROMPT.format(context=context, question=question)
        if hints:
            clean_hints = [_safe_str(x) for x in hints if _safe_str(x)]
            if clean_hints:
                prompt += "\nPotential evidence:\n- " + "\n- ".join(clean_hints[:4]) + "\n"
        try:
            answer = _pred_to_str(self._serve(prompt))
        except Exception as e:
            LOG.warning(f"Grounded answer generation failed: {e}")
            return ""
        if not answer or _normalize_text(answer) == "unknown":
            return ""
        return answer


def build_raw_training_records(
    train_examples: List[Dict[str, Any]],
    max_context_chars: int,
) -> List[Dict[str, Any]]:
    rows = []
    for idx, ex in enumerate(train_examples):
        question = _safe_str(ex.get("question"))
        answer = _safe_str(ex.get("answer"))
        context = flatten_example_context(ex, max_context_chars=max_context_chars, prefer_supporting=True)
        if not question or not answer or not context:
            continue
        rows.append(make_ft_record(context, question, answer, "raw", f"raw_{idx}"))
    return deduplicate_records(rows)


def build_eval_records(
    eval_examples: List[Dict[str, Any]],
    max_context_chars: int,
) -> List[Dict[str, Any]]:
    rows = []
    for idx, ex in enumerate(eval_examples):
        question = _safe_str(ex.get("question"))
        answer = _safe_str(ex.get("answer"))
        context = flatten_example_context(ex, max_context_chars=max_context_chars, prefer_supporting=True)
        if not question or not answer or not context:
            continue
        rows.append(
            {
                "id": f"eval_{idx}",
                "instruction": MULTIHOP_INSTRUCTION,
                "input": build_prompt_input(context=context, question=question),
                "answer": answer,
                "question": question,
            }
        )
    return rows


def build_source_documents(
    train_examples: List[Dict[str, Any]],
    max_docs: Optional[int],
    max_doc_chars: int,
) -> List[Dict[str, Any]]:
    docs = []
    seen = set()
    for ex_idx, ex in enumerate(train_examples):
        for doc_idx, doc in enumerate(extract_context_docs(ex, prefer_supporting=True)):
            title = _safe_str(doc.get("title"))
            text = _truncate_text(doc.get("text", ""), max_doc_chars)
            if not text:
                continue
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            docs.append(
                {
                    "text": text,
                    "title": title,
                    "source_doc_id": f"doc_{ex_idx}_{doc_idx}",
                }
            )
            if max_docs and len(docs) >= max_docs:
                return docs
    return docs


def generate_atomic_records(
    llm,
    source_docs: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not source_docs:
        return [], []

    ppl = atomic_rag_pipeline(
        llm=llm,
        input_key="text",
        max_per_task=args.atomic_max_per_task,
        max_question=args.atomic_max_question,
        llm_verify_filter_threshold=args.atomic_verify_filter_threshold,
    )
    results = ppl(source_docs)

    ft_rows = []
    width_seed = []
    for idx, item in enumerate(results):
        context = _safe_str(item.get("text"))
        question = _safe_str(item.get("question"))
        answer = _safe_str(item.get("refined_answer") or item.get("answer"))
        identifier = _safe_str(item.get("identifier") or item.get("title"))
        if not context or not question or not answer:
            continue
        ft_rows.append(make_ft_record(context, question, answer, "atomic", f"atomic_{idx}"))
        width_seed.append(
            {
                "question": question,
                "identifier": identifier,
                "answer": answer,
                "source_text": context,
                "source_doc_id": item.get("source_doc_id", ""),
            }
        )
    return deduplicate_records(ft_rows), width_seed


def generate_depth_records(
    llm,
    source_docs: List[Dict[str, Any]],
    args: argparse.Namespace,
    answer_helper: GroundedAnswerHelper,
) -> List[Dict[str, Any]]:
    if not source_docs or args.depth_rounds <= 0:
        return []

    run_depth = depth_qa_pipeline(
        llm=llm,
        input_key="text",
        output_key="depth_question",
        n_rounds=args.depth_rounds,
        depth_verify_filter_threshold=args.depth_verify_filter_threshold,
    )
    results = run_depth(source_docs)

    ft_rows = []
    for item in tqdm(results, desc="generate depth qa"):
        context = _safe_str(item.get("text"))
        if not context:
            continue
        for round_id in range(1, args.depth_rounds + 1):
            question = _safe_str(item.get(f"depth_question_{round_id}"))
            if not question:
                continue
            answer = answer_helper.answer(context=context, question=question)
            if not answer:
                continue
            ft_rows.append(
                make_ft_record(
                    context=context,
                    question=question,
                    answer=answer,
                    source_type="depth",
                    sample_id=f"depth_{round_id}_{len(ft_rows)}",
                )
            )
    return deduplicate_records(ft_rows)


def generate_width_records(
    llm,
    width_seed: List[Dict[str, Any]],
    args: argparse.Namespace,
    answer_helper: GroundedAnswerHelper,
) -> List[Dict[str, Any]]:
    if len(width_seed) < 2:
        return []

    run_width = width_qa_pipeline(
        llm=llm,
        input_question_key="question",
        input_identifier_key="identifier",
        input_answer_key="answer",
        output_question_key="generated_width_task",
        check_require_state_one=args.width_require_state_one,
        width_filter_threshold=args.width_filter_threshold,
        merge_pair_stride=args.merge_pair_stride,
    )
    results = run_width(width_seed)

    ft_rows = []
    for item in tqdm(results, desc="generate width qa"):
        question = _safe_str(item.get("generated_width_task"))
        qa_index = item.get("qa_index", [])
        if not question or not isinstance(qa_index, list):
            continue

        contexts = []
        hints = []
        for idx in qa_index:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(width_seed):
                continue
            ctx = _safe_str(width_seed[idx].get("source_text"))
            ans = _safe_str(width_seed[idx].get("answer"))
            if ctx and ctx not in contexts:
                contexts.append(ctx)
            if ans:
                hints.append(ans)

        merged_context = "\n\n".join(contexts)
        if not merged_context:
            continue
        answer = answer_helper.answer(context=merged_context, question=question, hints=hints)
        if not answer:
            continue
        ft_rows.append(
            make_ft_record(
                context=merged_context,
                question=question,
                answer=answer,
                source_type="width",
                sample_id=f"width_{len(ft_rows)}",
            )
        )
    return deduplicate_records(ft_rows)


def build_augmented_training_data(
    train_examples: List[Dict[str, Any]],
    llm,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    source_docs = build_source_documents(
        train_examples=train_examples,
        max_docs=args.max_aug_docs,
        max_doc_chars=args.max_doc_chars,
    )
    print(f"\n可用于增强的支持文档数: {len(source_docs)}")

    answer_helper = GroundedAnswerHelper(llm)

    atomic_rows, width_seed = generate_atomic_records(llm, source_docs, args)
    depth_rows = generate_depth_records(llm, source_docs, args, answer_helper)
    width_rows = generate_width_records(llm, width_seed, args, answer_helper)

    all_rows = deduplicate_records(atomic_rows + depth_rows + width_rows)
    return {
        "source_docs": source_docs,
        "atomic_rows": atomic_rows,
        "depth_rows": depth_rows,
        "width_rows": width_rows,
        "all_rows": all_rows,
    }


def run_model_eval(model_path_or_obj: Any, eval_prompts: List[str]) -> List[str]:
    if not eval_prompts:
        return []

    if isinstance(model_path_or_obj, str):
        model = lazyllm.TrainableModule(model_path_or_obj)
        model.evalset(eval_prompts)
        model.start()
        model.eval()
        raw = model.eval_result or []
        return [_pred_to_str(x) for x in raw]

    model = model_path_or_obj
    model.evalset(eval_prompts)
    model.eval()
    raw = model.eval_result or []
    return [_pred_to_str(x) for x in raw]


def calculate_eval_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Any]:
    paired = []
    for pred, ref in zip(predictions, references):
        paired.append({"re_answer": pred, "golden_answer": ref})

    f1_pipeline = qa_evaluation_pipeline(
        prediction_key="re_answer",
        ground_truth_key="golden_answer",
        output_key="F1Score",
    )
    scored = f1_pipeline(paired) if paired else []

    avg_f1 = 0.0
    if scored:
        avg_f1 = sum(float(item.get("F1Score", 0.0) or 0.0) for item in scored) / len(scored)

    em_hits = 0
    for pred, ref in zip(predictions, references):
        if _normalize_text(pred) == _normalize_text(ref):
            em_hits += 1

    total = len(references)
    em = (em_hits / total) if total else 0.0
    return {
        "num_samples": total,
        "avg_f1": avg_f1,
        "exact_match": em,
    }


def evaluate_model_on_evalset(
    model_path_or_obj: Any,
    eval_rows: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    prompts = []
    refs = []
    questions = []
    for item in eval_rows:
        prompt = item.get("instruction", "")
        inp = item.get("input", "")
        prompts.append(prompt + "\n\n" + inp if prompt and inp else (prompt or inp))
        refs.append(item.get("answer", ""))
        questions.append(item.get("question", ""))

    predictions = run_model_eval(model_path_or_obj, prompts)
    metrics = calculate_eval_metrics(predictions, refs)
    details = []
    for i in range(len(refs)):
        details.append(
            {
                "question": questions[i],
                "reference": refs[i],
                "prediction": predictions[i] if i < len(predictions) else "",
            }
        )

    if output_path:
        ensure_dir(os.path.dirname(output_path) or ".")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "details": details}, f, ensure_ascii=False, indent=2)
    return metrics


def run_finetune(
    base_model: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int,
    per_device_batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    cutoff_len: int,
    warmup_ratio: float,
    ngpus: int,
):
    print(f'\n{"=" * 72}')
    print("开始微调（Agentic RAG 多跳增强实验）")
    print(f"  基座模型: {base_model}")
    print(f"  训练数据: {train_data_path}")
    print(f"  输出目录: {output_dir}")
    print(
        f"  epochs={num_epochs}, lr={learning_rate}, batch={per_device_batch_size}, "
        f"grad_accum={gradient_accumulation_steps}, cutoff_len={cutoff_len}, warmup_ratio={warmup_ratio}"
    )
    print(f'{"=" * 72}\n')

    model = (
        lazyllm.TrainableModule(base_model)
        .mode("finetune")
        .trainset(train_data_path)
        .finetune_method(
            (
                finetune.auto,
                {
                    "launcher": launchers.remote(nnode=1, nproc=1, ngpus=ngpus),
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": per_device_batch_size,
                    "learning_rate": learning_rate,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "cutoff_len": cutoff_len,
                    "warmup_ratio": warmup_ratio,
                },
            )
        )
    )
    model.update()
    print("微调完成！")
    return model


def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  样本数     : {metrics.get('num_samples', 0)}")
    print(f"  平均 F1    : {metrics.get('avg_f1', 0.0):.4f}")
    print(f"  Exact Match: {metrics.get('exact_match', 0.0):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG 复杂多跳问题数据增强实验 Pipeline（LazyLLM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python agentic_rag_multihop_ft_ppl.py --run_experiment --max_train_samples 200 --max_eval_samples 50
  python agentic_rag_multihop_ft_ppl.py --build_dataset --build_augmented_dataset --max_train_samples 500
  python agentic_rag_multihop_ft_ppl.py --train_flag --eval_test --max_eval_samples 100
        """,
    )

    flow = parser.add_argument_group("流程控制")
    flow.add_argument("--run_experiment", action="store_true", help="一键执行：构建原始集 + 增强集 + 基线评测 + 微调 + 复测")
    flow.add_argument("--build_dataset", action="store_true", help="构建原始训练/评测集")
    flow.add_argument("--build_augmented_dataset", action="store_true", help="构建增强训练集")
    flow.add_argument("--train_flag", action="store_true", help="执行微调")
    flow.add_argument("--eval_test", action="store_true", help="执行评测")

    data = parser.add_argument_group("数据参数")
    data.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    data.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    data.add_argument("--train_split", type=str, default="train")
    data.add_argument("--eval_split", type=str, default="validation")
    data.add_argument("--max_train_samples", type=int, default=1000)
    data.add_argument("--max_eval_samples", type=int, default=200)
    data.add_argument("--max_context_chars", type=int, default=5000)
    data.add_argument("--max_doc_chars", type=int, default=2500)
    data.add_argument("--max_aug_docs", type=int, default=800)
    data.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    aug = parser.add_argument_group("增强参数")
    aug.add_argument("--atomic_max_per_task", type=int, default=5)
    aug.add_argument("--atomic_max_question", type=int, default=5)
    aug.add_argument("--atomic_verify_filter_threshold", type=int, default=1)
    aug.add_argument("--depth_rounds", type=int, default=2)
    aug.add_argument("--depth_verify_filter_threshold", type=int, default=1)
    aug.add_argument("--width_filter_threshold", type=int, default=1)
    aug.add_argument(
        "--merge_pair_stride",
        type=int,
        default=4,
        help="width 阶段合并相邻 QA 对时的步长（越大 LLM merge 调用越少，默认 4）",
    )
    aug.add_argument("--width_require_state_one", action="store_true")

    model = parser.add_argument_group("模型参数")
    model.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    model.add_argument("--finetuned_model_path", type=str, default=None, help="仅评测时可传入已微调模型路径")

    train = parser.add_argument_group("训练参数")
    train.add_argument("--num_epochs", type=int, default=3)
    train.add_argument("--per_device_batch_size", type=int, default=4)
    train.add_argument("--learning_rate", type=float, default=2e-5)
    train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train.add_argument("--cutoff_len", type=int, default=2048)
    train.add_argument("--warmup_ratio", type=float, default=0.05)
    train.add_argument("--ngpus", type=int, default=1)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.run_experiment:
        args.build_dataset = True
        args.build_augmented_dataset = True
        args.train_flag = True
        args.eval_test = True

    if not any([args.build_dataset, args.build_augmented_dataset, args.train_flag, args.eval_test]):
        print("请指定 --run_experiment / --build_dataset / --build_augmented_dataset / --train_flag / --eval_test 至少其一")
        return

    ensure_dir(args.output_dir)

    raw_train_path = os.path.join(args.output_dir, "raw_train.jsonl")
    eval_path = os.path.join(args.output_dir, "eval.jsonl")
    atomic_path = os.path.join(args.output_dir, "aug_atomic.jsonl")
    depth_path = os.path.join(args.output_dir, "aug_depth.jsonl")
    width_path = os.path.join(args.output_dir, "aug_width.jsonl")
    aug_train_path = os.path.join(args.output_dir, "augmented_train.jsonl")
    merged_train_path = os.path.join(args.output_dir, "merged_train.jsonl")
    summary_path = os.path.join(args.output_dir, "experiment_summary.json")

    print(f'\n{"=" * 84}')
    print(" " * 12 + "Agentic RAG 复杂多跳问题数据增强实验")
    print(f'{"=" * 84}\n')

    train_examples = []
    eval_examples = []
    if args.build_dataset or args.build_augmented_dataset or args.train_flag or args.eval_test:
        train_examples = load_multihop_dataset(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.train_split,
            max_samples=args.max_train_samples,
        )
        eval_examples = load_multihop_dataset(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.eval_split,
            max_samples=args.max_eval_samples,
        )

    if args.build_dataset:
        print(">>> 步骤 1：构建原始训练/评测集")
        raw_train_rows = build_raw_training_records(train_examples, max_context_chars=args.max_context_chars)
        eval_rows = build_eval_records(eval_examples, max_context_chars=args.max_context_chars)
        save_jsonl(raw_train_path, raw_train_rows, keep_all_fields=False)
        save_jsonl(eval_path, eval_rows, keep_all_fields=True)
        print(f"  原始训练集: {raw_train_path} ({len(raw_train_rows)} 条)")
        print(f"  评测集    : {eval_path} ({len(eval_rows)} 条)")

    raw_train_rows = load_jsonl(raw_train_path)
    eval_rows = load_jsonl(eval_path)

    llm = lazyllm.TrainableModule(args.base_model)

    if args.build_augmented_dataset:
        print("\n>>> 步骤 2：使用 Agentic RAG 生成增强样本")
        aug_data = build_augmented_training_data(train_examples, llm, args)
        save_jsonl(atomic_path, aug_data["atomic_rows"], keep_all_fields=False)
        save_jsonl(depth_path, aug_data["depth_rows"], keep_all_fields=False)
        save_jsonl(width_path, aug_data["width_rows"], keep_all_fields=False)
        save_jsonl(aug_train_path, aug_data["all_rows"], keep_all_fields=False)

        merged_rows = deduplicate_records(raw_train_rows + aug_data["all_rows"])
        save_jsonl(merged_train_path, merged_rows, keep_all_fields=False)

        print(f"  atomic 增强: {len(aug_data['atomic_rows'])} 条")
        print(f"  depth 增强 : {len(aug_data['depth_rows'])} 条")
        print(f"  width 增强 : {len(aug_data['width_rows'])} 条")
        print(f"  总增强数据 : {len(aug_data['all_rows'])} 条 -> {aug_train_path}")
        print(f"  合并训练集 : {len(merged_rows)} 条 -> {merged_train_path}")

    finetuned_model = None
    baseline_metrics = None
    after_metrics = None

    if args.eval_test:
        if not eval_rows:
            print(f"\n错误：评测集不存在，请先执行 --build_dataset 生成 {eval_path}")
            return
        print("\n>>> 步骤 3：评测基座模型（原始验证集）")
        baseline_metrics = evaluate_model_on_evalset(
            model_path_or_obj=args.base_model,
            eval_rows=eval_rows,
            output_path=os.path.join(args.output_dir, "eval_before.json"),
        )
        print_metrics("基座模型结果", baseline_metrics)

    if args.train_flag:
        if not os.path.isfile(merged_train_path):
            if raw_train_rows:
                save_jsonl(merged_train_path, raw_train_rows, keep_all_fields=False)
            else:
                print(f"\n错误：训练数据不存在，请先执行 --build_dataset 或 --build_augmented_dataset")
                return
        print("\n>>> 步骤 4：基于“原始 + 增强”训练集进行微调")
        finetuned_model = run_finetune(
            base_model=args.base_model,
            train_data_path=merged_train_path,
            output_dir=os.path.join(args.output_dir, "finetuned_model"),
            num_epochs=args.num_epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cutoff_len=args.cutoff_len,
            warmup_ratio=args.warmup_ratio,
            ngpus=args.ngpus,
        )

    if args.eval_test:
        eval_target = finetuned_model or args.finetuned_model_path
        if eval_target is not None:
            print("\n>>> 步骤 5：评测微调后模型（原始验证集）")
            after_metrics = evaluate_model_on_evalset(
                model_path_or_obj=eval_target,
                eval_rows=eval_rows,
                output_path=os.path.join(args.output_dir, "eval_after.json"),
            )
            print_metrics("微调后结果", after_metrics)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "base_model": args.base_model,
        "raw_train_size": len(raw_train_rows),
        "eval_size": len(eval_rows),
        "baseline_metrics": baseline_metrics,
        "after_metrics": after_metrics,
        "paths": {
            "raw_train": raw_train_path,
            "eval": eval_path,
            "aug_atomic": atomic_path,
            "aug_depth": depth_path,
            "aug_width": width_path,
            "augmented_train": aug_train_path,
            "merged_train": merged_train_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n实验摘要已保存: {summary_path}")


if __name__ == "__main__":
    main(parse_args())
