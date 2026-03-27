"""
SQuAD 2.0 长上下文微调：使用 build_long_context_pipeline 将短 context 扩写并重组为长上下文，
再进行 SFT / LoRA 微调与评测。

pipeline 流程：
  SQuAD（短 context）
      ↓
  [pipeline] 32B LLM 扩写 + 长上下文重组（ContextExpansion + ContextReconstruction）
      ↓
  长上下文 SFT 数据

数据落盘：
  - 训练用未处理原始样本（与 pipeline 输入一致）：squad_raw.jsonl
    按 context 去重，且 context / question / answer 均非空；非全量 SQuAD
  - pipeline 处理后的长上下文数据：squad_ppl_long.jsonl
  - 训练集：squad_ppl_train.json
  - 评测集：squad_ppl_eval.jsonl

评测指标（与 run_narrativeqa.py 一致）：
  - Token-F1 / ROUGE-L / Exact Match，对多答案取最优

支持 --mode: prepare | infer | train | eval | full
"""
import os
import re
import json
import random
import string
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from datasets import load_dataset
from lazyllm import finetune, launchers, deploy, TrainableModule
from lazyllm.tools.data.pipelines.pt_data_ppl import (
    build_long_context_pipeline,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CKPT_DIR = os.path.join(BASE_DIR, 'ckpt')
RAW_JSONL_PATH = os.path.join(DATA_DIR, 'squad_raw.jsonl')
PPL_JSONL_PATH = os.path.join(DATA_DIR, 'squad_ppl_long.jsonl')
TRAIN_JSON_PATH = os.path.join(DATA_DIR, 'squad_ppl_train.json')
EVAL_JSONL_PATH = os.path.join(DATA_DIR, 'squad_ppl_eval.jsonl')

TRAIN_SAMPLES = 3000
EVAL_SAMPLES = 200
SOURCE_LOAD_LIMIT = 4000
TRAIN_CONTEXT_WINDOW = 8192

BASE_MODEL_PATH = (
    '/home/mnt/chenzhe1/.lazyllm/model/modelscope/qwen/'
    'Qwen2.5-7B-Instruct'
)
PPL_MODEL = 'qwen2.5-32b-instruct'
PPL_TARGET_WORDS = '700-1100'

INSTRUCTION = (
    'You are a reading comprehension assistant. '
    'Read the following document carefully and answer the question '
    'based solely on the information provided in the document. '
    'Give a concise and accurate answer.'
)
CONTEXT_TEMPLATE = 'Document:\n{context}\n\nQuestion: {question}'
PROMPT_TEMPLATE = '{instruction}\n\n{context_block}'
PPL_EXPANSION_PROMPT = (
    'Expand the given context into a high-quality long passage '
    'for QA training.\n'
    'Hard constraints:\n'
    '1. Keep all original facts and timeline unchanged.\n'
    '2. Ensure the provided answer remains directly inferable '
    'from the passage.\n'
    '3. Do NOT add contradictory facts, fabricated entities, or speculation.\n'
    '4. Do NOT explicitly highlight the answer.\n'
    '5. Keep the passage fluent and coherent with clear narrative flow.\n'
    f'6. Target length: about {PPL_TARGET_WORDS} English words.\n'
    'Output only the expanded context.'
)


# ──────────────────────────────────────────────
# 评测指标
# ──────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if x[i] == y[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev = curr
    return prev[n]


def _rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _token_f1_triplet(pred: str, gold: str) -> Tuple[float, float, float]:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    common = sum(
        min(gold_counts[t], pred_counts.get(t, 0)) for t in gold_counts
    )
    if common == 0:
        return 0.0, 0.0, 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def _best_scores(
    pred: str,
    gold_answers: List[str],
) -> Tuple[float, float, float, float, float]:
    best_rl = best_tf1 = best_em = 0.0
    best_tp = best_tr = 0.0
    for gold in gold_answers:
        rl = _rouge_l_f1(pred, gold)
        tp, tr, tf1 = _token_f1_triplet(pred, gold)
        em = 1.0 if _normalize(pred) == _normalize(gold) else 0.0
        if rl > best_rl:
            best_rl = rl
        if tf1 > best_tf1:
            best_tf1 = tf1
            best_tp = tp
            best_tr = tr
        if em > best_em:
            best_em = em
    return best_rl, best_tp, best_tr, best_tf1, best_em


def _norm_preds(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() if x is not None else '' for x in raw]
    return []


# ──────────────────────────────────────────────
# 数据准备
# ──────────────────────────────────────────────

def _context_dedup_key(context: str) -> str:
    """用于判断 context 是否重复：去首尾空白并压缩空白。"""
    return re.sub(r'\s+', ' ', context.strip())


def _load_squad_raw(limit: int) -> List[Dict]:
    """加载 SQuAD 2.0：有答案、三字段非空，且 context 不重复；最多 limit 条；落盘与 pipeline 输入一致。"""
    if os.path.exists(RAW_JSONL_PATH):
        print(f'原始数据已存在，直接加载: {RAW_JSONL_PATH}')
        records = []
        seen_ctx = set()
        with open(RAW_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if len(records) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ctx = (rec.get('context') or '').strip()
                q = (rec.get('question') or '').strip()
                ans = (rec.get('answer') or '').strip()
                if not ctx or not q or not ans:
                    continue
                key = _context_dedup_key(ctx)
                if key in seen_ctx:
                    continue
                seen_ctx.add(key)
                records.append(rec)
        print(
            f'加载原始数据: {len(records)} 条（按 context 去重，上限 {limit}）'
        )
        return records

    print('正在从 HuggingFace 加载 SQuAD 2.0 数据集...')
    dataset = None
    load_errors = []
    for dataset_name in ('squad_v2', 'rajpurkar/squad_v2'):
        try:
            dataset = load_dataset(dataset_name, split='train')
            print(f'已加载数据集: {dataset_name}')
            break
        except Exception as err:
            load_errors.append(f'{dataset_name}: {err}')
    if dataset is None:
        msg = '\n'.join(load_errors)
        raise RuntimeError(
            '无法加载 SQuAD 2.0 数据集，请检查 datasets 版本或网络环境。\n'
            f'尝试记录：\n{msg}'
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    records = []
    seen_ctx = set()
    with open(RAW_JSONL_PATH, 'w', encoding='utf-8') as f:
        for item in dataset:
            if len(records) >= limit:
                break
            answers = item.get('answers', {})
            answer_texts = (
                answers.get('text', [])
                if isinstance(answers, dict)
                else []
            )
            if not answer_texts:
                continue
            rec = {
                'context': item['context'].strip(),
                'question': item['question'].strip(),
                'answer': answer_texts[0].strip(),
                'gold_answers': [a.strip() for a in answer_texts if a.strip()],
            }
            if not rec['context'] or not rec['question'] or not rec['answer']:
                continue
            key = _context_dedup_key(rec['context'])
            if key in seen_ctx:
                continue
            seen_ctx.add(key)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            records.append(rec)

    print(
        f'原始数据已保存: {RAW_JSONL_PATH}，共 {len(records)} 条'
        f'（context 去重、三字段非空，上限 {limit}）'
    )
    if len(records) < limit:
        print(
            f'警告：去重后仅得到 {len(records)} 条，不足上限 {limit}；'
            '可增大扫描或检查数据集。'
        )
    return records


def _run_long_context_pipeline(records: List[Dict]) -> List[Dict]:
    """
    调用 build_long_context_pipeline：
      - ContextExpansion：32B LLM 将短 context 扩写为长文档
      - ContextReconstruction：将扩写文档与干扰段拼接成长上下文
    输出字段：{long_context, question, answer}
    """
    print('\n正在启动 32B LLM（vllm）用于 context 扩写...')
    llm = TrainableModule(PPL_MODEL).deploy_method(
        (deploy.vllm, {
            'tensor_parallel_size': 1,
            'max_num_seqs': 8,
        })
    ).start()
    print('LLM 启动完成')

    ppl = build_long_context_pipeline(
        llm=llm,
        context_key='context',
        question_key='question',
        answer_key='answer',
        expanded_key='expanded_context',
        long_context_key='long_context',
        expansion_prompt=PPL_EXPANSION_PROMPT,
        num_distractors=3,
        passage_sep='\n\n',
        seed=42,
    )

    print(f'正在通过 build_long_context_pipeline 处理 {len(records)} 条数据...')
    results = ppl(records)
    results = (
        results
        if isinstance(results, list)
        else ([] if not results else [results])
    )

    print(f'pipeline 完成: 输入 {len(records)} 条 → 输出 {len(results)} 条长上下文')

    gold_map = {
        (r['question'], r['answer']): r.get('gold_answers', [r['answer']])
        for r in records
    }
    valid = []
    for rec in results:
        if (
            not rec.get('long_context')
            or not rec.get('question')
            or not rec.get('answer')
        ):
            continue
        key = (rec['question'], rec['answer'])
        rec['gold_answers'] = gold_map.get(key, [rec['answer']])
        valid.append(rec)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PPL_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in valid:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'长上下文数据已保存: {PPL_JSONL_PATH}，共 {len(valid)} 条')

    getattr(llm, 'stop', lambda: None)()
    return valid


def _load_ppl_records() -> List[Dict]:
    records = []
    with open(PPL_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f'加载长上下文数据: {len(records)} 条')
    return records


def _to_sft(rec: Dict) -> Optional[Dict]:
    long_ctx = rec.get('long_context', '').strip()
    question = rec.get('question', '').strip()
    answer = rec.get('answer', '').strip()
    if not long_ctx or not question or not answer:
        return None
    context_block = CONTEXT_TEMPLATE.format(
        context=long_ctx,
        question=question,
    )
    return {
        'instruction': INSTRUCTION,
        'input': context_block,
        'output': answer,
        'gold_answers': rec.get('gold_answers', [answer]),
        'question': question,
    }


def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TRAIN_JSON_PATH) and os.path.exists(EVAL_JSONL_PATH):
        print('训练集与评测集已存在，跳过生成')
        return

    if os.path.exists(PPL_JSONL_PATH):
        print(f'长上下文数据已存在，直接加载: {PPL_JSONL_PATH}')
        records = _load_ppl_records()
    else:
        raw = _load_squad_raw(SOURCE_LOAD_LIMIT)
        if not raw:
            print('无有效原始数据，退出')
            return
        records = _run_long_context_pipeline(raw)

    if not records:
        print('无有效长上下文记录，退出')
        return

    rows = [_to_sft(r) for r in records]
    rows = [r for r in rows if r is not None]
    print(f'有效 SFT 样本: {len(rows)} 条')

    random.seed(42)
    random.shuffle(rows)
    train_list = rows[:TRAIN_SAMPLES]
    eval_indices = random.sample(
        range(len(train_list)), min(EVAL_SAMPLES, len(train_list))
    )
    eval_list = [train_list[i] for i in sorted(eval_indices)]

    train_sft = [
        {
            'instruction': r['instruction'],
            'input': r['input'],
            'output': r['output'],
        }
        for r in train_list
    ]
    with open(TRAIN_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_sft, f, ensure_ascii=False, indent=2)

    with open(EVAL_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in eval_list:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'训练集已保存: {TRAIN_JSON_PATH}，共 {len(train_list)} 条')
    print(f'评测集已保存: {EVAL_JSONL_PATH}，共 {len(eval_list)} 条')
    if train_list:
        ex = train_list[0]
        print('=== 训练集示例 ===')
        print(f'  问题: {ex["question"]}')
        print(f'  答案: {ex["output"]}')
        print(f'  长上下文长度: {len(ex["input"].split())} 词')


# ──────────────────────────────────────────────
# 评测工具
# ──────────────────────────────────────────────

def load_eval_samples(path: str) -> List[Dict]:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f'加载评测集 {len(samples)} 条')
    return samples


def build_eval_prompts(samples: List[Dict]) -> List[str]:
    return [
        PROMPT_TEMPLATE.format(
            instruction=INSTRUCTION,
            context_block=s.get('input', ''),
        )
        for s in samples
    ]


def _compute_metrics(
    samples: List[Dict],
    preds: List[str],
) -> Dict[str, float]:
    n = len(samples)
    if not n or not preds or len(preds) != n:
        return {}
    total_rl = total_tf1 = total_tp = total_tr = total_em = 0.0
    empty_count = 0
    for i, sample in enumerate(samples):
        pred = preds[i] if i < len(preds) else ''
        if not pred or not pred.strip():
            empty_count += 1
            continue
        gold_answers = sample.get('gold_answers') or [sample.get('output', '')]
        rl, tp, tr, tf1, em = _best_scores(pred, gold_answers)
        total_rl += rl
        total_tf1 += tf1
        total_tp += tp
        total_tr += tr
        total_em += em
    return {
        'rouge_l': total_rl / n,
        'token_f1': total_tf1 / n,
        'token_precision': total_tp / n,
        'token_recall': total_tr / n,
        'exact_match': total_em / n,
        'empty_pred_rate': empty_count / n,
        'num_samples': n,
    }


def _print_metrics(label: str, m: Dict[str, float]):
    if not m:
        return
    print(
        f'[{label}] '
        f'ROUGE-L={m.get("rouge_l", 0):.4f}  '
        f'Token-F1={m.get("token_f1", 0):.4f}  '
        f'Token-P={m.get("token_precision", 0):.4f}  '
        f'Token-R={m.get("token_recall", 0):.4f}  '
        f'EM={m.get("exact_match", 0):.4f}  '
        f'empty={m.get("empty_pred_rate", 0):.4f}'
    )


def run_eval(
    samples: List[Dict],
    base_preds: Optional[List[str]],
    ckpt_preds: Optional[List[str]],
    save_dir: Optional[str],
):
    base_preds = _norm_preds(base_preds)
    ckpt_preds = _norm_preds(ckpt_preds)
    metrics = {'num_samples': len(samples), 'base': None, 'ckpt': None}

    if base_preds and len(base_preds) == len(samples):
        m = _compute_metrics(samples, base_preds)
        metrics['base'] = m
        _print_metrics('基座模型', m)

    if ckpt_preds and len(ckpt_preds) == len(samples):
        m = _compute_metrics(samples, ckpt_preds)
        metrics['ckpt'] = m
        _print_metrics('微调模型', m)

    if not save_dir:
        return

    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, 'eval_squad_ppl_metrics.json'),
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(save_dir, 'eval_squad_ppl_results.jsonl'),
        'w',
        encoding='utf-8',
    ) as f:
        for i, s in enumerate(samples):
            rec = {
                'id': i + 1,
                'question': s.get('question', ''),
                'gold_answers': s.get('gold_answers') or [s.get('output', '')],
                'base_pred': (
                    base_preds[i]
                    if base_preds and i < len(base_preds)
                    else ''
                ),
                'ckpt_pred': (
                    ckpt_preds[i]
                    if ckpt_preds and i < len(ckpt_preds)
                    else ''
                ),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'评测结果已保存至: {save_dir}')


# ──────────────────────────────────────────────
# 训练 & 主流程
# ──────────────────────────────────────────────

def _gen_ckpt_dir() -> str:
    ts = datetime.now().strftime('%y%m%d%H%M%S')
    out = os.path.join(CKPT_DIR, f'qwen2_5_7b_squad_ppl_{ts}')
    os.makedirs(CKPT_DIR, exist_ok=True)
    return out


def main(
    model_path: str,
    mode: str,
    eval_data_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    eval_res_path: Optional[str] = None,
):
    eval_data_path = eval_data_path or EVAL_JSONL_PATH
    train_data_path = train_data_path or TRAIN_JSON_PATH

    # 仅在目标数据文件缺失时才触发数据准备，避免重复跑 pipeline。
    if mode == 'prepare':
        prepare_dataset()
        return
    if mode == 'infer':
        if not os.path.exists(eval_data_path):
            print('检测到评测数据不存在，开始执行数据准备...')
            prepare_dataset()
    elif mode in ('train', 'full'):
        need_prepare = (
            not os.path.exists(eval_data_path)
            or not os.path.exists(train_data_path)
        )
        if need_prepare:
            print('检测到训练或评测数据不存在，开始执行数据准备...')
            prepare_dataset()

    if not os.path.exists(eval_data_path):
        print(f'评测集不存在: {eval_data_path}，请先运行 prepare 或 full')
        return
    samples = load_eval_samples(eval_data_path)
    if not samples:
        print('无有效评测样本')
        return
    eval_prompts = build_eval_prompts(samples)

    if mode == 'infer':
        model = TrainableModule(model_path).deploy_method(
            (deploy.vllm, {
                'tensor_parallel_size': 1,
                'max_num_seqs': 4,
                'max_model_len': TRAIN_CONTEXT_WINDOW,
            })
        )
        model.evalset(eval_prompts)
        model.start()
        model.eval()
        base_preds = _norm_preds(model.eval_result)
        run_eval(samples, base_preds, None, None)
        return

    if mode in ('train', 'full'):
        target_path = _gen_ckpt_dir()
        base_preds = None

        if mode == 'full':
            base_model = TrainableModule(model_path).deploy_method(
                (deploy.vllm, {
                    'tensor_parallel_size': 1,
                    'max_num_seqs': 32,
                    'max_model_len': TRAIN_CONTEXT_WINDOW,
                })
            )
            base_model.evalset(eval_prompts)
            base_model.start()
            base_model.eval()
            base_preds = _norm_preds(base_model.eval_result)
            getattr(base_model, 'stop', lambda: None)()

        model = TrainableModule(model_path, target_path=target_path)
        model.mode('finetune')
        model.trainset(train_data_path)
        model.finetune_method((finetune.llamafactory, {
            'stage': 'sft',
            'finetuning_type': 'lora',
            'lora_rank': 16,
            'lora_alpha': 32,
            'learning_rate': 1e-4,
            'cutoff_len': TRAIN_CONTEXT_WINDOW,
            'val_size': 0.05,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'num_train_epochs': 5,
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            'save_steps': 50,
            'logging_steps': 10,
            'save_strategy': 'steps',
            'save_total_limit': 5,
            'launcher': launchers.empty(ngpus=1),
        }))
        model.deploy_method((deploy.vllm, {
            'tensor_parallel_size': 1,
            'max_num_seqs': 32,
            'max_model_len': TRAIN_CONTEXT_WINDOW,
        }))
        model.evalset(eval_prompts)
        model.update()

        ckpt_preds = _norm_preds(getattr(model, 'eval_result', None))
        save_dir = os.path.join(
            RESULTS_DIR,
            datetime.now().strftime('%Y%m%d_%H%M%S'),
        )
        run_eval(
            samples,
            base_preds,
            ckpt_preds if ckpt_preds else None,
            save_dir,
        )
        print(f'训练输出目录: {target_path}')
        return

    if mode == 'eval':
        eval_res_path = eval_res_path or os.path.join(
            RESULTS_DIR, 'eval_squad_ppl_results.jsonl'
        )
        if not os.path.exists(eval_res_path):
            print(f'结果文件不存在: {eval_res_path}')
            return
        samples_from_file, base_preds, ckpt_preds = [], [], []
        with open(eval_res_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                samples_from_file.append({
                    'question': rec.get('question', ''),
                    'output': (rec.get('gold_answers') or [''])[0],
                    'gold_answers': rec.get('gold_answers', []),
                })
                base_preds.append(rec.get('base_pred', ''))
                ckpt_preds.append(rec.get('ckpt_pred', ''))
        run_eval(samples_from_file, base_preds, ckpt_preds, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SQuAD 2.0 长上下文 pipeline SFT 微调与评测'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=BASE_MODEL_PATH,
        help='基座模型路径',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['prepare', 'infer', 'train', 'eval', 'full'],
        help=(
            'prepare=仅数据准备; infer=仅基座推理评测; '
            'train=训练+微调模型评测; eval=从结果文件重算指标; '
            'full=prepare+基座评测+训练+微调评测'
        ),
    )
    parser.add_argument(
        '--eval_data_path',
        type=str,
        default=None,
        help='评测集 JSONL 路径',
    )
    parser.add_argument(
        '--train_data_path',
        type=str,
        default=None,
        help='训练集 JSON 路径',
    )
    parser.add_argument(
        '--eval_res_path',
        type=str,
        default=None,
        help='eval 模式下的结果 JSONL 路径',
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        mode=args.mode,
        eval_data_path=args.eval_data_path,
        train_data_path=args.train_data_path,
        eval_res_path=args.eval_res_path,
    )
