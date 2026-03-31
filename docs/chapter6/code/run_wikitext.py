"""
Wikitext 预训练一体化脚本：数据准备、训练、评测一步到位。
支持 --mode: prepare | train | eval | full
"""
import os
import re
import json
import torch
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import lazyllm
from lazyllm import finetune, launchers
from datasets import load_dataset
from lazyllm.tools.data.pipelines.pt_text_ppl import build_text_pt_pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PRETRAIN_CKPT_DIR = os.path.join(BASE_DIR, 'pretrain_ckpt')
TRAIN_JSON_PATH = os.path.join(DATA_DIR, 'wikitext_train.json')
EVAL_JSONL_PATH = os.path.join(DATA_DIR, 'wikitext_eval.jsonl')
CLEANED_JSONL_PATH = os.path.join(DATA_DIR, 'wikitext_cleaned.jsonl')

TRAIN_SAMPLES = 3000
EVAL_SAMPLES = 200
RAW_LOAD_SAMPLES = 10000
PREFIX_RATIO = 0.6
SENTENCE_END_CHARS = '.!?;\n'

BASE_MODEL_PATH = (
    '/home/mnt/path/.lazyllm/model/modelscope/qwen/'
    'Qwen2.5-0.5B-Instruct'
)
MAX_EVAL_SAMPLES = None
MAX_NEW_TOKENS = 256
GEN_BATCH_SIZE = 32
GEN_TEMPERATURE = 0.5
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.05
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _split_at_sentence_boundary(text, target_ratio=0.5):
    n = len(text)
    if not n or target_ratio <= 0 or target_ratio >= 1:
        return text[: n // 2], text[n // 2:]
    target_pos = int(n * target_ratio)
    pattern = re.compile(f'[{re.escape(SENTENCE_END_CHARS)}]')
    matches = list(pattern.finditer(text))
    if not matches:
        return text[:target_pos], text[target_pos:]
    best_pos = 0
    best_dist = abs(0 - target_pos)
    for m in matches:
        end = m.end()
        if end >= n:
            continue
        dist = abs(end - target_pos)
        if dist < best_dist:
            best_dist = dist
            best_pos = end
    prefix, continuation = text[:best_pos].strip(), text[best_pos:].strip()
    if not prefix or not continuation:
        return text[:target_pos], text[target_pos:]
    return prefix, continuation


def _load_and_process_chunks():
    print(f'正在加载 wikitext-2-raw-v1 train，取前 {RAW_LOAD_SAMPLES} 条非空数据...')
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    raw_texts = []
    for row in dataset:
        if len(raw_texts) >= RAW_LOAD_SAMPLES:
            break
        t = row.get('text', '')
        if t and isinstance(t, str) and t.strip():
            raw_texts.append(t.strip())
    print(f'加载完成: 共 {len(raw_texts)} 条非空文本')

    print('正在通过 build_text_pt_pipeline 处理...')
    ppl = build_text_pt_pipeline(
        content_key='content',
        language='en',
        min_chars=100,
        max_chars=100000,
        max_tokens=1024,
        min_tokens=100,
    )
    data = [{'content': text} for text in raw_texts]
    chunks = ppl(data)
    chunks = (
        chunks
        if isinstance(chunks, list)
        else ([] if not chunks else [chunks])
    )
    print(f'处理完成: 共 {len(chunks)} 个 chunk')

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CLEANED_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'pipeline 结果已保存: {CLEANED_JSONL_PATH}')

    return chunks


def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TRAIN_JSON_PATH) and os.path.exists(EVAL_JSONL_PATH):
        print('预训练集与评测集已存在，跳过生成')
        return

    chunks = _load_and_process_chunks()
    if not chunks:
        print('处理后无有效 chunk，退出')
        return

    train_chunks = chunks[:TRAIN_SAMPLES]
    eval_chunks = chunks[:EVAL_SAMPLES]

    if not os.path.exists(TRAIN_JSON_PATH):
        texts = [c.get('content', c.get('text', '')) for c in train_chunks]
        with open(TRAIN_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(
                [{'text': t} for t in texts],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f'预训练集已保存：{TRAIN_JSON_PATH}，共 {len(texts)} 条')

    eval_samples = []
    for chunk in eval_chunks:
        ctx = chunk.get('content', chunk.get('text', ''))
        if not ctx:
            continue
        prefix, continuation = _split_at_sentence_boundary(
            ctx, target_ratio=PREFIX_RATIO
        )
        if not prefix or not continuation:
            prefix = ctx[:len(ctx) // 2]
            continuation = ctx[len(ctx) // 2:]
        eval_samples.append({'prefix': prefix, 'continuation': continuation})

    with open(EVAL_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in eval_samples:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'评测集已保存：{EVAL_JSONL_PATH}，共 {len(eval_samples)} 条')

    print('\n=== 处理后的训练集（第 1 条）===')
    if train_chunks:
        text = train_chunks[0].get('content', train_chunks[0].get('text', ''))
        print(f'  text: {text[:400]}{"..." if len(text) > 400 else ""}')
    print('\n=== 处理后的评测集（第 1 条）===')
    if eval_samples:
        rec = eval_samples[0]
        sp = '...' if len(rec['prefix']) > 300 else ''
        sc = '...' if len(rec['continuation']) > 300 else ''
        print(f'  prefix: {rec["prefix"][:300]}{sp}')
        print(f'  continuation: {rec["continuation"][:300]}{sc}')
    print()


def run_train():
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    target_path = os.path.join(
        PRETRAIN_CKPT_DIR, f'qwen2_5_0_5b_wikitext_{timestamp}'
    )
    os.makedirs(PRETRAIN_CKPT_DIR, exist_ok=True)

    model = lazyllm.TrainableModule(BASE_MODEL_PATH, target_path=target_path)
    model.mode('finetune')\
        .trainset(TRAIN_JSON_PATH)\
        .finetune_method((finetune.llamafactory, {
            'stage': 'pt',
            'finetuning_type': 'full',
            'learning_rate': 3e-5,
            'cutoff_len': 512,
            'val_size': 0.05,
            'optim': 'adamw_torch_fused',
            'bf16': True,
            'fp16': False,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'num_train_epochs': 20,
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            'save_steps': 20,
            'logging_steps': 5,
            'resume_from_checkpoint': None,
            'save_strategy': 'steps',
            'save_total_limit': 5,
            'launcher': launchers.empty(ngpus=1),
        }))\
        .update()
    return target_path


def _find_merge_dir_under(base: str, dir_name: str = 'lazyllm_merge') -> str:
    for root, dirs, _ in os.walk(base):
        if dir_name in dirs:
            return os.path.join(root, dir_name)
    return ''


def get_latest_pt_model_path() -> str:
    if not os.path.isdir(PRETRAIN_CKPT_DIR):
        return ''
    candidates = []
    for name in os.listdir(PRETRAIN_CKPT_DIR):
        path = os.path.join(PRETRAIN_CKPT_DIR, name)
        if os.path.isdir(path):
            merge = _find_merge_dir_under(path)
            if merge and os.path.isdir(merge):
                candidates.append((os.path.getmtime(merge), merge))
    if not candidates:
        return ''
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_eval_samples(path: str, max_samples: int = None) -> List[Dict]:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prefix = (item.get('prefix') or '').strip()
            continuation = (item.get('continuation') or '').strip()
            if not prefix or not continuation:
                continue
            samples.append({'prefix': prefix, 'continuation': continuation})
            if max_samples and len(samples) >= max_samples:
                break
    print(f'加载评测集 {len(samples)} 条（prefix → continuation）')
    return samples


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    return model, tokenizer


def compute_ppl(
    model, tokenizer, samples: List[Dict]
) -> Tuple[float, float, List[float], List[float]]:
    ppl_list = []
    loss_list = []
    for item in tqdm(samples, desc='PPL/loss', unit='条'):
        prefix, continuation = item['prefix'], item['continuation']
        full_text = prefix + continuation
        enc = tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        ).to(DEVICE)
        prefix_enc = tokenizer(
            prefix,
            return_tensors='pt',
            add_special_tokens=True,
        ).to(DEVICE)
        seq_len = enc.input_ids.size(1)
        prefix_len = min(prefix_enc.input_ids.size(1), seq_len - 1)
        input_ids = enc.input_ids
        labels = input_ids.clone()
        labels[:, :prefix_len] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.item()
        ppl = torch.exp(torch.tensor(loss)).item()
        loss_list.append(loss)
        ppl_list.append(ppl)
    n = len(ppl_list) if ppl_list else 0
    avg_loss = sum(loss_list) / n if n else 0.0
    avg_ppl = torch.exp(torch.tensor(min(avg_loss, 50.0))).item() if n else 0.0
    return avg_ppl, avg_loss, ppl_list, loss_list


def _bleu_simple(ref: str, pred: str) -> float:
    def _ngrams(s, n):
        s = ''.join(s.split())
        if len(s) < n:
            return []
        return [s[i:i + n] for i in range(len(s) - n + 1)]
    r, p = ''.join(ref.split()), ''.join(pred.split())
    if not r or not p:
        return 0.0
    bigram_r = set(_ngrams(r, 2))
    bigram_p = _ngrams(p, 2)
    if not bigram_p:
        return 0.0
    hit = sum(1 for b in bigram_p if b in bigram_r)
    prec = hit / len(bigram_p)
    rec = hit / len(bigram_r) if bigram_r else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def run_generation(
    model, tokenizer, samples: List[Dict]
) -> Tuple[float, List[Dict]]:
    results = []
    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        repetition_penalty=GEN_REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    for start in tqdm(
        range(0, len(samples), GEN_BATCH_SIZE),
        desc='生成',
        unit='batch',
    ):
        batch_items = samples[start:start + GEN_BATCH_SIZE]
        prefixes = [item['prefix'] for item in batch_items]
        prefix_lens = []
        for p in prefixes:
            enc_single = tokenizer(
                p,
                truncation=True,
                max_length=2048,
                add_special_tokens=True,
            )
            prefix_lens.append(len(enc_single.input_ids))
        enc = tokenizer(
            prefixes,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
            return_attention_mask=True,
        ).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, generation_config=gen_config)
        for i, item in enumerate(batch_items):
            pred_ids = gen[i, prefix_lens[i]:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
            ref = item['continuation']
            score = _bleu_simple(ref, pred)
            results.append({
                'prefix': prefixes[i],
                'continuation_ref': ref,
                'continuation_pred': pred,
                'bleu': score,
            })
    avg_bleu = (
        sum(r['bleu'] for r in results) / len(results) if results else 0.0
    )
    return avg_bleu, results


def run_eval_for_model(
    model_path: str, label: str, samples: List[Dict]
) -> Dict:
    print(f'\n[{label}] 加载模型: {model_path}')
    model, tokenizer = load_model_and_tokenizer(model_path)

    print(f'[{label}] 计算 PPL 与交叉熵损失 ...')
    avg_ppl, avg_loss, ppl_list, loss_list = compute_ppl(
        model, tokenizer, samples
    )
    print(f'[{label}] 平均 PPL: {avg_ppl:.4f}  平均 loss: {avg_loss:.4f}')

    print(f'[{label}] 生成续写并计算 BLEU ...')
    avg_bleu, gen_results = run_generation(model, tokenizer, samples)
    print(f'[{label}] 平均 BLEU (2-gram F1): {avg_bleu:.4f}')

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return {
        'ppl': avg_ppl,
        'loss': avg_loss,
        'bleu': avg_bleu,
        'ppl_per_sample': ppl_list,
        'loss_per_sample': loss_list,
        'gen_results': gen_results,
    }


def run_eval(pt_model_path: str = None, out_run_dir: str = None) -> str:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if not os.path.exists(EVAL_JSONL_PATH):
        print(f'评测集不存在: {EVAL_JSONL_PATH}，请先运行 prepare 或 full')
        return ''

    samples = load_eval_samples(EVAL_JSONL_PATH, max_samples=MAX_EVAL_SAMPLES)
    if not samples:
        print('无有效样本')
        return ''

    base_metrics = run_eval_for_model(BASE_MODEL_PATH, '基座模型', samples)

    pt_metrics = None
    if pt_model_path and os.path.isdir(pt_model_path):
        pt_metrics = run_eval_for_model(pt_model_path, '预训练模型', samples)
    else:
        if pt_model_path:
            print(f'预训练模型路径不存在: {pt_model_path}，仅保存基座评测结果')
        else:
            print('未指定预训练模型路径，仅保存基座评测结果')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = out_run_dir or os.path.join(RESULTS_DIR, ts)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, 'eval_pt_wikitext_metrics.json')
    results_path = os.path.join(run_dir, 'eval_pt_wikitext_results.jsonl')

    metrics = {
        'num_samples': len(samples),
        'base': {
            'ppl': base_metrics['ppl'],
            'loss': base_metrics['loss'],
            'bleu': base_metrics['bleu'],
        },
        'pt': (
            {
                'ppl': pt_metrics['ppl'],
                'loss': pt_metrics['loss'],
                'bleu': pt_metrics['bleu'],
            }
            if pt_metrics
            else None
        ),
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(results_path, 'w', encoding='utf-8') as f:
        for i in range(len(samples)):
            rec = {
                'id': i + 1,
                'prefix_len': len(samples[i]['prefix']),
                'continuation_len': len(samples[i]['continuation']),
                'base_ppl': base_metrics['ppl_per_sample'][i],
                'base_loss': base_metrics['loss_per_sample'][i],
                'base_pred': (
                    base_metrics['gen_results'][i]['continuation_pred'][:200]
                ),
                'base_bleu': base_metrics['gen_results'][i]['bleu'],
            }
            if pt_metrics:
                rec['pt_ppl'] = pt_metrics['ppl_per_sample'][i]
                rec['pt_loss'] = pt_metrics['loss_per_sample'][i]
                rec['pt_pred'] = (
                    pt_metrics['gen_results'][i]['continuation_pred'][:200]
                )
                rec['pt_bleu'] = pt_metrics['gen_results'][i]['bleu']
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'\n评测结果已保存至: {run_dir}')
    print(f'  指标: {metrics_path}')
    print(f'  逐条结果: {results_path}')
    print('\n' + '=' * 60)
    print('Wikitext 预训练评测（prefix → continuation）')
    print('=' * 60)
    print(
        f'基座模型   PPL={base_metrics["ppl"]:.4f}  '
        f'loss={base_metrics["loss"]:.4f}  '
        f'BLEU={base_metrics["bleu"]:.4f}'
    )
    if pt_metrics:
        print(
            f'预训练模型 PPL={pt_metrics["ppl"]:.4f}  '
            f'loss={pt_metrics["loss"]:.4f}  '
            f'BLEU={pt_metrics["bleu"]:.4f}'
        )
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Wikitext 预训练：数据准备、训练、评测一体化')
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['prepare', 'train', 'eval', 'full'],
        help='prepare=仅准备数据; train=准备+训练; eval=仅评测; full=准备+训练+评测',
    )
    parser.add_argument(
        '--pt_model_path',
        type=str,
        default=None,
        help='预训练模型目录（eval/full 时使用；full 未指定则用本次训练 ckpt）',
    )
    args = parser.parse_args()

    if args.mode == 'prepare':
        prepare_dataset()
        return

    target_path = ''
    if args.mode in ('train', 'full'):
        prepare_dataset()
        target_path = run_train()
        print(f'训练输出目录: {target_path}')

    pt_path = args.pt_model_path
    if args.mode == 'full' and not pt_path and target_path:
        pt_path = _find_merge_dir_under(target_path)
        if pt_path:
            print(f'使用本次训练合并模型: {pt_path}')
    if not pt_path and args.mode == 'eval':
        pt_path = get_latest_pt_model_path()
        if pt_path:
            print(f'使用最新预训练模型: {pt_path}')
    if args.mode in ('eval', 'full'):
        run_eval(pt_model_path=pt_path)


# 使用示例：
# 1. 仅准备数据：     python run_wikitext.py --mode=prepare
# 2. 准备 + 训练：     python run_wikitext.py --mode=train
# 3. 仅评测：         python run_wikitext.py --mode=eval
#                    [--pt_model_path=path/to/lazyllm_merge]
# 4. 一步到位：       python run_wikitext.py --mode=full
if __name__ == '__main__':
    main()
