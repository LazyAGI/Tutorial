from lazyllm.tools.data.prompts.domain_finetune import DOMAIN_PRESETS
from lazyllm.tools.data.pipelines.domain_pretrain_pipelines import (
    DOMAIN_PRETRAIN_FEATURES,
    build_text_pt_plus_domain_pretrain_pipeline,
)
from lazyllm import pipeline
from lazyllm import LOG, finetune, launchers
import lazyllm
import os
import re
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Ensure LazyLLM is in the path
_LAZYLLM_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../LazyLLM'
))
if _LAZYLLM_ROOT not in sys.path:
    sys.path.insert(0, _LAZYLLM_ROOT)


try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

FINANCE_DATASET_NAME = 'ashraq/financial-news-articles'
FINANCE_CONTENT_KEY = 'content'

FINANCE_KEYWORDS = [

    'stock', 'stocks', 'equity', 'equities', 'bond', 'bonds', 'fund', 'funds',
    'et', 'derivative', 'derivatives', 'futures', 'options', 'swap',
    'share', 'shares', 'index', 'indices', 'benchmark',

    'earnings', 'revenue', 'profit', 'loss', 'dividend', 'cash flow',
    'balance sheet', 'income statement', 'valuation',
    'ipo', 'merger', 'acquisition', 'm&a', 'spin-of',

    'inflation', 'gdp', 'interest rate', 'rates', 'central bank',
    'fed', 'ecb', 'monetary policy', 'fiscal policy',
    'unemployment', 'economic growth', 'recession',

    'investor', 'investors', 'trader', 'traders', 'analyst', 'analysts',
    'portfolio', 'hedge fund', 'asset manager', 'broker',

    'currency', 'currencies', 'forex', 'fx', 'exchange rate', 'usd',
    'eur', 'jpy', 'cny',
]


def resolve_domain_keywords(
    domain: str,
    language: str,
    domain_keywords: Optional[List[str]],
) -> Optional[List[str]]:
    if domain_keywords is not None:
        return domain_keywords
    if domain == 'finance' and language == 'en':
        return list(FINANCE_KEYWORDS)
    preset = DOMAIN_PRESETS.get(domain) or DOMAIN_PRESETS.get('general', {})
    kw = list(preset.get('pretrain_keywords') or [])
    return kw if kw else None


PREFIX_RATIO = 0.5

MAX_EVAL_SAMPLES_DEFAULT = None
EVAL_MAX_LENGTH = 2048
EVAL_MAX_NEW_TOKENS = 256
EVAL_GEN_BATCH_SIZE = 32
EVAL_GEN_TEMPERATURE = 0.5
EVAL_GEN_TOP_P = 0.9
EVAL_GEN_REPETITION_PENALTY = 1.05


def _split_at_sentence_boundary(
    text: str,
    target_ratio: float = PREFIX_RATIO,
    language: str = 'en',
) -> Tuple[str, str]:
    text = (text or '').strip()
    if not text:
        return '', ''
    use_zh_boundary = (
        language == 'zh' or
        bool(re.search(r'[\u4e00-\u9fff]', text[: min(500, len(text))]))
    )
    if use_zh_boundary:
        parts = [p for p in re.split(r'(?<=[。！？!?])\s*', text) if p.strip()]
        if len(parts) <= 1:
            parts = [p for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
    else:
        parts = [p for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
    if len(parts) <= 1:
        mid = max(1, int(len(text) * target_ratio))
        return text[:mid], text[mid:].strip()
    total = len(text)
    acc = 0
    i = 0
    for i, p in enumerate(parts):
        acc += len(p) + (1 if i < len(parts) - 1 else 0)
        if acc >= total * target_ratio:
            break
    i = min(i, len(parts) - 1)
    if use_zh_boundary:
        prefix = ''.join(parts[: i + 1])
        continuation = ''.join(parts[i + 1:]).strip()
    else:
        prefix = ' '.join(parts[: i + 1])
        continuation = ' '.join(parts[i + 1:]).strip()
    return prefix, continuation

# ---------------------------------------------------------------------------


def load_text_data(
    dataset_name: str = FINANCE_DATASET_NAME,
    split: str = 'train',
    max_samples: Optional[int] = None,
    content_key: str = FINANCE_CONTENT_KEY,
    local_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if local_path:
        LOG.info(f'正在从本地加载数据: {local_path}')
        items = []
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        LOG.warning(f'跳过无效 JSON 行: {line[:100]}...')
        except Exception as e:
            raise RuntimeError(f'加载本地文件 {local_path} 失败: {e}')

        LOG.info(f'从本地加载完成: {len(items)} 条记录')
        return items

    # 从 HuggingFace 加载
    if load_dataset is None:
        raise ImportError('请先安装 datasets：pip install datasets')

    LOG.info(f'正在从 HuggingFace 加载 {dataset_name} [{split}]...')
    try:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f'加载 {dataset_name} 失败: {e}\n'
            '可能需要设置 HF_ENDPOINT 或确认数据集名称正确。'
        )

    raw_n = len(ds)
    if max_samples and raw_n > max_samples:
        ds = ds.select(range(max_samples))
        LOG.info(f'截取前 {max_samples} 条记录（原始 {raw_n} 条）')

    items = [dict(row) for row in ds]
    if max_samples:
        items = items[:max_samples]

    LOG.info(f'加载完成：{len(items)} 条记录')
    return items

# ---------------------------------------------------------------------------


def build_pretrain_dataset(
    raw_items: List[Dict[str, Any]],
    domain: str = 'finance',
    content_key: str = FINANCE_CONTENT_KEY,
    output_dir: str = 'dataset/financial_pt',
    language: str = 'en',
    domain_keywords: Optional[List[str]] = None,
    enabled: Optional[Dict[str, bool]] = None,
    options: Optional[Dict[str, Any]] = None,
    eval_seed: int = 42,
    eval_ratio: float = 0.02,
    eval_from_train: bool = True,
) -> Dict[str, Any]:
    if not raw_items:
        raise ValueError('raw_items 为空，请先加载数据集')

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'train.json')
    eval_file = os.path.join(output_dir, 'eval.jsonl')

    _sep = '=' * 60
    print(f'\n{_sep}')
    print(f'领域预训练数据构建（domain={domain}, language={language}）')
    print(_sep)
    print(f'  输入条数    : {len(raw_items)}')
    print(f'  文本字段    : {content_key}')
    print(f'  输出目录    : {output_dir}')
    print(_sep)

    domain_keywords = resolve_domain_keywords(
        domain, language, domain_keywords
    )

    def _chunk_plain_text(item: Dict[str, Any]) -> str:
        return (item.get('content') or item.get(content_key) or '').strip()

    core_pipeline = build_text_pt_plus_domain_pretrain_pipeline(
        domain=domain,
        content_key=content_key,
        language=language,
        enabled=enabled,
        options=options,
        domain_keywords=domain_keywords,
    )

    def _save_chunks(chunked_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunked_items:
            LOG.warning(
                '分块后无数据。常见原因：① 每条样本不足 min_chars/min_words'
                '（中文默认已放宽）；② JSON 无 text/content 等正文字段，'
                '需 --field_mapping；③ 领域 N-gram/关键词过滤过严。'
            )
            return {'chunks': [], 'stats': {'total_chunks': 0}}

        total_chunks = len(chunked_items)
        avg_chars = sum(len(_chunk_plain_text(item))
                        for item in chunked_items) / max(total_chunks, 1)
        stats = {
            'total_chunks': total_chunks,
            'avg_chars_per_chunk': round(avg_chars, 1),
        }
        if '_keyword_hits' in chunked_items[0]:
            avg_hits = sum(item.get('_keyword_hits', 0)
                           for item in chunked_items) / total_chunks
            stats['avg_keyword_hits'] = round(avg_hits, 2)
        if '_ngram_repetition_ratio' in chunked_items[0]:
            avg_rep = sum(item.get('_ngram_repetition_ratio', 0)
                          for item in chunked_items) / total_chunks
            stats['avg_ngram_repetition_ratio'] = round(avg_rep, 4)
        return {'chunks': chunked_items, 'stats': stats}

    with pipeline() as ppl:
        ppl.process = core_pipeline
        ppl.save = _save_chunks

    result = ppl(raw_items)

    chunks = result.get('chunks', [])
    stats = result.get('stats', {})
    total_chunks = len(chunks)

    rng = random.Random(eval_seed)
    eval_ratio = float(eval_ratio)
    if eval_ratio <= 0:
        n_eval = 0
    else:
        n_eval = max(1, int(total_chunks * eval_ratio))
        n_eval = min(n_eval, max(1, int(total_chunks * 0.4)))

    if not eval_from_train:
        shuffled = list(range(total_chunks))
        rng.shuffle(shuffled)
        eval_idx = set(shuffled[:n_eval])
        train_chunks = [chunks[i]
                        for i in range(total_chunks) if i not in eval_idx]
        eval_chunks = [chunks[i] for i in shuffled[:n_eval]]
    else:

        train_chunks = list(chunks)
        eval_chunks = []

    train_contexts = [_chunk_plain_text(item) for item in train_chunks]
    train_contexts = [c for c in train_contexts if c]
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump([{'text': c} for c in train_contexts],
                  f, ensure_ascii=False, indent=2)
    print(f'预训练集已写入：{train_file}，共 {len(train_contexts)} 条')

    eval_samples = []
    if n_eval > 0:
        if eval_from_train:

            idxs = list(range(len(train_contexts)))
            rng.shuffle(idxs)
            idxs = idxs[: min(n_eval, len(idxs))]
            for i in idxs:
                ctx = train_contexts[i]
                prefix, continuation = _split_at_sentence_boundary(
                    ctx, target_ratio=PREFIX_RATIO, language=language
                )
                if prefix and continuation:
                    eval_samples.append(
                        {'prefix': prefix, 'continuation': continuation})
        else:
            for chunk in eval_chunks:
                ctx = _chunk_plain_text(chunk)
                prefix, continuation = _split_at_sentence_boundary(
                    ctx, target_ratio=PREFIX_RATIO, language=language
                )
                if prefix and continuation:
                    eval_samples.append(
                        {'prefix': prefix, 'continuation': continuation})
    with open(eval_file, 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    LOG.info(f'测试集已保存: {eval_file}（{len(eval_samples)} 条，prefix+continuation）')

    if eval_from_train:
        msg = f'\n  处理后分块数: {total_chunks} 个（训练 {len(train_chunks)}，'
        msg += f'测试来自训练集抽样 {len(eval_samples)} 条）'
        print(msg)
    else:
        msg = f'\n  处理后分块数: {total_chunks} 个（训练 {len(train_chunks)}，'
        msg += f'测试 {len(eval_chunks)}）'
        print(msg)
    print(f'  平均字符数 : {stats.get("avg_chars_per_chunk", 0):.1f}')
    if 'avg_keyword_hits' in stats:
        print(f'  平均关键词命中: {stats.get("avg_keyword_hits", 0):.2f}')
    if 'avg_ngram_repetition_ratio' in stats:
        print(
            f'  平均N-gram重复率: {stats.get("avg_ngram_repetition_ratio", 0):.4f}')

    print('\n--- 样例分块（前 3 个）---')
    if chunks:
        for i, chunk in enumerate(chunks[:3], 1):
            print(f'\n[分块 {i}]')
            ct = _chunk_plain_text(chunk)
            text = ct[:200] + '...' if len(ct) > 200 else ct
            print(f'文本: {text}')
            meta = {k: v for k, v in chunk.items() if k.startswith('_')}
            if meta:
                print(f'元数据: {meta}')
    else:
        print('（无分块样例，请检查过滤条件是否过严）')
    print('---\n')

    return {
        'output_dir': output_dir,
        'train_file': train_file,
        'eval_file': eval_file,
        'total_chunks': total_chunks,
        'train_count': len(train_chunks),
        'eval_count': (
            len(eval_samples) if eval_from_train else len(eval_chunks)
        ),
        'stats': stats,
    }


def evaluate_chunk_quality(
    chunk_file: str,
    max_samples: int = 100,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:

    if not os.path.isfile(chunk_file):
        LOG.warning(f'分块文件不存在: {chunk_file}')
        return {}
    try:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        LOG.warning(f'无法解析 JSON: {chunk_file}，{e}')
        return {}
    if not isinstance(data, list) or not data:
        LOG.warning(f'JSON 数据为空或不是 list: {chunk_file}')
        return {}
    rows = data[:max_samples] if max_samples else data
    texts = [(obj.get('text') or '').strip()
             for obj in rows if isinstance(obj, dict)]
    texts = [t for t in texts if t]
    if not texts:
        LOG.warning(f'JSON 中未找到 text 字段: {chunk_file}')
        return {}
    stats = {
        'total_chunks_sampled': len(texts),
        'avg_length_chars': sum(len(t) for t in texts) / len(texts),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stats_file = os.path.join(output_dir, 'chunk_quality_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        LOG.info(f'分块质量统计已保存: {stats_file}')

    print(f'\n{"=" * 60}')
    print('分块质量评估')
    print(f'{"=" * 60}')
    print(f'  采样分块数: {stats["total_chunks_sampled"]}')
    print(f'  平均字符数: {stats["avg_length_chars"]:.1f}')
    if 'avg_keyword_hits' in stats:
        print(f'  平均关键词命中: {stats["avg_keyword_hits"]:.2f}')
    if 'avg_repetition_ratio' in stats:
        print(f'  平均重复率: {stats["avg_repetition_ratio"]:.4f}')
    print(f'{"=" * 60}\n')

    return stats

# ---------------------------------------------------------------------------


def _get_eval_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def load_eval_samples(
        path: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:

    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            prefix = (item.get('prefix') or '').strip()
            continuation = (item.get('continuation') or '').strip()
            if not prefix or not continuation:
                continue
            samples.append({'prefix': prefix, 'continuation': continuation})
            if max_samples and len(samples) >= max_samples:
                break
    LOG.info(f'加载评测集 {len(samples)} 条（prefix → continuation）')
    return samples


def load_model_and_tokenizer(model_path: str):
    try:
        from lazyllm import thirdparty
        import torch
    except ImportError as e:
        raise ImportError('评估需要 transformers 与 torch，请先安装') from e
    path = model_path.rstrip('/\\')
    device = _get_eval_device()
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    tokenizer = thirdparty.transformers.AutoTokenizer.from_pretrained(
        path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = thirdparty.transformers.AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=dtype
    ).to(device)
    model.eval()
    return model, tokenizer


def _word_ngrams(s: str, n: int = 2) -> List[Tuple[str, ...]]:
    tokens = s.split()
    if len(tokens) < n:
        return []
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _bleu2_f1(reference: str, hypothesis: str) -> float:
    ref_ng = _word_ngrams(reference, 2)
    hyp_ng = _word_ngrams(hypothesis, 2)
    if not hyp_ng:
        return 0.0
    ref_set = set(ref_ng)
    match = sum(1 for g in hyp_ng if g in ref_set)
    prec = match / len(hyp_ng)
    rec = (match / len(ref_ng)) if ref_ng else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def compute_ppl(
    model,
    tokenizer,
    samples: List[Dict[str, str]],
    max_length: int = EVAL_MAX_LENGTH,
    device: Optional[str] = None,
    mode: str = 'conditional',
) -> Tuple[float, float, List[float], List[float]]:
    import torch
    dev = device or _get_eval_device()
    ppl_list: List[float] = []
    loss_list: List[float] = []
    total_tokens = 0
    total_loss = 0.0

    for item in tqdm(samples, desc='PPL/loss', unit='条'):
        prefix, continuation = item['prefix'], item['continuation']
        full_text = prefix + continuation

        enc = tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).to(dev)

        input_ids = enc.input_ids
        seq_len = input_ids.size(1)

        if mode == 'conditional':

            prefix_enc = tokenizer(
                prefix, return_tensors='pt', add_special_tokens=True).to(dev)
            prefix_len = min(prefix_enc.input_ids.size(1), seq_len - 1)
            labels = input_ids.clone()
            labels[:, :prefix_len] = -100
            num_tokens = seq_len - prefix_len
        else:

            labels = input_ids.clone()
            labels[:, 0] = -100
            num_tokens = seq_len - 1

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens

        ppl = float(torch.exp(torch.tensor(min(loss, 50.0))).item())
        loss_list.append(loss)
        ppl_list.append(ppl)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_ppl = float(torch.exp(torch.tensor(min(avg_loss, 50.0))).item())

    return avg_ppl, avg_loss, ppl_list, loss_list


def run_generation(
    model,
    tokenizer,
    samples: List[Dict[str, str]],
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    batch_size: int = EVAL_GEN_BATCH_SIZE,
    temperature: float = EVAL_GEN_TEMPERATURE,
    top_p: float = EVAL_GEN_TOP_P,
    repetition_penalty: float = EVAL_GEN_REPETITION_PENALTY,
    max_length: int = EVAL_MAX_LENGTH,
    device: Optional[str] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    from lazyllm import thirdparty
    import torch
    dev = device or _get_eval_device()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gen_config = thirdparty.transformers.GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_id,
    )
    results: List[Dict[str, Any]] = []
    for start in tqdm(range(0, len(samples), batch_size),
                      desc='生成', unit='batch'):
        batch_items = samples[start: start + batch_size]
        prefixes = [item['prefix'] for item in batch_items]
        prefix_lens = []
        for p in prefixes:
            enc_single = tokenizer(
                p,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            prefix_lens.append(len(enc_single.input_ids))
        enc = tokenizer(
            prefixes,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_attention_mask=True,
        ).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, generation_config=gen_config)
        for i, item in enumerate(batch_items):
            pred_ids = gen[i, prefix_lens[i]:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
            ref = item['continuation']
            score = _bleu2_f1(ref, pred)
            results.append({
                'prefix': prefixes[i],
                'continuation_re': ref,
                'continuation_pred': pred,
                'bleu': score,
            })
    avg_bleu = sum(r['bleu'] for r in results) / \
        len(results) if results else 0.0
    return avg_bleu, results


def run_eval_for_model(
    model_path: str,
    label: str,
    samples: List[Dict[str, str]],
    max_length: int = EVAL_MAX_LENGTH,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    gen_batch_size: int = EVAL_GEN_BATCH_SIZE,
    ppl_mode: str = 'unconditional',
) -> Dict[str, Any]:
    import torch
    LOG.info(f'[{label}] 加载模型: {model_path}')
    model, tokenizer = load_model_and_tokenizer(model_path)
    device = _get_eval_device()

    LOG.info(f'[{label}] 计算 PPL 与交叉熵损失（模式: {ppl_mode}）...')
    avg_ppl, avg_loss, ppl_list, loss_list = compute_ppl(
        model, tokenizer, samples,
        max_length=max_length, device=device, mode=ppl_mode
    )
    LOG.info(f'[{label}] 平均 PPL: {avg_ppl:.4f}  平均 loss: {avg_loss:.4f}')

    LOG.info(f'[{label}] 生成续写并计算 BLEU-2 F1 ...')
    avg_bleu = 0.0
    gen_results = []
    avg_bleu, gen_results = run_generation(
        model, tokenizer, samples,
        max_new_tokens=max_new_tokens,
        batch_size=gen_batch_size,
        max_length=max_length,
        device=device,
    )
    LOG.info(f'[{label}] 平均 BLEU-2 F1: {avg_bleu:.4f}')

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'ppl': avg_ppl,
        'loss': avg_loss,
        'bleu': avg_bleu,
        'ppl_per_sample': ppl_list,
        'loss_per_sample': loss_list,
        'gen_results': gen_results,
    }


def evaluate_pretrained_model(
    base_model_path: str,
    pretrained_model_path: Optional[str],
    eval_jsonl_path: str,
    output_dir: Optional[str] = None,
    max_eval_samples: Optional[int] = MAX_EVAL_SAMPLES_DEFAULT,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    ppl_max_length: int = EVAL_MAX_LENGTH,
    gen_batch_size: int = EVAL_GEN_BATCH_SIZE,
    ppl_mode: str = 'unconditional',
) -> Dict[str, Any]:
    base_model_path = base_model_path.rstrip('/\\')
    if not os.path.isfile(eval_jsonl_path):
        LOG.warning(f'评测集不存在: {eval_jsonl_path}')
        return {}

    samples = load_eval_samples(eval_jsonl_path, max_samples=max_eval_samples)
    if not samples:
        LOG.warning('无有效 prefix/continuation 样本')
        return {}

    random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass

    print(f'\n>>> PPL 计算模式: {ppl_mode}')
    print('    - conditional: 条件 PPL（仅评估 continuation 部分）')
    print('    - unconditional: 无条件 PPL（评估整个文本，推荐用于预训练评测）\n')

    base_metrics = run_eval_for_model(
        base_model_path, '基座模型', samples,
        max_length=ppl_max_length,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
        ppl_mode=ppl_mode,
    )
    pt_metrics: Optional[Dict[str, Any]] = None
    if pretrained_model_path and os.path.isdir(
            pretrained_model_path.rstrip('/\\')):
        pt_metrics = run_eval_for_model(
            pretrained_model_path.rstrip('/\\'), '预训练模型', samples,
            max_length=ppl_max_length,
            max_new_tokens=max_new_tokens,
            gen_batch_size=gen_batch_size,
            ppl_mode=ppl_mode,
        )
    else:
        LOG.warning(f'预训练模型路径不存在或未指定: {pretrained_model_path}，仅保存基座评测结果')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir or '.', 'eval_runs', ts)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, 'eval_pt_metrics.json')
    results_path = os.path.join(run_dir, 'eval_pt_results.jsonl')

    metrics = {
        'num_samples': len(samples),
        'base': {
            'ppl': base_metrics['ppl'],
            'loss': base_metrics['loss'],
            'bleu': base_metrics['bleu']
        },
        'pt': (
            {'ppl': pt_metrics['ppl'], 'loss': pt_metrics['loss'],
                'bleu': pt_metrics['bleu']}
            if pt_metrics else None
        ),
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    LOG.info(f'评测结果已保存至: {run_dir}')
    LOG.info(f'  指标: {metrics_path}')

    with open(results_path, 'w', encoding='utf-8') as f:
        for i in range(len(samples)):
            rec = {
                'id': i + 1,
                'prefix_len': len(samples[i]['prefix']),
                'continuation_len': len(samples[i]['continuation']),
                'base_ppl': base_metrics['ppl_per_sample'][i],
                'base_loss': base_metrics['loss_per_sample'][i],
                'base_pred': (
                    base_metrics['gen_results'][i]['continuation_pred'] or ''
                )[:200],
                'base_bleu': base_metrics['gen_results'][i]['bleu'],
            }
            if pt_metrics:
                rec['pt_ppl'] = pt_metrics['ppl_per_sample'][i]
                rec['pt_loss'] = pt_metrics['loss_per_sample'][i]
                rec['pt_pred'] = (pt_metrics['gen_results'][i]
                                  ['continuation_pred'] or '')[:200]
                rec['pt_bleu'] = pt_metrics['gen_results'][i]['bleu']
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    LOG.info(f'  逐条结果: {results_path}')

    print('\n' + '=' * 60)
    print('预训练评测（prefix → continuation）')
    print('=' * 60)
    msg = f'基座模型   PPL={base_metrics["ppl"]:.4f}  '
    msg += f'loss={base_metrics["loss"]:.4f}  '
    msg += f'BLEU-2 F1={base_metrics["bleu"]:.4f}'
    print(msg)
    if pt_metrics:
        msg = f'预训练模型 PPL={pt_metrics["ppl"]:.4f}  '
        msg += f'loss={pt_metrics["loss"]:.4f}  '
        msg += f'BLEU-2 F1={pt_metrics["bleu"]:.4f}'
        print(msg)
    print('=' * 60 + '\n')

    return {
        'metrics': metrics,
        'run_dir': run_dir,
        'metrics_path': metrics_path,
        'results_path': results_path,
        'base_metrics': base_metrics,
        'pt_metrics': pt_metrics,
    }

# ---------------------------------------------------------------------------


def run_pretrain(
    pretrain_data_path: str,
    base_model: str,
    train_target_path: str = 'financial_pt',
    num_epochs: int = 1,
    per_device_batch_size: int = 4,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 4,
    cutoff_len: int = 2048,
    warmup_ratio: float = 0.05,
    ngpus: int = 1,
    save_steps: int = 100,
    logging_steps: int = 10,
    save_total_limit: int = 3,
    train_launcher: str = 'remote',
    sco_partition: str = 'a800',
    sco_resource: str = 'N3lS.1i.160.1',
):

    print(f'\n{"=" * 60}')
    print('开始 LLM 预训练')
    print(f'  基座模型: {base_model}')
    print(f'  训练数据: {pretrain_data_path}')
    print(f'  launcher: {train_launcher}')
    print(f'{"=" * 60}\n')

    if train_launcher == 'sco':
        launcher = launchers.sco(
            ngpus=ngpus, partition=sco_partition, resource=sco_resource
        )
    else:
        launcher = launchers.remote(ngpus=max(1, ngpus), sync=True)

    model = lazyllm.TrainableModule(base_model, target_path=train_target_path)
    model = model.mode('finetune')\
        .trainset(pretrain_data_path)\
        .finetune_method((finetune.llamafactory, {
            'stage': 'pt',
            'finetuning_type': 'full',
            'learning_rate': learning_rate,
            'cutoff_len': cutoff_len,
            'val_size': 0.1,
            'optim': 'adamw_torch_fused',
            'bf16': True,
            'fp16': False,
            'per_device_train_batch_size': per_device_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'num_train_epochs': num_epochs,
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': warmup_ratio,
            'save_steps': save_steps,
            'logging_steps': logging_steps,
            'resume_from_checkpoint': None,
            'save_strategy': 'steps',
            'save_total_limit': save_total_limit,
            'launcher': launcher,
        }))\
        .update()
    print('预训练完成！')
    return model

# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='金融领域预训练 Pipeline（LazyLLM）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用环境前可执行: source activate_lazy.sh
  python medical_domain_pt_ppl.py --build_dataset --max_samples 1000
  python medical_domain_pt_ppl.py --build_dataset
      --local_path /path/to/data.jsonl
  python medical_domain_pt_ppl.py --train_flag
  python medical_domain_pt_ppl.py --eval_quality --max_samples 50
  python medical_domain_pt_ppl.py --eval_model
  python medical_domain_pt_ppl.py --eval_model
      --pretrained_model /path/to/pretrained
        """,
    )

    data = parser.add_argument_group('数据参数')
    data.add_argument('--build_dataset', action='store_true', help='构建预训练数据集')
    data.add_argument('--dataset_name', type=str,
                      default=FINANCE_DATASET_NAME, help='HuggingFace 数据集名称')
    data.add_argument('--local_path', type=str, default=None,
                      help='本地 JSONL 文件路径（优先使用）')
    data.add_argument('--split', type=str, default='train')
    data.add_argument('--max_samples', type=int,
                      default=None, help='最多加载条数，None=全量')
    data.add_argument('--content_key', type=str,
                      default=FINANCE_CONTENT_KEY, help='文本字段 key')
    data.add_argument('--output_dir', type=str,
                      default='/home/mnt/zhangkejun/work/dataset/financial_pt')
    data.add_argument('--domain', type=str, default='finance',
                      help='领域：medical/finance/...，与预设关键词一致')
    data.add_argument('--language', type=str, default='en',
                      choices=['zh', 'en'], help='语料与 Pipeline 语言')
    data.add_argument('--eval_seed', type=int,
                      default=42, help='划分 eval 子集时的随机种子')
    data.add_argument('--eval_ratio', type=float, default=0.02,
                      help='eval 占比（默认 0.02；上限会被限制到 0.05；'
                           '<=0 表示不生成 eval）')
    data.add_argument('--eval_from_train', action='store_true',
                      help='将 eval 从训练集里抽样生成（与 train.json 重叠；'
                           '更偏拟合/过拟合检查）')

    ppl = parser.add_argument_group('Pipeline 参数')
    ppl.add_argument('--enabled_features', type=str, nargs='+',
                     help='启用的功能，可选: ' + ', '.join(DOMAIN_PRETRAIN_FEATURES))
    ppl.add_argument('--domain_keywords', type=str,
                     nargs='+', default=None, help='自定义领域关键词')

    field_norm = parser.add_argument_group('字段归一化参数')
    field_norm.add_argument('--field_mapping', type=str, nargs='+',
                            default=None,
                            help='显式字段映射，格式: src1=dst1 src2=dst2')
    field_norm.add_argument('--concat_fields', type=str, nargs='+',
                            default=None,
                            help='需要拼接到 content_key 的字段列表')
    field_norm.add_argument('--concat_separator', type=str, default='\n\n',
                            help='多字段拼接分隔符')

    # 常用选项示例
    ppl.add_argument('--enable_language_filter',
                     action='store_true', help='启用语言过滤')
    ppl.add_argument('--enable_domain_relevance',
                     action='store_true', help='启用领域相关性评分')
    ppl.add_argument('--min_relevance_score', type=float, default=0.1,
                     help='领域相关性最低得分')
    ppl.add_argument('--max_tokens', type=int,
                     default=1024, help='分块最大 token 数')
    ppl.add_argument('--min_tokens', type=int,
                     default=200, help='分块最小 token 数')
    ppl.add_argument('--ngram_n', type=int, default=20,
                     help='NGramRepetitionFilter 中的 n')
    ppl.add_argument('--max_repetition_ratio', type=float,
                     default=0.4, help='NGram 重复度最大比例')

    model = parser.add_argument_group('模型参数')
    model.add_argument('--base_model', type=str,
                       default='Qwen2.5-0.5B-Instruct')
    model.add_argument('--train_flag', action='store_true', help='执行预训练')

    eval_group = parser.add_argument_group('质量评估参数')
    eval_group.add_argument(
        '--eval_quality', action='store_true', help='分块数据质量统计')
    eval_group.add_argument('--max_eval_samples', type=int, default=100)

    eval_model_group = parser.add_argument_group('预训练模型评估（困惑度+文本补全）')
    eval_model_group.add_argument(
        '--eval_model', action='store_true',
        help='对基座与预训练模型做 PPL+BLEU-2 F1 对比评估')
    eval_model_group.add_argument(
        '--pretrained_model', type=str, default=None,
        help='预训练模型目录，默认 output_dir/pretrained_model')
    eval_model_group.add_argument(
        '--eval_chunk_file', type=str, default=None,
        help='评测集 JSONL（prefix+continuation），默认 output_dir/eval.jsonl')
    eval_model_group.add_argument(
        '--eval_max_ppl_samples', type=int, default=500,
        help='评测最多样本数，0 表示不限制')
    eval_model_group.add_argument(
        '--eval_max_new_tokens', type=int, default=256, help='生成续写最大 token 数')
    eval_model_group.add_argument(
        '--ppl_mode', type=str, default='unconditional',
        choices=['conditional', 'unconditional'],
        help='PPL 计算模式：conditional（条件PPL，仅continuation） '
             '或 unconditional（整个文本，推荐）'
    )

    train = parser.add_argument_group('训练参数')
    train.add_argument('--num_epochs', type=int, default=10)
    train.add_argument('--per_device_batch_size', type=int, default=4)
    train.add_argument('--learning_rate', type=float, default=6e-6)
    train.add_argument('--gradient_accumulation_steps', type=int, default=8)
    train.add_argument('--cutoff_len', type=int, default=2048)
    train.add_argument('--warmup_ratio', type=float, default=0.05)
    train.add_argument('--ngpus', type=int, default=1)
    train.add_argument('--train_target_path', type=str,
                       default='./models/financial_pt',
                       help='TrainableModule 输出子目录名')
    train.add_argument('--train_launcher', type=str, default='sco',
                       choices=['remote', 'sco'], help='launcher 类型')
    train.add_argument('--sco_partition', type=str, default='a800')
    train.add_argument('--sco_resource', type=str, default='N3lS.1i.160.1')
    train.add_argument('--save_steps', type=int, default=100)
    train.add_argument('--save_total_limit', type=int, default=3)
    train.add_argument('--train_logging_steps', type=int, default=10)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    print(f'\n{"=" * 80}')
    print(' ' * 10 + '领域预训练 Pipeline（LazyLLM）')
    print(f'{"=" * 80}\n')

    # 构建 enabled 配置（功能开关；与 DOMAIN_PRETRAIN_FEATURES 对应）
    enabled = None
    if args.enabled_features:
        enabled = {feature: True for feature in args.enabled_features}

    # 构建 options 配置（仅放各阶段的数值/路径等参数，开关放在 enabled 中）
    options = {}

    # 字段归一化参数
    if args.field_mapping:
        fm = {}
        for pair in args.field_mapping:
            if '=' in pair:
                src, dst = pair.split('=', 1)
                fm[src.strip()] = dst.strip()
        if fm:
            options['field_mapping'] = fm
    if args.concat_fields:
        options['concat_fields'] = args.concat_fields
    if args.concat_separator != '\n\n':
        options['concat_separator'] = args.concat_separator

    if args.enable_domain_relevance:
        options['min_relevance_score'] = getattr(
            args, 'min_relevance_score', 0.1)
    options['max_tokens'] = args.max_tokens
    options['min_tokens'] = args.min_tokens
    # 放宽 NGram 重复过滤条件：更大的 n、更高的允许重复比例
    options['ngram_n'] = getattr(args, 'ngram_n', 20)
    options['max_repetition_ratio'] = getattr(
        args, 'max_repetition_ratio', 0.6)

    # 默认启用垂直领域增强功能；通用 PT 功能由 build_text_pt_pipeline 自动处理
    if enabled is None:
        enabled = {
            'field_normalization': True,
            'text_normalization': True,
            'sensitive_info_cleaning': True,
            'language_filter': True,
            'domain_keyword_filter': False,
            'domain_relevance_scorer': True,
            'ngram_repetition_filter': True,
        }

    train_file = os.path.join(args.output_dir, 'train.json')
    eval_file_path = os.path.join(args.output_dir, 'eval.jsonl')

    if args.build_dataset:
        print('>>> 步骤 1：加载文本数据')
        raw_items = load_text_data(
            dataset_name=args.dataset_name,
            split=args.split,
            max_samples=args.max_samples,
            content_key=args.content_key,
            local_path=args.local_path,
        )
        print('\n>>> 步骤 2：Pipeline 处理（规范化 → 过滤 → 去重 → 分块）')
        print(f'    启用功能: {", ".join([k for k, v in enabled.items() if v])}')

        result = build_pretrain_dataset(
            raw_items=raw_items,
            domain=args.domain,
            content_key=args.content_key,
            output_dir=args.output_dir,
            language=args.language,
            domain_keywords=args.domain_keywords,
            enabled=enabled,
            options=options,
            eval_seed=args.eval_seed,
            eval_ratio=args.eval_ratio,
            eval_from_train=args.eval_from_train,
        )
        train_file = result['train_file']
        eval_file_path = result['eval_file']
        total_chunks = result['total_chunks']
        print(f'预训练数据集构建完成！共生成 {total_chunks} 个分块')

    if args.eval_quality:
        if not os.path.exists(train_file):
            print(f'\n错误：分块文件不存在，请先运行 --build_dataset 生成 {train_file}')
            return
        print('\n>>> 质量评估：分块数据统计')
        evaluate_chunk_quality(
            chunk_file=train_file,
            max_samples=args.max_eval_samples,
            output_dir=args.output_dir,
        )

    if args.train_flag:
        if not os.path.exists(train_file):
            print(f'\n错误：预训练数据不存在，请先运行 --build_dataset 生成 {train_file}')
            return
        print('\n>>> 步骤 3：执行 LLM 预训练')
        run_pretrain(
            pretrain_data_path=train_file,
            base_model=args.base_model,
            train_target_path=args.train_target_path,
            num_epochs=args.num_epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cutoff_len=args.cutoff_len,
            warmup_ratio=args.warmup_ratio,
            ngpus=args.ngpus,
            save_steps=args.save_steps,
            logging_steps=args.train_logging_steps,
            save_total_limit=args.save_total_limit,
            train_launcher=args.train_launcher,
            sco_partition=args.sco_partition,
            sco_resource=args.sco_resource,
        )

    if args.eval_model:
        eval_file = args.eval_chunk_file or eval_file_path
        pretrained_path = './models/financial_pt/lazyllm_merge'
        if not os.path.isfile(eval_file):
            msg = f'\n错误：评测集不存在: {eval_file}，'
            msg += '请先运行 --build_dataset 或指定 --eval_chunk_file'
            print(msg)
            return
        evaluate_pretrained_model(
            base_model_path=args.base_model,
            pretrained_model_path=pretrained_path,
            eval_jsonl_path=eval_file,
            output_dir=args.output_dir,
            max_eval_samples=(
                args.eval_max_ppl_samples or MAX_EVAL_SAMPLES_DEFAULT
            ),
            max_new_tokens=args.eval_max_new_tokens,
            ppl_max_length=EVAL_MAX_LENGTH,
            gen_batch_size=EVAL_GEN_BATCH_SIZE,
            ppl_mode=args.ppl_mode,
        )

    if (not args.build_dataset and not args.train_flag
            and not args.eval_quality and not args.eval_model):
        print('请指定 --build_dataset / --train_flag / '
              '--eval_quality / --eval_model 至少其一')
        print('示例：python domain_pt_ppl.py --build_dataset --max_samples 1000')
        print('      python domain_pt_ppl.py --eval_model '
              '--pretrained_model /path/to/pretrained')


if __name__ == '__main__':
    main(parse_args())
