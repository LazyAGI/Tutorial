'''
Flickr8k 多模态预训练一体化脚本：数据准备、训练、评测一步到位。
数据集：Flickr8k（HuggingFace: clip-benchmark/wds_flickr8k）
模型：Qwen2.5-VL-3B
Pipeline：build_mm_pt_pipeline（图像完整性/分辨率过滤、去重）
评测：对齐能力（R@K）+ 生成能力（CIDEr）+ 语义一致性（CLIP Score）
支持 --mode: prepare | train | eval | full
'''
import os
import json
import math
import torch
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
import lazyllm
from lazyllm import finetune, launchers
from lazyllm.tools.data.pipelines.pt_data_ppl import build_mm_pt_pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PRETRAIN_CKPT_DIR = os.path.join(BASE_DIR, 'pretrain_ckpt')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
RAW_JSONL_PATH = os.path.join(DATA_DIR, 'flickr_raw.jsonl')
CLEANED_JSONL_PATH = os.path.join(DATA_DIR, 'flickr_cleaned.jsonl')
TRAIN_JSON_PATH = os.path.join(DATA_DIR, 'flickr_train.json')
EVAL_JSONL_PATH = os.path.join(DATA_DIR, 'flickr_eval.jsonl')

TRAIN_SAMPLES = 3000
EVAL_SAMPLES = 200
SOURCE_LOAD_LIMIT = 4000

BASE_MODEL_PATH = (
    '/mnt/lustre/share_data/lazyllm/models/Qwen2.5-VL-3B-Instruct'
)
MAX_NEW_TOKENS = 128
GEN_BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ──────────────────────────────────────────────
# 数据准备
# ──────────────────────────────────────────────

def _load_flickr_raw(limit):
    '''从 HuggingFace 加载 Flickr8k，保存原始记录到 RAW_JSONL_PATH，返回记录列表。'''
    if os.path.exists(RAW_JSONL_PATH):
        print(f'原始数据已存在，直接加载: {RAW_JSONL_PATH}')
        records = []
        with open(RAW_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f'加载原始数据: 共 {len(records)} 条')
        return records

    print(f'正在加载 Flickr8k，取前 {limit} 条...')
    dataset = load_dataset(
        'clip-benchmark/wds_flickr8k',
        split='train',
        trust_remote_code=True,
    )
    os.makedirs(IMAGES_DIR, exist_ok=True)
    records = []
    for idx, item in enumerate(dataset):
        if len(records) >= limit:
            break
        img_obj = item.get('jpg') or item.get('image') or item.get('png')
        if img_obj is None:
            continue
        captions_raw = (
            item.get('txt')
            or item.get('caption')
            or item.get('captions')
            or ''
        )
        if isinstance(captions_raw, bytes):
            captions_raw = captions_raw.decode('utf-8', errors='ignore')
        captions = [c.strip() for c in captions_raw.split('\n') if c.strip()]
        if not captions:
            continue
        img_path = os.path.join(IMAGES_DIR, f'{idx:06d}.jpg')
        if not os.path.exists(img_path):
            if hasattr(img_obj, 'save'):
                img_obj.save(img_path)
            else:
                Image.open(BytesIO(img_obj)).save(img_path)
        records.append({
            'image_path': img_path,
            'text': captions[0],
            'captions': captions,
        })
    print(f'加载完成: 共 {len(records)} 条有效记录')

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RAW_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'原始数据已保存: {RAW_JSONL_PATH}')
    return records


def _run_mm_pipeline(raw_records):
    '''运行 build_mm_pt_pipeline 并保存到 CLEANED_JSONL_PATH。'''
    # print(f'\n正在启动基座 VLM（vllm）用于图文相关性过滤...')
    # vlm = TrainableModule(BASE_MODEL_PATH).deploy_method((deploy.vllm, {
    #     'tensor_parallel_size': 1,
    #     'max_num_seqs': 16,
    # })).start()
    # print('VLM 启动完成')

    print(f'正在通过 build_mm_pt_pipeline 处理 {len(raw_records)} 条原始记录...')
    ppl = build_mm_pt_pipeline(
        image_key='image_path',
        text_key='text',
        # vlm=vlm,
        min_width=256,
        min_height=256,
        max_side=1024,
        relevance_threshold=0.6,
        use_dedup=True,
    )
    results = ppl(raw_records)
    results = (
        results
        if isinstance(results, list)
        else ([] if not results else [results])
    )
    print(f'pipeline 完成: 输入 {len(raw_records)} 条 → 有效 {len(results)} 条'
          f'（过滤 {len(raw_records) - len(results)} 条）')

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CLEANED_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'清洗后数据已保存: {CLEANED_JSONL_PATH}')
    return results


def _load_cleaned():
    records = []
    with open(CLEANED_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f'加载清洗后数据: 共 {len(records)} 条')
    return records


def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TRAIN_JSON_PATH) and os.path.exists(EVAL_JSONL_PATH):
        print('训练集与评测集已存在，跳过生成')
        return

    if os.path.exists(CLEANED_JSONL_PATH):
        print(f'清洗后数据已存在，直接加载: {CLEANED_JSONL_PATH}')
        cleaned = _load_cleaned()
    else:
        raw = _load_flickr_raw(SOURCE_LOAD_LIMIT)
        if not raw:
            print('无有效原始数据，退出')
            return
        cleaned = _run_mm_pipeline(raw)

    if not cleaned:
        print('pipeline 无有效输出，退出')
        return

    train_recs = cleaned[:TRAIN_SAMPLES]
    eval_recs = cleaned[:EVAL_SAMPLES]

    if not os.path.exists(TRAIN_JSON_PATH):
        train_data = [
            {
                'image': _norm_image_path(rec.get('image_path')),
                'text': rec['text'],
            }
            for rec in train_recs
        ]
        with open(TRAIN_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f'训练集已保存：{TRAIN_JSON_PATH}，共 {len(train_data)} 条')

    eval_samples = [
        {
            'image_path': _norm_image_path(rec.get('image_path')),
            'caption': (rec.get('captions') or [rec['text']])[0],
        }
        for rec in eval_recs
    ]
    with open(EVAL_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in eval_samples:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'评测集已保存：{EVAL_JSONL_PATH}，共 {len(eval_samples)} 条')

    print('\n=== 训练集第 1 条 ===')
    if train_recs:
        r = train_recs[0]
        print(f'  image_path: {r["image_path"]}')
        print(f'  text: {r["text"][:200]}')
    print('\n=== 评测集第 1 条 ===')
    if eval_samples:
        r = eval_samples[0]
        print(f'  image_path: {r["image_path"]}')
        print(f'  caption: {r["caption"][:200]}')
    print()


# ──────────────────────────────────────────────
# 模型训练
# ──────────────────────────────────────────────

def run_train():
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    target_path = os.path.join(
        PRETRAIN_CKPT_DIR, f'qwen2_5vl_3b_flickr_{timestamp}'
    )
    os.makedirs(PRETRAIN_CKPT_DIR, exist_ok=True)

    model = lazyllm.TrainableModule(BASE_MODEL_PATH, target_path=target_path)
    model.mode('finetune')\
        .trainset(TRAIN_JSON_PATH)\
        .finetune_method((finetune.llamafactory, {
            'stage': 'pt',
            'finetuning_type': 'full',
            'learning_rate': 1.2e-5,
            'cutoff_len': 512,
            'val_size': 0.05,
            'optim': 'adamw_torch_fused',
            'bf16': True,
            'fp16': False,
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 10,
            'num_train_epochs': 15,
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


# ──────────────────────────────────────────────
# 评测工具函数
# ──────────────────────────────────────────────

def _norm_image_path(path):
    if isinstance(path, list) and path:
        return path[0]
    return path if isinstance(path, str) else ''


def load_eval_samples(path: str) -> List[Dict]:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            img_path = _norm_image_path(item.get('image_path'))
            caption = item.get('caption')
            if isinstance(caption, list) and caption:
                caption = caption[0]
            if not img_path or not caption:
                continue
            samples.append({'image_path': img_path, 'caption': caption})
    print(f'加载评测集 {len(samples)} 条')
    return samples


def load_vl_model(model_path: str):
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    return model, processor


def _generate_caption(model, processor, image_path) -> str:
    path = _norm_image_path(image_path)
    image = Image.open(path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': 'Describe the image in one sentence.'},
    ]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    pred = processor.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return pred


def _get_image_embedding(model, processor, image_path) -> 'torch.Tensor':
    '''用 Qwen2.5-VL 视觉编码器提取图像均值特征作为 CLIP Score 代理。'''
    path = _norm_image_path(image_path)
    image = Image.open(path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': '.'},
    ]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]
        emb = hidden.mean(dim=0)
    return emb / (emb.norm() + 1e-8)


def _get_text_embedding(model, processor, text: str) -> 'torch.Tensor':
    inputs = processor(text=[text], return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]
        emb = hidden.mean(dim=0)
    return emb / (emb.norm() + 1e-8)


def _clip_score(
    img_emb: 'torch.Tensor', txt_emb: 'torch.Tensor'
) -> float:
    return max(0.0, torch.dot(img_emb.float(), txt_emb.float()).item()) * 2.5


def _tokenize_caption(cap: str) -> List[str]:
    return cap.lower().split()


def _compute_cider(  # noqa: C901
    refs_list: List[List[str]], preds: List[str]
) -> float:
    '''简化版 CIDEr-D（n=1..4 TF-IDF 加权 n-gram cosine，不做高斯平滑）。'''
    n_max = 4
    doc_freq: Dict[Tuple, int] = defaultdict(int)
    all_ref_ngrams = []
    all_pred_ngrams = []
    n_docs = len(refs_list)

    for refs in refs_list:
        seen = set()
        for ref in refs:
            toks = _tokenize_caption(ref)
            for n in range(1, n_max + 1):
                for i in range(len(toks) - n + 1):
                    ng = tuple(toks[i:i + n])
                    if ng not in seen:
                        doc_freq[ng] += 1
                        seen.add(ng)

    for refs in refs_list:
        ref_ngrams: Dict[Tuple, float] = defaultdict(float)
        for ref in refs:
            toks = _tokenize_caption(ref)
            for n in range(1, n_max + 1):
                for i in range(len(toks) - n + 1):
                    ref_ngrams[tuple(toks[i:i + n])] += 1.0 / len(refs)
        all_ref_ngrams.append(ref_ngrams)

    for pred in preds:
        toks = _tokenize_caption(pred)
        pred_ngrams: Dict[Tuple, float] = defaultdict(float)
        for n in range(1, n_max + 1):
            for i in range(len(toks) - n + 1):
                pred_ngrams[tuple(toks[i:i + n])] += 1.0
        all_pred_ngrams.append(pred_ngrams)

    scores = []
    for ref_ngrams, pred_ngrams in zip(all_ref_ngrams, all_pred_ngrams):
        all_keys = set(ref_ngrams) | set(pred_ngrams)
        ref_vec, pred_vec = [], []
        for ng in all_keys:
            idf = math.log((n_docs + 1.0) / (doc_freq.get(ng, 0) + 1.0))
            ref_vec.append(ref_ngrams.get(ng, 0.0) * idf)
            pred_vec.append(pred_ngrams.get(ng, 0.0) * idf)
        r_norm = math.sqrt(sum(v * v for v in ref_vec)) + 1e-8
        p_norm = math.sqrt(sum(v * v for v in pred_vec)) + 1e-8
        dot = sum(r * p for r, p in zip(ref_vec, pred_vec))
        scores.append(dot / (r_norm * p_norm))
    return sum(scores) / len(scores) if scores else 0.0


def _recall_at_k(
    img_embs: List['torch.Tensor'],
    txt_embs: List['torch.Tensor'],
    k_list: Tuple = (1, 5, 10),
) -> Dict[str, float]:
    '''Image-to-Text R@K：看 ground-truth 是否进入 top-K。'''
    n = len(img_embs)
    img_mat = torch.stack(img_embs).float()
    txt_mat = torch.stack(txt_embs).float()
    sim = img_mat @ txt_mat.T
    results = {}
    for k in k_list:
        hit = sum(
            1
            for i in range(n)
            if i in sim[i].topk(min(k, n)).indices.tolist()
        )
        results[f'R@{k}'] = hit / n
    return results


# ──────────────────────────────────────────────
# 评测主流程
# ──────────────────────────────────────────────

def run_eval_for_model(
    model_path: str, label: str, samples: List[Dict]
) -> Dict:
    print(f'\n[{label}] 加载模型: {model_path}')
    model, processor = load_vl_model(model_path)

    preds, img_embs, txt_embs, refs_list = [], [], [], []

    print(f'[{label}] 生成描述 + 提取嵌入...')
    for item in tqdm(samples, desc=label, unit='条'):
        img_path = item['image_path']
        pred = _generate_caption(model, processor, img_path)
        preds.append(pred)
        refs_list.append([item['caption']])
        img_embs.append(_get_image_embedding(model, processor, img_path))
        txt_embs.append(_get_text_embedding(model, processor, pred))

    cider = _compute_cider(refs_list, preds)
    clip_scores = [_clip_score(ie, te) for ie, te in zip(img_embs, txt_embs)]
    avg_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
    recall = _recall_at_k(img_embs, txt_embs)

    print(
        f'[{label}] CIDEr={cider:.4f}  CLIP Score={avg_clip:.4f}  '
        f'R@1={recall["R@1"]:.4f}  R@5={recall["R@5"]:.4f}  '
        f'R@10={recall["R@10"]:.4f}'
    )

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return {
        'cider': cider,
        'clip_score': avg_clip,
        'recall': recall,
        'preds': preds,
        'clip_per_sample': clip_scores,
    }


def run_eval(pt_model_path: str = None, out_run_dir: str = None) -> str:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if not os.path.exists(EVAL_JSONL_PATH):
        print(f'评测集不存在: {EVAL_JSONL_PATH}，请先运行 prepare 或 full')
        return ''

    samples = load_eval_samples(EVAL_JSONL_PATH)
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
    metrics_path = os.path.join(run_dir, 'eval_pt_flickr_metrics.json')
    results_path = os.path.join(run_dir, 'eval_pt_flickr_results.jsonl')

    def _fmt(m):
        return {
            'cider': m['cider'],
            'clip_score': m['clip_score'],
            'R@1': m['recall']['R@1'],
            'R@5': m['recall']['R@5'],
            'R@10': m['recall']['R@10'],
        }

    metrics = {
        'num_samples': len(samples),
        'base': _fmt(base_metrics),
        'pt': _fmt(pt_metrics) if pt_metrics else None,
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(results_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(samples):
            rec = {
                'id': i + 1,
                'image_path': item['image_path'],
                'caption': item['caption'],
                'base_pred': base_metrics['preds'][i],
                'base_clip': base_metrics['clip_per_sample'][i],
            }
            if pt_metrics:
                rec['pt_pred'] = pt_metrics['preds'][i]
                rec['pt_clip'] = pt_metrics['clip_per_sample'][i]
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'\n评测结果已保存至: {run_dir}')
    print(f'  指标: {metrics_path}')
    print(f'  逐条结果: {results_path}')
    print('\n' + '=' * 60)
    print('Flickr8k 多模态预训练评测（image → caption）')
    print('=' * 60)
    b = base_metrics
    print(
        f'基座模型   CIDEr={b["cider"]:.4f}  CLIP={b["clip_score"]:.4f}'
        f'  R@1={b["recall"]["R@1"]:.4f}  R@5={b["recall"]["R@5"]:.4f}'
        f'  R@10={b["recall"]["R@10"]:.4f}'
    )
    if pt_metrics:
        p = pt_metrics
        print(
            f'预训练模型 CIDEr={p["cider"]:.4f}  CLIP={p["clip_score"]:.4f}'
            f'  R@1={p["recall"]["R@1"]:.4f}  R@5={p["recall"]["R@5"]:.4f}'
            f'  R@10={p["recall"]["R@10"]:.4f}'
        )
    return run_dir


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Flickr8k 多模态预训练：数据准备、训练、评测一体化'
    )
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
# 1. 仅准备数据：     python run_flickr.py --mode=prepare
# 2. 准备 + 训练：     python run_flickr.py --mode=train
# 3. 仅评测：         python run_flickr.py --mode=eval
#                    [--pt_model_path=path/to/lazyllm_merge]
# 4. 一步到位：       python run_flickr.py --mode=full
if __name__ == '__main__':
    main()
