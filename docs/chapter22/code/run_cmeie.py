'''
CMeIE 医疗信息抽取微调：数据准备、训练、推理、评测一体化。
数据集：Aunderline/CMeIE，输出格式为单个 JSON 三元组（subject/predicate/object）。
支持 --mode: prepare | infer | train | eval | full
'''
import os
import re
import json
import random
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any

from datasets import load_dataset
import lazyllm
from lazyllm import finetune, launchers, deploy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CKPT_DIR = os.path.join(BASE_DIR, 'ckpt')
TRAIN_JSON_PATH = os.path.join(DATA_DIR, 'cmeie_train.json')
EVAL_JSONL_PATH = os.path.join(DATA_DIR, 'cmeie_eval.jsonl')
RAW_JSONL_PATH = os.path.join(DATA_DIR, 'cmeie_raw.jsonl')

TRAIN_SAMPLES = 3000
EVAL_SAMPLES = 200

INSTRUCTION = (
    '你是一个专业的医疗信息抽取专家。请从给定文本中提取一个医疗三元组。\n'
    '要求：\n'
    '1. 输出必须是纯JSON对象格式，不要添加```json或其他标记\n'
    '2. 包含 subject、predicate、object 三个字段\n'
    '3. 不要输出任何解释、说明或额外文字\n'
    '输出格式示例：{"subject":"阿司匹林","predicate":"适应症","object":"心血管疾病"}'
)
PROMPT_TEMPLATE = '{instruction}\n\n输入文本：\n{input}'


def _format_cmeie_to_sft_single(item: Dict) -> Optional[Dict]:
    '''将 CMeIE 一条样本转为 SFT 格式，output 为单个三元组 JSON。'''
    text = item.get('text', '').strip().replace('\n', ' ')
    raw_spo_list = item.get('spo_list', [])
    if not raw_spo_list:
        return None
    spo = raw_spo_list[0]
    obj_data = spo.get('object', {})
    obj_value = (
        obj_data.get('@value', '')
        if isinstance(obj_data, dict)
        else str(obj_data)
    )
    sub = spo.get('subject', '').strip()
    pre = spo.get('predicate', '').strip()
    obj = obj_value.strip()
    if not sub or not pre or not obj:
        return None
    triple = {'subject': sub, 'predicate': pre, 'object': obj}
    return {
        'instruction': INSTRUCTION,
        'input': text,
        'output': json.dumps(triple, ensure_ascii=False),
    }


def _is_valid_single(entry: Dict, max_length: int = 512) -> bool:
    if not entry.get('input') or len(entry['input']) > max_length:
        return False
    out = entry.get('output')
    if not isinstance(out, str) or not out.strip():
        return False
    try:
        parsed = json.loads(out)
        if not isinstance(parsed, dict):
            return False
        if not all(k in parsed for k in ('subject', 'predicate', 'object')):
            return False
    except json.JSONDecodeError:
        return False
    return True


def prepare_dataset():
    '''从 HuggingFace 加载 CMeIE，转为单三元组 SFT 格式，切分训练/评测并落盘。'''
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TRAIN_JSON_PATH) and os.path.exists(EVAL_JSONL_PATH):
        print('训练集与评测集已存在，跳过生成')
        return
    print('正在从 HuggingFace 加载 CMeIE 数据集...')
    dataset = load_dataset(
        'Aunderline/CMeIE',
        split='train',
        trust_remote_code=True,
    )
    raw_list = list(dataset)
    with open(RAW_JSONL_PATH, 'w', encoding='utf-8') as f:
        for entry in raw_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f'原始数据集已保存：{RAW_JSONL_PATH}，共 {len(raw_list)} 条')
    if raw_list:
        print('=== 原始数据示例 ===')
        print(json.dumps(raw_list[0], ensure_ascii=False, indent=2))
        print('=== 示例结束 ===')
    rows = []
    for entry in raw_list:
        formatted = _format_cmeie_to_sft_single(entry)
        if formatted and _is_valid_single(formatted):
            rows.append(formatted)
    random.seed(42)
    random.shuffle(rows)
    train_list = rows[:TRAIN_SAMPLES]
    eval_list = rows[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]
    with open(TRAIN_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_list, f, ensure_ascii=False, indent=2)
    with open(EVAL_JSONL_PATH, 'w', encoding='utf-8') as f:
        for rec in eval_list:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'训练集已保存：{TRAIN_JSON_PATH}，共 {len(train_list)} 条')
    print(f'评测集已保存：{EVAL_JSONL_PATH}，共 {len(eval_list)} 条')
    if train_list:
        print('=== 训练集示例 ===')
        print(json.dumps(train_list[0], ensure_ascii=False, indent=2))
        print('=== 示例结束 ===')
    if eval_list:
        print('=== 评测集示例 ===')
        print(json.dumps(eval_list[0], ensure_ascii=False, indent=2))
        print('=== 示例结束 ===')


def load_eval_samples(path: str) -> List[Dict]:
    '''加载评测样本，每条含 instruction/input/output。'''
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    print(f'加载评测集 {len(samples)} 条')
    return samples


def build_eval_prompts(samples: List[Dict]) -> List[str]:
    '''根据 samples 构建评测用 prompt 列表。'''
    return [
        PROMPT_TEMPLATE.format(
            instruction=s.get('instruction', INSTRUCTION),
            input=s.get('input', ''),
        )
        for s in samples
    ]


def _parse_triple(s: Optional[str]) -> Optional[Dict[str, str]]:
    '''从字符串解析出 subject/predicate/object，支持被 ``` 包裹。'''
    if not s or not str(s).strip():
        return None
    raw = str(s).strip()
    m = re.search(
        r'\{[^{}]*"subject"[^{}]*"predicate"[^{}]*"object"[^{}]*\}',
        raw,
        re.DOTALL,
    )
    if m:
        raw = m.group(0)
    try:
        out = json.loads(raw)
        if isinstance(out, dict) and all(
            k in out for k in ('subject', 'predicate', 'object')
        ):
            return {
                k: str(out.get(k, '')).strip()
                for k in ('subject', 'predicate', 'object')
            }
    except json.JSONDecodeError:
        pass
    return None


def _compute_metrics_from_preds(
    samples: List[Dict],
    preds: List[str],
) -> Dict[str, float]:
    '''根据 samples 的 output 与 preds 计算各项指标。'''
    n = len(samples)
    if not n or not preds or len(preds) != n:
        return {}
    exact_match = 0
    slot_correct = 0
    slot_total = 0
    json_ok = 0
    empty_count = 0
    tp = fp = fn = 0
    for i, sample in enumerate(samples):
        gold_str = sample.get('output', '')
        pred_str = preds[i] if i < len(preds) else ''
        gold_t = _parse_triple(gold_str)
        pred_t = _parse_triple(pred_str)
        if not (pred_str and str(pred_str).strip()):
            empty_count += 1
        if pred_t is not None:
            json_ok += 1
        if gold_t is None:
            continue
        if pred_t is not None:
            if (
                gold_t.get('subject') == pred_t.get('subject')
                and gold_t.get('predicate') == pred_t.get('predicate')
                and gold_t.get('object') == pred_t.get('object')
            ):
                exact_match += 1
                tp += 1
            else:
                fp += 1
            for k in ('subject', 'predicate', 'object'):
                slot_total += 1
                if gold_t.get(k) == pred_t.get(k):
                    slot_correct += 1
        else:
            fn += 1
            slot_total += 3
    return {
        'exact_match': exact_match / n if n else 0.0,
        'slot_accuracy': slot_correct / slot_total if slot_total else 0.0,
        'json_compliance': json_ok / n if n else 0.0,
        'empty_pred_rate': empty_count / n if n else 0.0,
        'strict_precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'strict_recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'strict_f1': (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else 0.0
        ),
    }


def _print_metrics(label: str, m: Dict[str, float]):
    if not m:
        return
    print(
        f'[{label}] exact_match={m.get("exact_match", 0):.4f} '
        f'slot_accuracy={m.get("slot_accuracy", 0):.4f} '
        f'json_compliance={m.get("json_compliance", 0):.4f} '
        f'empty_pred_rate={m.get("empty_pred_rate", 0):.4f} '
        f'strict_precision={m.get("strict_precision", 0):.4f} '
        f'strict_recall={m.get("strict_recall", 0):.4f} '
        f'strict_f1={m.get("strict_f1", 0):.4f}'
    )


def _norm_preds(raw: Any) -> List[str]:
    '''将模型评测原始输出规范为与 samples 等长的字符串列表。'''
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() if x is not None else '' for x in raw]
    return []


def default_cmeie_eval(
    samples: List[Dict],
    base_preds: Optional[List[str]],
    ckpt_preds: Optional[List[str]],
    save_dir: Optional[str],
    eval_fn=None,
):
    '''统一评测入口：计算 base/ckpt 指标、打印、可选落盘。'''
    if eval_fn is not None:
        eval_fn(samples, base_preds, ckpt_preds, save_dir)
        return
    base_preds = _norm_preds(base_preds)
    ckpt_preds = _norm_preds(ckpt_preds)
    metrics = {'num_samples': len(samples), 'base': None, 'ckpt': None}
    if base_preds and len(base_preds) == len(samples):
        m = _compute_metrics_from_preds(samples, base_preds)
        metrics['base'] = m
        _print_metrics('基座模型', m)
    if ckpt_preds and len(ckpt_preds) == len(samples):
        m = _compute_metrics_from_preds(samples, ckpt_preds)
        metrics['ckpt'] = m
        _print_metrics('微调模型', m)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metrics_path = os.path.join(save_dir, 'eval_cmeie_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        results_path = os.path.join(save_dir, 'eval_cmeie_results.jsonl')
        with open(results_path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(samples):
                rec = {
                    'id': i + 1,
                    'input': s.get('input', ''),
                    'gold': s.get('output', ''),
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


def _gen_ckpt_dir() -> str:
    ts = datetime.now().strftime('%y%m%d%H%M%S')
    out = os.path.join(CKPT_DIR, f'qwen2_5_0_5b_cmeie_{ts}')
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

    prepare_dataset()
    if mode == 'prepare':
        return

    if not os.path.exists(eval_data_path):
        print(f'评测集不存在: {eval_data_path}，请先运行 prepare 或 full')
        return
    samples = load_eval_samples(eval_data_path)
    if not samples:
        print('无有效评测样本')
        return
    eval_prompts = build_eval_prompts(samples)

    if mode == 'infer':
        model = lazyllm.TrainableModule(model_path).deploy_method(
            (deploy.vllm, {
                'tensor_parallel_size': 1,
                'max_num_seqs': 32,
                'max_model_len': 512,
            })
        )
        model.evalset(eval_prompts)
        model.start()
        model.eval()
        base_preds = _norm_preds(model.eval_result)
        default_cmeie_eval(samples, base_preds, None, None)
        return

    if mode in ('train', 'full'):
        prepare_dataset()
        target_path = _gen_ckpt_dir()
        base_preds = None
        if mode == 'full':
            base_model = lazyllm.TrainableModule(model_path).deploy_method(
                (deploy.vllm, {
                    'tensor_parallel_size': 1,
                    'max_num_seqs': 32,
                    'max_model_len': 512,
                })
            )
            base_model.evalset(eval_prompts)
            base_model.start()
            base_model.eval()
            base_preds = _norm_preds(base_model.eval_result)
            getattr(base_model, 'stop', lambda: None)()
        model = lazyllm.TrainableModule(model_path, target_path=target_path)
        model.mode('finetune')
        model.trainset(train_data_path)
        model.finetune_method((finetune.llamafactory, {
            'stage': 'sft',
            'finetuning_type': 'lora',
            'learning_rate': 3e-5,
            'cutoff_len': 512,
            'val_size': 0.05,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'num_train_epochs': 8,
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            'save_steps': 200,
            'logging_steps': 10,
            'save_strategy': 'steps',
            'save_total_limit': 5,
            'launcher': launchers.empty(ngpus=1),
        }))
        model.deploy_method((deploy.vllm, {
            'tensor_parallel_size': 1,
            'max_num_seqs': 32,
            'max_model_len': 512,
        }))
        model.evalset(eval_prompts)
        model.update()
        ckpt_preds = _norm_preds(getattr(model, 'eval_result', None))
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(RESULTS_DIR, ts)
        default_cmeie_eval(
            samples,
            base_preds,
            ckpt_preds if ckpt_preds else None,
            save_dir,
        )
        print(f'训练输出目录: {target_path}')
        return

    if mode == 'eval':
        eval_res_path = eval_res_path or os.path.join(
            RESULTS_DIR, 'eval_cmeie_results.jsonl'
        )
        if not os.path.exists(eval_res_path):
            print(f'结果文件不存在: {eval_res_path}')
            return
        samples_from_file = []
        base_preds = []
        ckpt_preds = []
        with open(eval_res_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                samples_from_file.append({
                    'input': rec.get('input', ''),
                    'output': rec.get('gold', ''),
                })
                base_preds.append(rec.get('base_pred', ''))
                ckpt_preds.append(rec.get('ckpt_pred', ''))
        default_cmeie_eval(samples_from_file, base_preds, ckpt_preds, None)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CMeIE 医疗信息抽取：数据准备、训练、推理、评测'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=(
            '/home/mnt/path/.lazyllm/model/modelscope/qwen/'
            'Qwen2.5-0.5B-Instruct'
        ),
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
