import json
import os
from typing import List, Dict, Optional
import argparse
from pathlib import Path

# ============ 默认配置 ============
# 默认使用公开可下载、结构简单的函数调用评测集。
# 顶层字段可近似映射为:
# - query <- user
# - tools <- tools
# - answer <- completion
DEFAULT_DATASET_REPO = 'NousResearch/func-calling-eval'
DEFAULT_DATASET_SPLIT = 'train'
DEFAULT_CONFIG_NAME = 'G1'  # 可选: G1, G2, G3
DEFAULT_OUTPUT_DIR = Path(__file__).parent / 'data'
DEFAULT_OUTPUT_FILE = 'toolbench_lazyllm.jsonl'
DEFAULT_MAX_SAMPLES = 1000

HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_ENDPOINT', HF_ENDPOINT)


def load_toolbench_data(data_path: str) -> List[Dict]:
    '''
    加载ToolBench数据集

    Args:
        data_path: 数据文件路径，支持.json或.jsonl格式

    Returns:
        数据样本列表
    '''
    print(f'Loading ToolBench data from: {data_path}')

    if data_path.endswith('.jsonl'):
        # JSONL格式：每行一个样本
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
    elif data_path.endswith('.json'):
        # JSON格式：整个文件是一个列表或对象
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            if isinstance(raw_data, list):
                data = raw_data
            elif isinstance(raw_data, dict):
                # ToolBench格式可能包含多个字段
                data = raw_data.get('data', raw_data.get('train', raw_data.get('test', [])))
            else:
                data = [raw_data]
    else:
        raise ValueError(f'Unsupported file format: {data_path}')

    print(f'Loaded {len(data)} samples')
    return data


def load_toolbench_from_hf(repo_id: str = DEFAULT_DATASET_REPO,
                           split: str = DEFAULT_DATASET_SPLIT,
                           max_samples: Optional[int] = None) -> List[Dict]:
    '''
    从HuggingFace加载ToolBench-Formatted数据集

    Args:
        repo_id: HuggingFace数据集仓库ID
        split: 数据集split (train/test等)
        max_samples: 最大样本数限制

    Returns:
        数据样本列表
    '''
    try:
        from datasets import load_dataset
    except ImportError:
        print('Error: datasets library not installed.')
        print('Please install: pip install datasets')
        raise

    print(f'Loading ToolBench-Formatted from HuggingFace: {repo_id}')
    print(f'  - Split: {split}')
    if max_samples:
        print(f'  - Max samples: {max_samples}')

    try:
        dataset = load_dataset(repo_id, split=split)
        data = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            data.append(dict(item))
            if (i + 1) % 1000 == 0:
                print(f'  Loaded {i + 1} samples...')

        print(f'Successfully loaded {len(data)} samples from HuggingFace')
        return data

    except Exception as e:
        print(f'Error loading from HuggingFace: {e}')
        print('\nTroubleshooting:')
        print('  1. Check your internet connection')
        print('  2. Set HF_TOKEN if the dataset is gated: export HF_TOKEN=your_token')
        print('  3. Use --input to specify a local file instead')
        raise


def format_tool_description(tool: Dict) -> str:
    '''
    格式化工具描述

    Args:
        tool: 工具信息字典

    Returns:
        格式化的工具描述字符串
    '''
    if isinstance(tool, str):
        return tool

    if not isinstance(tool, dict):
        return str(tool)

    name = tool.get('name', '')
    description = tool.get('description', '')
    parameters = tool.get('parameters', {})

    # 构建参数描述
    if isinstance(parameters, dict) and 'properties' in parameters:
        param_desc = []
        for param_name, param_info in parameters['properties'].items():
            param_type = param_info.get('type', 'string')
            param_desc.append(f'{param_name} ({param_type})')
        param_str = ', '.join(param_desc)
        return f'{name}: {description} Parameters: {param_str}'
    else:
        return f'{name}: {description}'


def _stringify_value(value) -> str:
    '''将字段值稳定转换为字符串。'''
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _get_query(sample: Dict) -> str:
    '''提取用户问题。'''
    return (
        sample.get('query')
        or sample.get('question')
        or sample.get('user')
        or sample.get('prompt')
        or ''
    )


def _get_answer(sample: Dict) -> str:
    '''提取最终答案。'''
    return _stringify_value(
        sample.get('answer')
        or sample.get('final_answer')
        or sample.get('completion')
    )


def _get_tools(sample: Dict) -> List:
    '''提取工具列表，并兼容字符串或单对象格式。'''
    tools = sample.get('tools', [])

    if not tools:
        return []

    if isinstance(tools, list):
        return tools

    if isinstance(tools, dict):
        return [tools]

    if isinstance(tools, str):
        raw = tools.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return [_stringify_value(parsed)]

    return [_stringify_value(tools)]


def _extract_from_tool_calls(sample: Dict, trajectory: List[str]) -> None:
    '''从 tool_calls 字段提取轨迹。'''
    for i, call in enumerate(sample['tool_calls']):
        # 添加推理步骤（如果有的话）
        if f'thought_{i+1}' in sample:
            trajectory.append(f"Thought: {sample[f'thought_{i+1}']}")

        # 添加工具调用
        tool_name = call.get('name', '')
        arguments = call.get('arguments', {})
        args_str = json.dumps(arguments, ensure_ascii=False)
        trajectory.append(f'Action: {tool_name}({args_str})')

        # 添加观察结果（如果有的话）
        if f'observation_{i+1}' in sample:
            trajectory.append(f"Observation: {sample[f'observation_{i+1}']}")


def _extract_from_conversation(sample: Dict, trajectory: List[str]) -> None:
    '''从 conversation 字段提取轨迹。'''
    conversation = sample.get('conversation', sample.get('conversations', []))
    for turn in conversation:
        role = turn.get('role', '')
        content = turn.get('content', '')

        if role == 'assistant':
            # 解析assistant的推理过程
            if 'thought' in content.lower():
                trajectory.append(f'Thought: {content}')
            elif 'action' in content.lower():
                trajectory.append(f'Action: {content}')
        elif role == 'tool':
            trajectory.append(f'Observation: {content}')


def _extract_from_other_fields(sample: Dict, trajectory: List[str]) -> None:
    '''从其他字段构建轨迹。'''
    # 简单的推理过程
    trajectory.append(f'Thought: I need to answer: {_get_query(sample)}')

    # 如果有tool_calls信息
    if 'tool_calls' in sample:
        for call in sample['tool_calls']:
            tool_name = call.get('name', '')
            arguments = call.get('arguments', {})
            args_str = json.dumps(arguments, ensure_ascii=False)
            trajectory.append(f'Action: {tool_name}({args_str})')


def extract_trajectory_from_toolbench(sample: Dict) -> List[str]:  # noqa: C901
    '''
    从ToolBench样本中提取推理轨迹

    Args:
        sample: ToolBench数据样本

    Returns:
        轨迹步骤列表
    '''
    trajectory = []

    # 处理tool_calls（如果存在）
    if 'tool_calls' in sample and sample['tool_calls']:
        _extract_from_tool_calls(sample, trajectory)
    # 处理conversation格式（如果存在）
    elif 'conversation' in sample or 'conversations' in sample:
        _extract_from_conversation(sample, trajectory)
    # 如果没有轨迹信息，尝试从其他字段构建
    else:
        _extract_from_other_fields(sample, trajectory)

    return trajectory


def convert_toolbench_to_lazyllm(data: List[Dict],
                                 max_samples: Optional[int] = None,
                                 include_difficulty: bool = False) -> List[Dict]:
    '''
    将ToolBench数据集转换为LazyLLM格式

    Args:
        data: ToolBench数据列表
        max_samples: 最大样本数限制
        include_difficulty: 是否包含难度信息

    Returns:
        LazyLLM格式的数据列表
    '''
    lazyllm_data = []

    for i, sample in enumerate(data):
        if max_samples and i >= max_samples:
            break

        try:
            # 提取基本信息
            query = _get_query(sample)
            tools = _get_tools(sample)
            answer = _get_answer(sample)

            # 构建工具描述
            if tools:
                tools_str = '\n'.join([format_tool_description(tool) for tool in tools])
                instruction = (
                    'You are a helpful assistant. Answer the following questions as best you can. '
                    f'You have access to the following tools:\n{tools_str}'
                )
            else:
                instruction = 'You are a helpful assistant. Answer the following question as best you can.'

            # 提取推理轨迹
            trajectory_steps = extract_trajectory_from_toolbench(sample)

            # 添加最终答案
            if answer and (not trajectory_steps or not trajectory_steps[-1].startswith('Final Answer:')):
                trajectory_steps.append(f'Final Answer: {answer}')

            # 构建输出文本
            output = '\n'.join(trajectory_steps)

            # 构建LazyLLM样本
            lazyllm_sample = {
                'instruction': instruction,
                'input': query,
                'output': output
            }

            # 可选：添加难度信息
            if include_difficulty and 'difficulty' in sample:
                lazyllm_sample['difficulty'] = sample['difficulty']

            lazyllm_data.append(lazyllm_sample)

        except Exception as e:
            print(f'Warning: Failed to process sample {i}: {e}')
            continue

    print(f'Successfully converted {len(lazyllm_data)} samples')
    return lazyllm_data


def save_lazyllm_data(data: List[Dict], output_path: str, format: str = 'jsonl') -> None:
    '''
    保存转换后的数据

    Args:
        data: 转换后的数据
        output_path: 输出路径
        format: 输出格式 ('jsonl' 或 'json')
    '''
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f'Saving {len(data)} samples to {output_path}')

    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'jsonl':
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print('Data saved successfully!')


def run_with_defaults():
    '''使用默认配置运行（下载ToolBench-Formatted并转换）'''
    print('=' * 60)
    print('Function-Calling-to-LazyLLM Converter (Default Mode)')
    print('=' * 60)
    print()
    print('Configuration:')
    print(f'  Dataset: {DEFAULT_DATASET_REPO}')
    print(f'  Split: {DEFAULT_DATASET_SPLIT}')
    print(f'  Max samples: {DEFAULT_MAX_SAMPLES}')
    print(f'  Output: {DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE}')
    print()

    # 确保输出目录存在
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 从HuggingFace加载数据
    data = load_toolbench_from_hf(
        repo_id=DEFAULT_DATASET_REPO,
        split=DEFAULT_DATASET_SPLIT,
        max_samples=DEFAULT_MAX_SAMPLES
    )

    # 转换格式
    lazyllm_data = convert_toolbench_to_lazyllm(data)

    # 保存结果
    output_path = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE
    save_lazyllm_data(lazyllm_data, str(output_path), 'jsonl')

    # 显示统计信息
    print()
    print('=' * 60)
    print('Conversion Summary')
    print('=' * 60)
    print(f'  Input samples: {len(data)}')
    print(f'  Output samples: {len(lazyllm_data)}')
    print(f'  Success rate: {len(lazyllm_data)/len(data)*100:.1f}%')
    print(f'  Output file: {output_path}')
    print()

    # 显示示例
    if lazyllm_data:
        print('Sample output:')
        sample = lazyllm_data[0]
        print(f'  Instruction: {sample["instruction"][:100]}...')
        print(f'  Input: {sample["input"][:100]}...')
        print(f'  Output: {sample["output"][:100]}...')


def main():
    '''主函数'''
    parser = argparse.ArgumentParser(
        description='Convert function-calling dataset to LazyLLM format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 使用默认配置（自动下载公开的函数调用数据集）
  python toolbench2llm.py

  # 使用本地文件
  python toolbench2llm.py -i toolbench_data.json -o output.jsonl

  # 从HuggingFace加载指定数据集
  python toolbench2llm.py --hf-repo NousResearch/func-calling-eval -o output.jsonl
        '''
    )
    parser.add_argument('--input', '-i', default=None,
                        help='Input function-calling data file path (optional, uses HF dataset by default)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output LazyLLM data file path (optional)')
    parser.add_argument('--hf-repo', default=DEFAULT_DATASET_REPO,
                        help=f'HuggingFace dataset repo (default: {DEFAULT_DATASET_REPO})')
    parser.add_argument('--hf-split', default=DEFAULT_DATASET_SPLIT,
                        help=f'Dataset split (default: {DEFAULT_DATASET_SPLIT})')
    parser.add_argument('--max_samples', '-n', type=int, default=None,
                        help='Maximum number of samples to convert')
    parser.add_argument('--format', '-f', choices=['jsonl', 'json'], default='jsonl',
                        help='Output format (default: jsonl)')
    parser.add_argument('--include_difficulty', action='store_true',
                        help='Include difficulty information in output')

    args = parser.parse_args()

    # 如果没有提供input和output，使用默认模式
    if args.input is None and args.output is None:
        run_with_defaults()
        return

    # 如果提供了input但没提供output，使用默认输出路径
    if args.input and args.output is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        args.output = str(DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE)

    # 确保output是必需的（当提供了input时）
    if args.input and args.output is None:
        parser.error('--output/-o is required when --input/-i is specified')

    # 加载数据
    if args.input:
        # 从本地文件加载
        data = load_toolbench_data(args.input)
    else:
        # 从HuggingFace加载
        data = load_toolbench_from_hf(
            repo_id=args.hf_repo,
            split=args.hf_split,
            max_samples=args.max_samples
        )

    # 转换格式
    lazyllm_data = convert_toolbench_to_lazyllm(
        data,
        max_samples=args.max_samples,
        include_difficulty=args.include_difficulty
    )

    # 保存结果
    save_lazyllm_data(lazyllm_data, args.output, args.format)

    # 显示统计信息
    print()
    print('Conversion Statistics:')
    print(f'  Input samples: {len(data)}')
    print(f'  Output samples: {len(lazyllm_data)}')
    print(f'  Success rate: {len(lazyllm_data)/len(data)*100:.1f}%')

    # 显示示例
    if lazyllm_data:
        print()
        print('Sample output:')
        sample = lazyllm_data[0]
        print(f'  Instruction: {sample["instruction"][:200]}...')
        print(f'  Input: {sample["input"]}')
        print(f'  Output: {sample["output"][:300]}...')


if __name__ == '__main__':
    main()
