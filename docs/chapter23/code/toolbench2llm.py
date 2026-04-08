import json
import os
from typing import List, Dict, Optional
import argparse


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


def format_tool_description(tool: Dict) -> str:
    '''
    格式化工具描述

    Args:
        tool: 工具信息字典

    Returns:
        格式化的工具描述字符串
    '''
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
    for turn in sample['conversation']:
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
    trajectory.append(f"Thought: I need to answer: {sample.get('query', '')}")

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
    elif 'conversation' in sample:
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
            query = sample.get('query', sample.get('question', ''))
            tools = sample.get('tools', [])
            answer = sample.get('answer', sample.get('final_answer', ''))

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
            if answer and not trajectory_steps[-1].startswith('Final Answer:'):
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f'Saving {len(data)} samples to {output_path}')

    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'jsonl':
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print('Data saved successfully!')


def main():
    '''主函数'''
    parser = argparse.ArgumentParser(description='Convert ToolBench dataset to LazyLLM format')
    parser.add_argument('--input', '-i', required=True, help='Input ToolBench data file path')
    parser.add_argument('--output', '-o', required=True, help='Output LazyLLM data file path')
    parser.add_argument('--max_samples', '-n', type=int, help='Maximum number of samples to convert')
    parser.add_argument('--format', '-f', choices=['jsonl', 'json'], default='jsonl',
                        help='Output format (default: jsonl)')
    parser.add_argument('--include_difficulty', action='store_true',
                        help='Include difficulty information in output')

    args = parser.parse_args()

    # 加载数据
    data = load_toolbench_data(args.input)

    # 转换格式
    lazyllm_data = convert_toolbench_to_lazyllm(
        data,
        max_samples=args.max_samples,
        include_difficulty=args.include_difficulty
    )

    # 保存结果
    save_lazyllm_data(lazyllm_data, args.output, args.format)

    # 显示统计信息
    print('转换统计:')
    print(f'  输入样本数: {len(data)}')
    print(f'  输出样本数: {len(lazyllm_data)}')
    print(f'  成功率: {len(lazyllm_data)/len(data)*100:.1f}%')

    # 显示示例
    if lazyllm_data:
        print('转换结果示例:')
        sample = lazyllm_data[0]
        print(f'Instruction: {sample["instruction"][:200]}...')
        print(f'Input: {sample["input"]}')
        print(f'Output: {sample["output"][:300]}...')


if __name__ == '__main__':
    # 命令行模式
    main()

    # 或者直接运行示例
    # 示例数据转换（用于测试）
    # sample_toolbench_data = [
    #     {
    #         "query": "What's the weather like in Tokyo?",
    #         "tools": [
    #             {"name": "get_weather", "description": "Get current weather",
    #              "parameters": {"properties": {"location": {"type": "string"}}}}
    #         ],
    #         "tool_calls": [
    #             {"name": "get_weather", "arguments": {"location": "Tokyo"}}
    #         ],
    #         "answer": "The weather in Tokyo is sunny with 25C."
    #     }
    # ]
    #
    # lazyllm_data = convert_toolbench_to_lazyllm(sample_toolbench_data)
    # save_lazyllm_data(lazyllm_data, "sample_converted.jsonl")
