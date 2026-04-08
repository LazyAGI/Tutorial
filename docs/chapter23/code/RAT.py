import json
import random
from typing import List, Dict


TOOL_DATABASE = {
    'weather_current': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a specific location.',
        'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}}}
    },
    'weather_forecast': {
        'name': 'get_weather_forecast',
        'description': 'Get the weather forecast for the next 7 days.',
        'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}}}
    },
    'stock_price': {
        'name': 'get_stock_price',
        'description': 'Get the current stock price given a ticker symbol.',
        'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}}}
    },
    'currency_converter': {
        'name': 'convert_currency',
        'description': 'Convert amount from one currency to another.',
        'parameters': {
            'type': 'object',
            'properties': {
                'amount': {'type': 'number'},
                'from': {'type': 'string'},
                'to': {'type': 'string'}
            }
        }
    },
    'search_engine': {
        'name': 'google_search',
        'description': 'Search for general information on the internet.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}
    }
}


class RATDatasetBuilder:
    def __init__(self, tool_db: Dict):
        self.tool_db = tool_db

    def mock_retriever(self, query: str, top_k: int = 3) -> List[str]:
        if 'weather' in query.lower():
            return ['weather_current', 'weather_forecast', 'search_engine']
        elif 'stock' in query.lower() or 'price' in query.lower():
            return ['stock_price', 'currency_converter', 'search_engine']
        else:
            return list(self.tool_db.keys())[:top_k]

    def construct_training_sample(self, user_query: str, ground_truth_tool_id: str, top_k: int = 3):
        retrieved_ids = self.mock_retriever(user_query, top_k)

        if ground_truth_tool_id not in retrieved_ids:
            retrieved_ids.pop()
            retrieved_ids.append(ground_truth_tool_id)

        random.shuffle(retrieved_ids)

        tools_context = [self.tool_db[tid] for tid in retrieved_ids]

        target_tool = self.tool_db[ground_truth_tool_id]

        args = '{}'
        if 'location' in target_tool['parameters']['properties']:
            args = '{"location": "Shanghai"}'

        output_sequence = (
            f"Thought: The user is asking about '{user_query}'. "
            f"Looking at the tools, '{target_tool['name']}' is the most appropriate. "
            f"'{retrieved_ids[0]}' and others are not as specific or relevant.\n"
            f'Action: {target_tool["name"]}({args})'
        )

        return {
            'instruction': 'Answer the user query using the provided tools.',
            'input_tools': json.dumps(tools_context, indent=2),
            'user_query': user_query,
            'output': output_sequence
        }


builder = RATDatasetBuilder(TOOL_DATABASE)

sample = builder.construct_training_sample(
    user_query='What is the current weather in Shanghai?',
    ground_truth_tool_id='weather_current',
    top_k=3
)

print('### RAT 训练样本结构 ###\n')
print('--- [Input Context (包含干扰项)] ---')
print(sample['input_tools'])
print('\n--- [User Query] ---')
print(sample['user_query'])
print('\n--- [Target Output (模型需要学习辨别)] ---')
print(sample['output'])
