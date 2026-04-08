#!/usr/bin/env python3
'''
Calculate accuracy of model responses.

Extract answers from boxed{} format and compare with ground truth.
'''
import json
import re


def extract_boxed_answer(response):
    '''Extract answer from boxed{} format in response.'''
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer):
    '''Normalize answer for comparison.'''
    if answer is None:
        return None

    answer = str(answer).strip()

    answer = answer.replace(',', '')
    answer = answer.replace('$', '')
    answer = answer.replace(' ', '')

    try:
        return float(answer)
    except ValueError:
        return answer.lower()


def load_predictions(json_file):
    '''Load predictions from JSON file.'''
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = {}
    for item in data:
        test_id = item.get('test_case_id')
        response = item.get('response', '')
        extracted_answer = extract_boxed_answer(response)
        predictions[test_id] = {
            'question': item.get('question', ''),
            'response': response,
            'extracted_answer': extracted_answer,
            'normalized_answer': normalize_answer(extracted_answer)
        }

    return predictions


def load_ground_truth(jsonl_file):
    '''Load ground truth answers from JSONL file.'''
    ground_truth = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            data = json.loads(line.strip())
            answer = data.get('answer', '')
            ground_truth[idx] = {
                'question': data.get('problem', ''),
                'answer': answer,
                'normalized_answer': normalize_answer(answer)
            }

    return ground_truth


def calculate_accuracy(predictions, ground_truth):
    '''Calculate accuracy by comparing predictions with ground truth.'''
    total = len(predictions)
    correct = 0
    incorrect = 0
    missing_answer = 0

    results = []

    for test_id, pred in predictions.items():
        if test_id not in ground_truth:
            print(f'Warning: No ground truth for test case {test_id}')
            continue

        gt = ground_truth[test_id]

        if pred['extracted_answer'] is None:
            missing_answer += 1
            is_correct = False
        else:
            is_correct = pred['normalized_answer'] == gt['normalized_answer']
            if is_correct:
                correct += 1
            else:
                incorrect += 1

        results.append({
            'test_case_id': test_id,
            'question': pred['question'],
            'predicted_answer': pred['extracted_answer'],
            'normalized_predicted': pred['normalized_answer'],
            'ground_truth': gt['answer'],
            'normalized_truth': gt['normalized_answer'],
            'is_correct': is_correct
        })

    accuracy = (correct / total) * 100 if total > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'missing_answer': missing_answer,
        'accuracy': accuracy,
        'results': results
    }


def save_results(accuracy_data, output_file):
    '''Save accuracy results to JSON file.'''
    summary = {
        'total_test_cases': accuracy_data['total'],
        'correct_predictions': accuracy_data['correct'],
        'incorrect_predictions': accuracy_data['incorrect'],
        'missing_answers': accuracy_data['missing_answer'],
        'accuracy_percentage': round(accuracy_data['accuracy'], 2)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'detailed_results': accuracy_data['results']
        }, f, ensure_ascii=False, indent=2)

    return summary


def main():
    '''Main function.'''
    predictions_file = (
        '/home/mnt/huangchongjin/LLM/GRPO/cold_sft/sft_dataset/'
        'problem_codesft.json'
    )
    ground_truth_file = (
        '/home/mnt/huangchongjin/LLM/GRPO/grpo_dataset/'
        'gsm8k_test_converted.jsonl'
    )
    output_file = 'accuracy_codesft.json'

    print('=' * 80)
    print('Model Accuracy Calculator')
    print('=' * 80)
    print(f'Predictions File: {predictions_file}')
    print(f'Ground Truth File: {ground_truth_file}')
    print(f'Output File: {output_file}')
    print('=' * 80)

    print('\nLoading predictions...')
    predictions = load_predictions(predictions_file)
    print(f'Loaded {len(predictions)} predictions')

    print('\nLoading ground truth...')
    ground_truth = load_ground_truth(ground_truth_file)
    print(f'Loaded {len(ground_truth)} ground truth answers')

    print('\nCalculating accuracy...')
    accuracy_data = calculate_accuracy(predictions, ground_truth)

    print('\n' + '=' * 80)
    print('Accuracy Summary')
    print('=' * 80)
    print(f"Total Test Cases: {accuracy_data['total']}")
    print(f"Correct: {accuracy_data['correct']}")
    print(f"Incorrect: {accuracy_data['incorrect']}")
    print(f"Missing Answers: {accuracy_data['missing_answer']}")
    print(f"Accuracy: {accuracy_data['accuracy']:.2f}%")
    print('=' * 80)

    print('\nSaving results...')
    save_results(accuracy_data, output_file)
    print(f'Results saved to: {output_file}')

    print('\n' + '=' * 80)
    print('Incorrect Examples (first 5):')
    print('=' * 80)
    incorrect_examples = [
        r for r in accuracy_data['results'] if not r['is_correct']
    ][:5]
    for i, example in enumerate(incorrect_examples, 1):
        print(f'\nExample {i}:')
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Predicted: {example['predicted_answer']}")
        print(f"  Ground Truth: {example['ground_truth']}")

    print('\n' + '=' * 80)
    print('Done!')
    print('=' * 80)


if __name__ == '__main__':
    main()
