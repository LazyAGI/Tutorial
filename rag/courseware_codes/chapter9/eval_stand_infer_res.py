import os
import json

import lazyllm
from lazyllm.tools.eval import ResponseRelevancy, Faithfulness


os.environ['LAZYLLM_EVAL_RESULT_DIR'] = 'path/to/save_res/after_eval'
lazyllm.config.refresh('LAZYLLM_EVAL_RESULT_DIR')

llm = lazyllm.TrainableModule('Qwen2-72B-Instruct-AWQ').deploy_method(lazyllm.deploy.Vllm).start()
embd = lazyllm.TrainableModule('bge-m3').start()

def load_data(load_path):
    with open(load_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, save_path):
    with open(save_path, 'a') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')

def infer_res(data_path):
    data = load_data(data_path)
    m1 = ResponseRelevancy(llm, embd, concurrency=250, prompt_lang='zh')
    res1 = m1(data)

    m2 = Faithfulness(llm, concurrency=250, prompt_lang='zh')
    res2 = m2(data)
    data = {
        'ResponseRelevancy': res1,
        'Faithfulness': res2,
        'data_path': data_path,
    }
    save_data(data, 'after_eval/final_res.json')

if __name__ == '__main__':
    dir_path = 'path/to/infer_res/for_eval'
    json_files = []
    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            json_files.append(os.path.join(dir_path, file))
    print(json_files)
    for data_path in json_files:
        infer_res(data_path)
