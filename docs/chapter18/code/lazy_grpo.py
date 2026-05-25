import os

HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_ENDPOINT', HF_ENDPOINT)

import lazyllm
from huggingface_hub import hf_hub_download
from lazyllm import launchers


DATASET_REPO_ID = os.environ.get(
    'CH18_DATASET_REPO_ID',
    'rirqing/18chapter_data',
)
GRPO_TRAIN_FILENAME = os.environ.get(
    'CH18_GRPO_TRAIN_FILENAME',
    'gsm8k_train_converted.jsonl',
)
GRPO_TEST_FILENAME = os.environ.get(
    'CH18_GRPO_TEST_FILENAME',
    'gsm8k_test_converted.jsonl',
)


def download_dataset_file(filename):
    '''Download a file from the chapter 18 Hugging Face dataset repo.'''
    return hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=filename,
        repo_type='dataset',
    )

model_path = 'Qwen/Qwen2.5-0.5B-Instruct'
output_path = 'GRPO/Qwen2.5-r1-cot-output'

m = lazyllm.TrainableModule(model_path, output_path)\
    .mode('finetune')\
    .trainset(lambda: lazyllm.package(
        download_dataset_file(GRPO_TRAIN_FILENAME),
        download_dataset_file(GRPO_TEST_FILENAME)
    ))\
    .finetune_method(
        (lazyllm.finetune.easyr1, {
            'data.rollout_batch_size': 64,
            'data.val_batch_size': 32,
            'worker.actor.global_batch_size': 32,
            'trainer.save_model_only': False,
            'trainer.total_epochs': 1,
            'worker.rollout.tensor_parallel_size': 1,
            'trainer.save_freq': 10,
            'trainer.save_checkpoint_path': output_path,
            'launcher': launchers.empty(ngpus=1),
        }))
m.update()