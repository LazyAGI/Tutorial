import lazyllm
from lazyllm import launchers

model_path = 'your_sft_model_path'
output_path = 'GRPO/Qwen2.5-r1-cot-output'

m = lazyllm.TrainableModule(model_path, output_path)\
    .mode('finetune')\
    .trainset(lambda: lazyllm.package(
        '/GRPO/grpo_dataset/gsm8k_train_converted.jsonl',
        '/GRPO/grpo_dataset/gsm8k_test_converted.jsonl'
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
            'trainer.load_checkpoint_path': output_path + '/global_step_110',
            'launcher': launchers.sco(
                ngpus=1,
                partition='a800',
                resource='N3lS.Ii.I60.1',
            ),
        }))
