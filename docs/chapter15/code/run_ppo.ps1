$env:HF_HUB_OFFLINE="0"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$ACTOR_MODEL_PATH = "LLaMA-Factory/models/LLM-Research/Llama-3.2-1B-Instruct"
$RM_MODEL_PATH = "D:/coding/LLaMA-Factory/models/LLM-Research/Llama-3.2-1B-Instruct"
    #奖励模型量化
    #5060启用BF16    
    #序列截断
cd D:\coding\LLaMA-Factory
python src/train.py `
    --stage ppo `
    --do_train `
    --model_name_or_path "$ACTOR_MODEL_PATH" `
    --trust_remote_code True `
    --reward_model "$RM_MODEL_PATH" `
    --reward_model_quantization_bit 4 `
    --reward_model_type full `
    --dataset ultrafeedback_ppo `
    --template default `
    --finetuning_type lora `
    --lora_target all `
    --output_dir saves/Llama-3.2-1B-Instruct/ppo `
    --overwrite_output_dir `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 8 `
    --gradient_checkpointing True `
    --learning_rate 1e-5 `
    --num_train_epochs 1.0 `
    --bf16 True `
    --cutoff_len 1024 `
    --top_k 0 `
    --top_p 0.9 `
    --ppo_score_norm True `
    --ppo_whiten_rewards True `
    --logging_steps 10 `
    --save_steps 100 `
    --plot_loss True `
    --report_to none