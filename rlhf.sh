python rlhf.py --output_dir './save' \
    --seed 2023 \
    --step_per_device_batch_size 2 \
    --rollout_per_device_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-5 \
    --warmup_steps 5 \
    --kl_coef 0.0067 \
    --total_epochs 10 \
    --save_steps 2000 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --init_value_with_reward True \
    --evaluation_strategy "steps"

