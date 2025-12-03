#!/bin/bash
set -e

export WORK_DIR=$(pwd)

source .venv/bin/activate
cd src

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_MODE="true" # Enable Log if you want to save the rollout of model during RL
export FUSE_TEMPERATURE="false" # Enable temperature fusion
export AVS_TEMPERATURE="0.45"
export WANDB_MODE="offline" # Enable wandb if you want to use the wandb in the video dataset
export PLOG='false'

export USE_VLLM=1
# h100_1 172.27.44.74
# h100_4 172.21.211.4 172.27.45.131
export VLLM_SERVER_PATH=http://172.21.212.4:12428
export SAM_HOST=172.21.211.4

export USE_LOCAL_SAM=0
export CLIP_REWARD=1



# export PGROUND=1
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export LOG_PATH="train_logs/no_kl_vos_no_hint_e_3"
mkdir -p "${WORK_DIR}/${LOG_PATH}"
export TRAIN_PATH="${WORK_DIR}/${LOG_PATH}"

# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.

# --max_pixels 1000000 \
#  'SegZero' 'MeVIS'


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12435" \
    -m omni_r1.grpo \
    --output_dir $TRAIN_PATH \
    --model_name_or_path '/mnt/public/weight/Qwen2.5-Omni-7B/' \
    --resume_from_checkpoint '/mnt/public/home/zhonghao/Omni-R1/train_logs/no_kl_vos_no_hint_e_3/checkpoint-600' \
    --datasets_json 'datasets_h100.json' \
    --training_datasets 'ReVOS' 'MeVIS' \
    --use_multi_vos false \
    --use_prompt_forcing false \
    --deepspeed local_scripts/zero2.json \
    --max_prompt_length 32768 \
    --max_completion_length 800 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name no_kl_vos_no_hint_e3 \
    --save_steps 200 \
    --beta 0.0 \
    --alpha_k 1.0 \
    --alpha_k_ratio 0.0 \
    --alpha_a 1.0 \
    --alpha_g 2.0 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8 \

