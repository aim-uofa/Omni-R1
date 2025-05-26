#!/bin/bash
export WORK_DIR=$(pwd)

cd src/r1-v

export SAM_HOST="172.21.211.2" # Set the host address of SAM server
export SAM_PORT="32224" # Set the port of SAM server

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_MODE="true" # Enable Log if you want to save the rollout of model during RL
export FUSE_TEMPERATURE="false" # Enable temperature fusion
export AVS_TEMPERATURE="0.45"

export WANDB_MODE="offline" # Enable wandb if you want to use the wandb in the video dataset
export PLOG='false'

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1


export LOG_PATH="train_logs/omni_r1"
mkdir -p "${WORK_DIR}/${LOG_PATH}"
export TRAIN_PATH="${WORK_DIR}/${LOG_PATH}"

# For resume training:  --resume_from_checkpoint Model_Path \
# training_datasets 'ReVOS' 'MeVIS' 'SegZero' 'RefAVS' \
# use_avs_sample_number 1600 \
# use_multi_vos true: use multi-object part of VOS datasets as extra training data, default is false
# use_prompt_forcing: prompt model to generate 4 frames and descriptions for each video, default is false

# use_peft is not supported yet, so set it to false
# Qwen/Qwen2.5-Omni-3B

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12434" \
    src/open_r1/grpo.py \
    --output_dir $TRAIN_PATH \
    --model_name_or_path 'Qwen/Qwen2.5-Omni-7B' \
    --datasets_json 'datasets.json' \
    --training_datasets 'ReVOS' 'MeVIS' 'SegZero' 'RefAVS' \
    --use_multi_vos false \
    --use_prompt_forcing false \
    --use_avs_sample_number 1600 \
    --deepspeed local_scripts/zero3_offload.json \
    --use_peft false \
    --max_prompt_length 32768 \
    --max_completion_length 2048 \
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
    --num_train_epochs 1 \
    --run_name omni_r1 \
    --save_steps 200 \
    --beta 0.04 \
    --alpha_k 1.0 \
    --alpha_k_ratio 1.0 \
    --alpha_a 1.0 \
    --alpha_g 0.0 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
