#!/bin/bash
set -e

# $(pwd)
export WORK_DIR=../outputs/

cd src

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_MODE="true" # Enable Log if you want to save the rollout of model during RL
export FUSE_TEMPERATURE="false" # Enable temperature fusion
export AVS_TEMPERATURE="0.45"
export WANDB_MODE="offline" # Enable wandb if you want to use the wandb in the video dataset
export SWANLAB_MODE="offline" 
export PLOG='false'

# if using vllm as grounding model, set the remote address
export USE_VLLM=1
export VLLM_SERVER_PATH=http://33.182.81.100:12428

# use local sam or set remote sam server
export USE_LOCAL_SAM=1
export SAM_PATH=sam2.1-hiera-large/sam2.1_hiera_large.pt

# or use remote sam server
# export USE_LOCAL_SAM=0
# export SAM_HOST=10.12.99.38
# export SAM_PORT=32224

# export CLIP_REWARD=1

# to load qwen2_5omni_thinker ckpt
# export USE_THINKER=1

## for grounding prompt, if official qwen2.5omni prompt is not preferred,

# use segzero style grounding prompt
# export USE_GROUNDING_PROMPT_OMNI_SEGZERO=1

# or use vision-reasoner style grounding prompt
export USE_GROUNDING_PROMPT_VISIONRESONER=1

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export LOG_PATH="train_logs/"
mkdir -p "${WORK_DIR}/${LOG_PATH}"
export TRAIN_PATH="${WORK_DIR}/${LOG_PATH}"

export MODEL_PATH=/weight/Qwen2.5-Omni-7B

# For resume training:  --resume_from_checkpoint Model_Path \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12435" \
    omni_r1/grpo.py \
    --output_dir $TRAIN_PATH \
    --model_name_or_path $MODEL_PATH \
    --datasets_json 'datasets_demo.json' \
    --training_datasets 'RefAVS' 'MeVIS' 'ReVOS' \
    --use_avs_sample_number 1600 \
    --use_multi_vos false \
    --use_prompt_forcing true \
    --deepspeed local_scripts/zero3_offload.json \
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
    --num_train_epochs 1 \
    --run_name omni_r1 \
    --save_steps 100 \
    --beta 0.04 \
    --alpha_k 1.0 \
    --alpha_k_ratio 0.0 \
    --alpha_a 1.0 \
    --alpha_g 2.0 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 8
