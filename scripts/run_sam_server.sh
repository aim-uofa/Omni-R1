export CUDA_VISIBLE_DEVICES=2,3
# export HF_ENDPOINT="https://hf-mirror.com"

python src/sam_api/sam_server.py \
    --port 32224 \
    --processes 4 \
    --host 0.0.0.0 \
    --use_video