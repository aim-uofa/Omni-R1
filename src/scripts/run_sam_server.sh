export CUDA_VISIBLE_DEVICES=0,1
# export HF_ENDPOINT="https://hf-mirror.com"

cd src/r1-v
uv run src/sam_api/sam_server.py \
    --port 32224 \
    --processes 8 \
    --host 0.0.0.0 \
    --use_video