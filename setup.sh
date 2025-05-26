#!/bin/bash
export MODEL=Omni
pypi_mirror="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync --no-build-isolation-package flash-attn -i $pypi_mirror
source .venv/bin/activate
# uv pip install flash_attn --no-build-isolation -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


# 检查 transformers 包的版本
echo "正在检查 transformers 包的版本..."
python3 -c "import transformers; print('Transformers 版本:', transformers.__version__)"

if [ "$MODEL" = "Omni" ]; then
    # 使用 Python 查找 Qwen2_5_OmniModel 类的定义文件
    QWEN2_5_OMNI_MODEL_PATH=$(python3 -c "import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as m; print(m.__file__)" 2>/dev/null)

    # 检查是否成功获取路径
    if [ -n "$QWEN2_5_OMNI_MODEL_PATH" ]; then
        export QWEN2_5_OMNI_MODEL_PATH
        echo "QWEN2_5_OMNI_MODEL_PATH 环境变量已设置为: $QWEN2_5_OMNI_MODEL_PATH"
        cp -f transformers/modeling_qwen2_5_omni.py $QWEN2_5_OMNI_MODEL_PATH
        echo "QWEN2_5_OMNI_MODEL环境替换完成"
    else
        echo "未找到 Qwen2_5_OmniModel 类。请确保已安装包含该类的 transformers 版本。是否安装？[Y/n]"
        read -r answer
        if [[ "$answer" == "Y" || "$answer" == "y" || -z "$answer" ]]; then
            uv pip install transformers/transformers_omni.zip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
            echo "已安装 transformers omni 版本。"
            QWEN2_5_OMNI_MODEL_PATH=$(python3 -c "import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as m; print(m.__file__)" 2>/dev/null)
            echo "QWEN2_5_OMNI_MODEL_PATH 环境变量已设置为: $QWEN2_5_OMNI_MODEL_PATH"
            cp -f transformers/modeling_qwen2_5_omni.py $QWEN2_5_OMNI_MODEL_PATH
            echo "QWEN2_5_OMNI_MODEL环境替换完成"
        else
            echo "未修改环境。退出..."
        fi
    fi
fi

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# pip install -r requirements.txt

# cd src/qwen-omni-utils
# pip install ".[decord]"
# cd ../../


# # 如果使用qwen2.5-vl 使用下面这行命令
# if [ "$MODEL" = "25VL" ]; then
#     pip install transformers_vl.zip
# else
#     pip install transformers_omni.zip
# fi

# if [ "$MODEL" = "Omni" ]; then
#     # 使用 Python 查找 Qwen2_5_OmniModel 类的定义文件
#     QWEN2_5_OMNI_MODEL_PATH=$(python3 -c "import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as m; print(m.__file__)" 2>/dev/null)

#     # 检查是否成功获取路径
#     if [ -n "$QWEN2_5_OMNI_MODEL_PATH" ]; then
#         export QWEN2_5_OMNI_MODEL_PATH
#         echo "QWEN2_5_OMNI_MODEL_PATH 环境变量已设置为: $QWEN2_5_OMNI_MODEL_PATH"
#         cp -f modeling_qwen2_5_omni.py $QWEN2_5_OMNI_MODEL_PATH
#         echo "QWEN2_5_OMNI_MODEL环境替换完成"
#     else
#         echo "未找到 Qwen2_5_OmniModel 类。请确保已安装包含该类的 transformers 版本。"
#     fi
# fi

# pip install flash-attn --no-build-isolation




# # vLLM support 
# pip install vllm==0.7.2

# cd ../../ && unzip transformers-main.zip && cd transformers-main
# pip install .
# cd .. && rm -rf transformers-main
# fix transformers version
# pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef