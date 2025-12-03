#!/bin/bash
python3 -c "import transformers; print('Transformers version:', transformers.__version__)"


QWEN2_5_OMNI_MODEL_PATH=$(python3 -c "import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as m; print(m.__file__)" 2>/dev/null)
QWEN2_5_OMNI_PROCESS_PATH=$(python3 -c "import transformers.models.qwen2_5_omni.processing_qwen2_5_omni as p; print(p.__file__)" 2>/dev/null)

if [ -n "$QWEN2_5_OMNI_MODEL_PATH" ]; then
    cp -f omni_utils/transformers/modeling_qwen2_5_omni.py $QWEN2_5_OMNI_MODEL_PATH
    cp -f omni_utils/transformers/processing_qwen2_5_omni.py $QWEN2_5_OMNI_PROCESS_PATH
    echo "QWEN2_5_OMNI_MODEL replaced."
else
    echo "Qwen2_5_OmniModel Not Found in transformers. Make sure you have installed the correct version."
fi