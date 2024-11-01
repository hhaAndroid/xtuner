set -x

export PYTHONPATH="$(pwd):$(pwd)/../"
#  -m debugpy --connect 10.140.0.31:15689
echo "================================================"
echo "单卡 static"
CUDA_VISIBLE_DEVICES=0 python qwen0_5_inference_static.py -s
echo "================================================"
echo "双卡 static"
CUDA_VISIBLE_DEVICES='0,1' torchrun --nproc-per-node=2 qwen0_5_inference_static.py
echo "================================================"
echo "单卡 dynamic"
CUDA_VISIBLE_DEVICES=0 python qwen0_5_inference_dynamic.py -s
echo "================================================"
echo "双卡 dynamic"
CUDA_VISIBLE_DEVICES='0,1' torchrun --nproc-per-node=2  qwen0_5_inference_dynamic.py
echo "================================================"
echo "单卡 hf static"
CUDA_VISIBLE_DEVICES=0 python qwen0_5_inference_static_complex.py -s
echo "================================================"
echo "双卡 hf static"
CUDA_VISIBLE_DEVICES='0,1' torchrun --nproc-per-node=2 qwen0_5_inference_static_complex.py
