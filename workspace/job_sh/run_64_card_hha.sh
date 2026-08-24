set -ex

ray stop --force

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

cd /mnt/shared-storage-user/huanghaian/code/xtuner/
source /mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/interns2_rl_397b/bin/activate

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export HF_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
export HUGGINGFACE_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

lagent_dir=/mnt/shared-storage-user/huanghaian/code/gitlab/lagent
bootcamp_dir=/mnt/shared-storage-user/llmit/user/guyuzhe/20250708/lvchengqi/projects/moe_rl/InternBootcamp
lmdeploy_dir=/mnt/shared-storage-user/huanghaian/code/lmdeploy
# lmdeploy_dir=/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_xtuner_rl_design/lmdeploy
xtuner_dir=/mnt/shared-storage-user/huanghaian/code/xtuner
intern_s2_delivery_dir=/mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/src

export PYTHONPATH=$lagent_dir:$intern_s2_delivery_dir:$lmdeploy_dir:$xtuner_dir:$bootcamp_dir:$PYTHONPATH 
export NLTK_DATA=/mnt/shared-storage-user/llmit/user/lishuaibin/mv2yidian/nltk_data

export XTUNER_USE_FA3=1
export XTUNER_MAX_CONCURRENCY=8192
export RAY_MAX_CONCURRENCY=8192
export XTUNER_LOG_LEVEL="INFO"
export UVICORN_LOG_LEVEL="CRITICAl"
export PERMUTE_COMPUTE_DTYPE=fp32
export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1

if [ "$XTUNER_USE_SGLANG" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi

export MAX_CONCURRENT=128
export XTUNER_ASYNCIO_DIAGNOSTICS=1

CONFIG_PATH="workspace/job_sh/interns2-35ba3-base05-20260424a-rl-data260426rc1-56k-badword-mtp4-0528.py"
MODEL_PATH="fake_interns2_35b"
DATA_PATH="fake_interns2_data"

export WORK_DIR="work_dirs_rl_6/interns2-35ba3-base05-20260424a-rl-data260426rc1-56k-badword-mtp4-0528"
export GLOBAL_BATCH_SIZE=1024 # 1024
export NUM_WORKERS=64 # 8

export RAY_record_ref_creation_sites=1

bash examples/v1/scripts/run_rl_submit.sh $CONFIG_PATH "lmdeploy" $MODEL_PATH $DATA_PATH