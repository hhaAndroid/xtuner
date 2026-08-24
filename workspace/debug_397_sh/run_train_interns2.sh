set -ex
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

source /mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/interns2_rl_397b/bin/activate

cd /mnt/shared-storage-user/huanghaian/code/xtuner

export DEBUG_ROLLOUT_DIR='work_dirs_rl/debug_rollout'
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False

export PYTHONUNBUFFERED=1
# export HF_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
# export HUGGINGFACE_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_EVALUATE_OFFLINE=1
# export HF_HUB_OFFLINE=1

lmdeploy_dir=/mnt/shared-storage-user/huanghaian/code/lmdeploy
xtuner_dir=/mnt/shared-storage-user/huanghaian/code/xtuner/
# intern_s2_delivery_dir=/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_xtuner_rl_design/crg_rl_projects/src
LAGENT_PATH="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent"

export PYTHONPATH=$lmdeploy_dir:$xtuner_dir:$PYTHONPATH:$LAGENT_PATH 

export XTUNER_USE_FA3=1
export XTUNER_ACTIVATION_OFFLOAD=1
export PERMUTE_COMPUTE_DTYPE=fp32
export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1

export XTUNER_RL_MEM_INTERVAL=20
export XTUNER_ASYNCIO_DIAGNOSTICS=1
export XTUNER_DEBUG_OFFLOAD_MEMORY=1
export XTUNER_DEBUG_FSDP_DEFERRED=1

export WORK_DIR="work_dirs_rl_397_1/debug_397_sh"
export MODEL_PATH="/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2/sft_official/sft_interns2_base02_20260625a_lr2e5_512gpu/20260625041103/hf-5338"
export DATA_PATH="/mnt/shared-storage-user/huanghaian/code/xtuner/ci_train_rl_mix_data.json"

export NUM_WORKERS=256 #

export XTUNER_DEBUG_MEMORY_STAGE_LIVE_TENSORS=1
export XTUNER_DEBUG_MEMORY_DEEP_REFERRERS=1
export XTUNER_DEBUG_MEMORY_REFERRER_TOPK=4
export XTUNER_DEBUG_MEMORY_REFERRER_DEPTH=5
export XTUNER_DEBUG_MEMORY_REFERRER_PATHS=40

bash examples/v1/scripts/run_rl.sh workspace/debug_397_sh/reasoning_rl_qwen3p5_mtp_ep.py "lmdeploy" $MODEL_PATH $DATA_PATH
