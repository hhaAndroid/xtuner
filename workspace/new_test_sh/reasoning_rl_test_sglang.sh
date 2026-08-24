set -euo pipefail

source /mnt/shared-storage-user/huanghaian/miniconda3/etc/profile.d/conda.sh
conda activate pt2121_all_en_sglang

XTUNER_REPO=/mnt/shared-storage-user/huanghaian/code/temp/xtuner
cd "$XTUNER_REPO"

# pt2121_all_en_sglang 环境

# export PATH=/usr/local/nvidia/bin/:$PATH
# export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/shared-storage-user/huanghaian/cuda/cuda-13.2.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SGLANG_REPO="${SGLANG_REPO:-/mnt/shared-storage-user/huanghaian/code/slime_package/xtuner_sglang/sglang}"
export PYTHONPATH="${SGLANG_REPO}/python:${SGLANG_REPO}:${XTUNER_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export WORK_DIR='work_dirs_rl/qwen3vl_8b_grpo_mixdata3-sglang-prefix-cache'

# SGLang prefix cache. When enabled, use the cache-aware longest-prefix-match
# scheduler so repeated prompts from prompt_repeat_k can reuse their prefixes.
export ENABLE_SGLANG_PREFIX_CACHE=True
export SGLANG_PREFIX_CACHE_SCHEDULE_POLICY=lpm

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck'

META_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/ci_train_rl_mix_data.json'
export EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/ci_eval_rl_mix_data.json'

# export XTUNER_DETERMINISTIC=1
export DEBUG_ROLLOUT_DIR='work_dirs_rl/debug_rollout'
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False
# export ONLY_CALC_MISMATCH_RATIO=1

export XTUNER_USE_FA3=1

# export ONLY_CALC_MISMATCH_RATIO=1
export TRAIN_BATCH_SIZE=16
# export CUDA_VISIBLE_DEVICES=1,2,3
export ENABLE_INITIAL_EVALUATE=False

export XTUNER_ASYNCIO_DIAGNOSTICS=1

bash examples/v1/scripts/run_rl.sh \
    examples/v1/config/reasoning_rl_qwen3p5vl_mtp_ep_sglang.py \
    sglang \
    "$QWEN3P5_VL_MODEL_PATH" \
    "$META_DATA_PATH"
