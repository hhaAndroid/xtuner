set -euo pipefail

source /mnt/shared-storage-user/huanghaian/miniconda3/etc/profile.d/conda.sh
conda activate pt2121_all_en_sglang

XTUNER_REPO=/mnt/shared-storage-user/huanghaian/code/temp/xtuner
cd "$XTUNER_REPO"

export CUDA_HOME=/mnt/shared-storage-user/huanghaian/cuda/cuda-13.2.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CRG_PATH="/mnt/shared-storage-user/huanghaian/code/agent_dev/crg_rl_projects/"
LAGENT_PATH="/mnt/shared-storage-user/huanghaian/code/agent_dev/lagent"
SGLANG_REPO="${SGLANG_REPO:-/mnt/shared-storage-user/huanghaian/code/slime_package/xtuner_sglang/sglang}"

export PYTHONPATH="${CRG_PATH}:${LAGENT_PATH}:${SGLANG_REPO}/python:${SGLANG_REPO}:$(pwd):${PYTHONPATH:-}"

export WORK_DIR="${WORK_DIR:-work_dirs_rl_agentic/agentic_rl_qwen3p5vl_mtp_ep_code_sglang}"

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck'

META_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/math_code_meta.json'

# export XTUNER_DETERMINISTIC=1
export DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-work_dirs_rl/debug_rollout}"
export DEBUG_TRAIN="${DEBUG_TRAIN:-False}"
export DEBUG_ROLLOUT="${DEBUG_ROLLOUT:-False}"

export XTUNER_USE_FA3=1

# export ONLY_CALC_MISMATCH_RATIO=1
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PROMPT_REPEAT_K="${PROMPT_REPEAT_K:-8}"
export MAX_CONCURRENT_SAMPLES="${MAX_CONCURRENT_SAMPLES:-$((TRAIN_BATCH_SIZE * PROMPT_REPEAT_K))}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export ENABLE_RETURN_ROUTED_EXPERTS=True
# export CUDA_VISIBLE_DEVICES=2,3
export ENABLE_EVALUATE="${ENABLE_EVALUATE:-False}"
export ENABLE_INITIAL_EVALUATE="${ENABLE_INITIAL_EVALUATE:-False}"
export TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-2}"

# Shared A/B knobs. Keep these values identical to agentic_rl_test.sh.
export TRAIN_EP_SIZE="${TRAIN_EP_SIZE:-4}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
export ROLLOUT_EXPERT_PARALLEL_SIZE="${ROLLOUT_EXPERT_PARALLEL_SIZE:-1}"
export ROLLOUT_CONTEXT_LENGTH="${ROLLOUT_CONTEXT_LENGTH:-69632}"
export SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-3}"
# With EAGLE topk=1, SGLang requires draft_tokens = steps + 1.  The default
# therefore preserves the prior 2-draft-token baseline.  Set both knobs to
# 4/3 respectively when matching LMDeploy's 3 draft candidates.
export SGLANG_SPECULATIVE_NUM_STEPS="${SGLANG_SPECULATIVE_NUM_STEPS:-$((SPECULATIVE_NUM_DRAFT_TOKENS - 1))}"
export MAMBA_PREFIX_CACHE_STATE_INTERVAL="${MAMBA_PREFIX_CACHE_STATE_INTERVAL:-256}"
# Supported values: extra_buffer (overlap enabled) or no_buffer (overlap disabled).
export SGLANG_MAMBA_RADIX_CACHE_STRATEGY="${SGLANG_MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"
# Set to True to disable radix/prefix caching while preserving the selected
# Mamba strategy and overlap scheduling for a controlled cache A/B test.
export SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-False}"

export RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
export MODEL_NAME="hha_xtuner_train_agentic_rl_qwen3p5vl_mtp_ep_code_${RUN_ID}"
export RL_LLM_MODEL="$MODEL_NAME"
export RL_LLM_BASE_URL="${RL_LLM_BASE_URL:-http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1}"
export RL_LLM_API_KEY="${RL_LLM_API_KEY:-sk-admin}"
export XTUNER_OTEL_RUN_ID="${XTUNER_OTEL_RUN_ID:-$MODEL_NAME}"
export SWAP_OPTIMIZER="${SWAP_OPTIMIZER:-True}"

export COMPASS_VERIFIER_V2_HOSTS="${COMPASS_VERIFIER_V2_HOSTS:-10.103.23.52:12345,10.103.23.52:12346,10.103.23.52:12347,10.103.23.52:12348,10.103.23.52:12349,10.103.23.52:12350,10.103.23.52:12351,10.103.23.52:12352}"

bash examples/v1/scripts/run_rl.sh \
    examples/v1/config/agentic_rl_qwen3p5vl_mtp_ep_code_sglang.py \
    sglang \
    "$QWEN3P5_VL_MODEL_PATH" \
    "$META_DATA_PATH"
