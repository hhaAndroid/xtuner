ray stop --force

cd /mnt/shared-storage-user/huanghaian/code/xtuner/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

CRG_PATH="/mnt/shared-storage-user/huanghaian/code/agent_dev/crg_rl_projects/"
LAGENT_PATH="/mnt/shared-storage-user/huanghaian/code/agent_dev/lagent"
SGLANG_REPO="${SGLANG_REPO:-/mnt/shared-storage-user/huanghaian/code/slime_package/xtuner_sglang/sglang}"

export PYTHONPATH="${CRG_PATH}:${LAGENT_PATH}:${SGLANG_REPO}/python:${SGLANG_REPO}:$(pwd):${PYTHONPATH:-}"

export WORK_DIR='work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang'

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck'

META_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/math_code_meta.json'

# export XTUNER_DETERMINISTIC=1
export DEBUG_ROLLOUT_DIR='work_dirs_rl/debug_rollout'
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False

export XTUNER_USE_FA3=1

# export ONLY_CALC_MISMATCH_RATIO=1
export TRAIN_BATCH_SIZE=16
export PROMPT_REPEAT_K=8
# export CUDA_VISIBLE_DEVICES=2,3
export ENABLE_INITIAL_EVALUATE=False
export TOTAL_TRAIN_STEPS=20

export MODEL_NAME="hha_xtuner_train_agentic_rl_qwen3p5vl_mtp_ep_code_${RUN_ID}"
export RL_LLM_MODEL="$MODEL_NAME"
export RL_LLM_BASE_URL="${RL_LLM_BASE_URL:-http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1}"
export RL_LLM_API_KEY="${RL_LLM_API_KEY:-sk-admin}"
export XTUNER_OTEL_RUN_ID="${XTUNER_OTEL_RUN_ID:-$MODEL_NAME}"
export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1

# export XTUNER_ASYNCIO_DIAG_TASK_LIMIT=500
# export XTUNER_ASYNCIO_DIAG_STACK_LIMIT=16
# export XTUNER_ASYNCIO_DIAG_AWAIT_DEPTH=32
# export XTUNER_ASYNCIO_RUN_WATCHDOG_S=60
# export XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S=60
# export XTUNER_ASYNCIO_DIAGNOSTICS=1
# export XTUNER_ASYNCIO_RUN_WATCHDOG_S=600
# export XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S=600

export COMPASS_VERIFIER_V2_HOSTS="10.102.139.22:12345,10.102.139.22:12346,10.102.139.22:12347,10.102.139.22:12348,10.102.139.22:12349,10.102.139.22:12350,10.102.139.22:12351,10.102.139.22:12352"

# export AGENT_OTEL_ENABLED=1
# export XTUNER_OTEL_ENABLED=1
# export AGENT_OTEL_DEBUG=1

# export OTEL_SERVICE_NAME=agent-rollout-trace
# export OTEL_TRACES_EXPORTER=otlp
# export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
# export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://10.102.250.69:14318/v1/traces

# export XTUNER_OTEL_SERVICE_NAME=xtuner-session-server
# export XTUNER_OTEL_RUN_ID=v1_tp1

#  1. SGLang 的 routed experts 返回格式和 LMDeploy 不一样
#      SessionServer 原来把字符串形式的 routed_experts 当成 LMDeploy 的 Ray shared_store key 去查，所以 SGLang 下会报找不到 shared_store。
#      修复：在 xtuner/v1/rl/rollout/session_server.py:543 先按 SGLang 的 base64 int32 tensor 解码，失败再 fallback 到 LMDeploy shared_store。
#   2. max_tokens=65536 在 SGLang 下越界
#      lagent 请求里 completion 给了 65536，但 SessionServer 还会把 messages 转成 input_ids，SGLang 会检查 input + completion + EAGLE reserved 是否超过 context。LMDeploy 会自动 cap，SGLang 直接 400。
#      修复：SessionServer 按 context_length - input_ids_len - safety_margin 下调 max_tokens/max_completion_tokens；worker 会按 EAGLE 配置计算 reserved token。
#   3. top_k=0 语义不兼容
#      XTuner/LMDeploy 里 top_k=0 表示 disable，但 SGLang OpenAI server 要求 top_k=-1 或 >=1。直连 SGLangWorker 原本有转换，SessionServer 路径绕过了它。
#      修复：只在 SGLang SessionServer 路径把 top_k=0 规范化成 -1，见 xtuner/v1/rl/rollout/session_server.py:201 和 xtuner/v1/rl/rollout/worker.py:884。

export XTUNER_SESSION_SERVER_STALL_LOG_S=30
export XTUNER_SESSION_SERVER_SLOW_WRITE_LOG_S=2

bash examples/v1/scripts/run_rl.sh examples/v1/config/agentic_rl_qwen3p5vl_mtp_ep_code_sglang.py "sglang" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
