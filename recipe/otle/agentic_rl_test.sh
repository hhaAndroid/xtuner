ray stop --force

cd /mnt/shared-storage-user/huanghaian/code/xtuner/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

CRG_PATH="/mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/"
LAGENT_PATH="/mnt/shared-storage-user/huanghaian/code/gateway/lagent"
LMDEPLOY_PATH="/mnt/shared-storage-user/huanghaian/code/lmdeploy"

export PYTHONPATH="${CRG_PATH}:${LAGENT_PATH}:${LMDEPLOY_PATH}:$(pwd):${PYTHONPATH:-}"

export WORK_DIR='work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code'

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
export TRAIN_BATCH_SIZE=8
export PROMPT_REPEAT_K=2
# export CUDA_VISIBLE_DEVICES=2,3
export ENABLE_INITIAL_EVALUATE=False
export TOTAL_TRAIN_STEPS=2

export MODEL_NAME="hha_xtuner_train_agentic_rl_qwen3p5vl_mtp_ep_code_${RUN_ID}"
export RL_LLM_MODEL="$MODEL_NAME"
export XTUNER_OTEL_RUN_ID="${XTUNER_OTEL_RUN_ID:-$MODEL_NAME}"
export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1

export XTUNER_ASYNCIO_DIAG_TASK_LIMIT=500
export XTUNER_ASYNCIO_DIAG_STACK_LIMIT=16
export XTUNER_ASYNCIO_DIAG_AWAIT_DEPTH=32
export XTUNER_ASYNCIO_RUN_WATCHDOG_S=60
export XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S=60
export XTUNER_ASYNCIO_DIAGNOSTICS=1
export XTUNER_ASYNCIO_RUN_WATCHDOG_S=600
export XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S=600

export COMPASS_VERIFIER_V2_HOSTS="10.102.217.35:23333,10.102.217.35:23334,10.102.217.35:23335,10.102.217.35:23336,10.102.217.35:23337,10.102.217.35:23338,10.102.217.35:23339,10.102.217.35:23340"

# export AGENT_OTEL_ENABLED=1
# export OTEL_TRACES_EXPORTER=otlp
# export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
# export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://10.102.250.69:4318/v1/traces

export AGENT_OTEL_ENABLED=1
export XTUNER_OTEL_ENABLED=1
export AGENT_OTEL_DEBUG=1

export OTEL_SERVICE_NAME=agent-rollout-trace
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://10.102.250.69:14318/v1/traces

export XTUNER_OTEL_SERVICE_NAME=xtuner-session-server
export XTUNER_OTEL_RUN_ID=v1_tp1 # baseline or v1_tp1

bash examples/v1/scripts/run_rl.sh examples/v1/config/agentic_rl_qwen3p5vl_mtp_ep_code.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
