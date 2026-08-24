ray stop --force

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export PYTHONPATH="/mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/:/mnt/shared-storage-user/huanghaian/code/lmdeploy:$(pwd)"
# export PYTHONPATH="$(pwd)"
export WORK_DIR='work_dirs_rl/qwen3vl_35b_grpo_mixdata'
export LAGENT_SRC_DIR='/mnt/shared-storage-user/huanghaian/code/gateway/lagent'

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck'
#/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/data/tb2_rl_tasks.jsonl
META_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/meta_data/rl_mix_data.json'
# META_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/meta_data/tb2.json'

# EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl'
EVAL_DATA_PATH=''

# export XTUNER_DETERMINISTIC=1
export XTUNER_USE_FA3=1

export DEBUG_ROLLOUT_DIR='work_dirs_rl/debug_rollout'
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False

# export ROLLOUT_ROUTER_MODE='url_pool_worker'
# export ROLLOUT_ROUTER_STICKY_SESSION=1

# export ROLLOUT_ROUTER_MODE='url_pool_session_server'

export ROLLOUT_ROUTER_MODE='third_party_session_server'
export ROLLOUT_THIRD_PARTY_ROUTED_URL='http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1'

# export ROLLOUT_ROUTER_STICKY_SESSION=0

export RL_LLM_MODEL='xtuner_qwen3p5_vl_35b_hha'
export MODEL_NAME=$RL_LLM_MODEL
export SANDBOX_PROVIDER_KEY='huangha-kdio28HD'
# session_id=os.environ["XTUNER_SESSION_ID"],
# bash examples/v1/scripts/run_rl.sh examples/v1/config/agent_tb2_rl_qwen3p5_rl.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH $EVAL_DATA_PATH
# bash examples/v1/scripts/run_rl.sh examples/v1/config/agent_tb2_rl_qwen3p5_rl.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH $EVAL_DATA_PATH
bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3p5_vl_35B_grpo_mixdata.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH $EVAL_DATA_PATH
