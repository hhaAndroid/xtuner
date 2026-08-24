ray stop --force

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
# export PYTHONPATH="/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_rl/lmdeploy:$(pwd)"
export PYTHONPATH="/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_rl/mtp_rl_dev/lmdeploy:$(pwd):/mnt/shared-storage-user/huanghaian/code/temp/xtuner/projects"
export WORK_DIR='work_dirs_rl/qwen3vl_8b_grpo_claw'

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/wangziyi/exp/mindcopilot_rl/work_dirs/ckpt/sft_interns2_pre_base03_20260413a_lr2e5_128gpu/20260414020822/hf-5646'
export QWEN3P5_VL_MODEL_PATH="/mnt/shared-storage-user/llmit1/user/wangziyi/exp/mindcopilot_rl/work_dirs/ckpt/interns2-35ba3-base05-20260424a-rl-data260428rc0-56k-badword-mtp4-resume800/20260430074140/hf-40"
export TRAIN_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/agent_dev/meta_data.json'
export RL_LLM_MODEL='xtuner-hha-qwen35-30b'

# export ENABLE_RETURN_ROUTED_EXPERTS=1
# export XTUNER_DETERMINISTIC=1

export XTUNER_USE_FA3=1
export DEFAULT_LAGENT_SRC='/mnt/shared-storage-user/huanghaian/code/gateway/lagent'

# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3p5_vl_35B_grpo_mixdata.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
bash examples/v1/scripts/run_rl.sh examples/v1/config/agent_rl_qwen35_30b_grpo.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $TRAIN_DATA_PATH

# export MODEL_PATH="/mnt/shared-storage-user/llmit1/user/wangziyi/exp/mindcopilot_rl/work_dirs/ckpt/interns2-35ba3-base05-20260424a-rl-data260428rc0-56k-badword-mtp4-resume800/20260430074140/hf-40"
# python examples/v1/config/agent_rl_qwen35_30b_grpo.py
