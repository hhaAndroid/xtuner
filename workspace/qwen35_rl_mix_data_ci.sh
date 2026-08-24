ray stop --force

# export PATH=/usr/local/nvidia/bin/:$PATH
# export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# export PYTHONPATH="/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_rl/lmdeploy:$(pwd)"
export PYTHONPATH="$(pwd)"
export WORK_DIR='work_dirs_rl1/qwen3vl_8b_grpo_gsm8k'

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
META_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/meta_data/rl_mix_data.json'

export ENABLE_RETURN_ROUTED_EXPERTS=1
# export XTUNER_DETERMINISTIC=1

export XTUNER_USE_FA3=1
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3p5_vl_35B_grpo_mixdata.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3p5_vl_35B_grpo_mixdata.py "sglang" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
