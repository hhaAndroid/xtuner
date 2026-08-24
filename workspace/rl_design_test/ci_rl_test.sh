ray stop --force

cd /mnt/shared-storage-user/huanghaian/code/xtuner/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
# export PYTHONPATH="/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_rl/lmdeploy:$(pwd)"
export PYTHONPATH="/mnt/shared-storage-user/huanghaian/code/lmdeploy:$(pwd)"
export WORK_DIR='work_dirs_rl/qwen3vl_8b_grpo_mixdata3'

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck'

META_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/meta_data/ci_train_rl_mix_data.json'
export EVAL_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/meta_data/ci_eval_rl_mix_data.json'

# export XTUNER_DETERMINISTIC=1
export DEBUG_ROLLOUT_DIR='work_dirs_rl/debug_rollout'
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False

export XTUNER_USE_FA3=1

# export ONLY_CALC_MISMATCH_RATIO=1
# export TRAIN_BATCH_SIZE=8
# export CUDA_VISIBLE_DEVICES=2,3

# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3p5_vl_35B_grpo_mixdata.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
# bash examples/v1/scripts/run_rl.sh workspace/configs/rl_qwen3p5_vl_35B_dapo_ep2_resume.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
bash examples/v1/scripts/run_rl.sh workspace/configs/rl_qwen3p5_vl_35B_dapo_debug.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
