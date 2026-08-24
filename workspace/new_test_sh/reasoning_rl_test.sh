ray stop --force

cd /mnt/shared-storage-user/huanghaian/code/xtuner/
LAGENT_PATH="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent"

# source /mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/interns2_rl_397b/bin/activate

# export PATH=/usr/local/nvidia/bin/:$PATH
# export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export CUDA_HOME=/mnt/shared-storage-user/huanghaian//cuda/cuda-13.2.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# export PYTHONPATH="/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_rl/lmdeploy:$(pwd)"
export PYTHONPATH="/mnt/shared-storage-user/huanghaian/code/lmdeploy:$(pwd):$LAGENT_PATH"
export WORK_DIR='work_dirs_rl3/qwen3vl_8b_grpo_mixdata3-lmdeploy-bs8-mtp'

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck'

META_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/xtuner/ci_train_rl_mix_data.json'
export EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/ci_eval_rl_mix_data.json'

# export XTUNER_DETERMINISTIC=1
export DEBUG_ROLLOUT_DIR='work_dirs_rl/debug_rollout'
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False

export XTUNER_USE_FA3=1

export TRAIN_BATCH_SIZE=8
# export CUDA_VISIBLE_DEVICES=1,2,3
export ENABLE_INITIAL_EVALUATE=False
export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1

export XTUNER_ACTIVATION_OFFLOAD=1

bash examples/v1/scripts/run_rl.sh examples/v1/config/reasoning_rl_qwen3p5vl_mtp_ep.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
