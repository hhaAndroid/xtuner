ray stop --force

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
# export PYTHONPATH="/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_rl/lmdeploy:$(pwd)"
export PYTHONPATH="/mnt/shared-storage-user/huanghaian/code/lmdeploy:$(pwd)"
export WORK_DIR='work_dirs_rl/qwen3vl_8b_grpo_mixdata3-lmdeploy'

# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_coldstart_g64/20260315020444/hf-175'
export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307'
# export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck'

META_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/ci_train_rl_mix_data.json'
export EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/ci_eval_rl_mix_data.json'

# export XTUNER_DETERMINISTIC=1
export DEBUG_ROLLOUT_DIR='work_dirs_rl/debug_rollout'
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False

export XTUNER_USE_FA3=1

# export ONLY_CALC_MISMATCH_RATIO=1

export TRAIN_BATCH_SIZE=8
# export CUDA_VISIBLE_DEVICES=1,2,3
export ENABLE_INITIAL_EVALUATE=False
export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1

# export XTUNER_ASYNCIO_RUN_WATCHDOG_S=60
# export XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S=60
export XTUNER_ASYNCIO_DIAGNOSTICS=1
export XTUNER_RL_MEM_INTERVAL=10

export XTUNER_ACTIVATION_OFFLOAD=1
# export XTUNER_DESTROY_TRAIN_NCCL_AFTER_SYNC=1

export SWAP_OPTIMIZER=True
export XTUNER_SUSPEND_TRAIN_NCCL_AFTER_SYNC=1
export XTUNER_SUSPEND_TRAIN_NCCL_INCLUDE_DEFAULT=1

bash examples/v1/scripts/run_rl.sh examples/v1/config/reasoning_rl_qwen3p5vl_mtp_ep.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
# 630s -> 592s
# 59g   -> 99g

# XTUNER_SUSPEND_TRAIN_NCCL_AFTER_SYNC=1  2.5g
# orig 4.2g
# XTUNER_SUSPEND_TRAIN_NCCL_AFTER_SYNC=1 + XTUNER_SUSPEND_TRAIN_NCCL_INCLUDE_DEFAULT=1  2.2
