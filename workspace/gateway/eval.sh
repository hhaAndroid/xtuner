ray stop --force
ulimit -n 65536  # OSError: [Errno 24] Too many open files

export PYTHONPATH=$(pwd):$PYTHONPATH
export RAY_MAX_CONCURRENCY=${RAY_MAX_CONCURRENCY:-1024}
export PYTHONUNBUFFERED=1

python recipe/skillsbench_agentloop/run_eval.py \
    --skillsbench-root /mnt/shared-storage-user/huanghaian/code/bench/skillsbench \
    --env-gateway-base-url http://env-gateway.ailab.ailab.ai \
    --model-path /mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B \
    --output-root ./work_dirs_1/skillsbench-eval \
    --task-name 3d-scan-calc \
    --gateway-port 38100 \
    --api-port 28000 \
    --tensor-parallel-size 2 \
    --set-env XTUNER_USE_LMDEPLOY=1 \
    --set-env LMD_SKIP_WARMUP=1 \
    --set-env XTUNER_USE_FA3=1
