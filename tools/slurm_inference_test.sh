set -x

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 30000))
echo $MASTER_ADDR
echo $MASTER_PORT

#  -m debugpy --connect 10.140.0.31:15689
echo "单卡 static"
srun -p llm_razor --gres=gpu:1 --time 1:00:00 python qwen0_5_inference_static.py
echo "双卡 static"
srun -p llm_razor --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=16 --time 1:00:00 python qwen0_5_inference_static.py
echo "单卡 dynamic"
srun -p llm_razor --gres=gpu:1 --time 1:00:00 python qwen0_5_inference_dynamic.py
echo "双卡 dynamic"
srun -p llm_razor --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=16 --time 1:00:00 python qwen0_5_inference_dynamic.py
