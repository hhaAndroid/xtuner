set -ex

cpu=128
mem=1100
node_num=${1:-"32"}  # 2

script=${2:-"/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/debug_397_sh/run_train_interns2.sh"}
job_name=${3:-$(basename ${script%.sh})}

# IMAGE_URL=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt28_20260227_c4befa9
# IMAGE_URL=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_a9c0c04
IMAGE_URL=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47

PIP_INDEX_URL=http://mirrors.h.pjlab.org.cn/pypi/simple
PIP_TRUSTED_HOST=mirrors.h.pjlab.org.cn
gpus=8

export CLUSTERX_CFG_PATH=/mnt/shared-storage-user/huanghaian/clusterx_puyugpu.yaml
export PIP_INDEX_URL=$PIP_INDEX_URL
export PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST
clusterx run -N $node_num --gpus-per-task $gpus --priority 9 --job-name $job_name --cpus-per-task $cpu --memory-per-task $mem --no-env --image $IMAGE_URL \
  sh $script 
