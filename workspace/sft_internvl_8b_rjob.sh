
REPLICAS=64
name=n$REPLICAS-internvl-8b-sft6

fs=huanghaian  # 前面创建的文件系统
fs1=llmrazor-share
fs2=large-model-center-share-weights
fs3=intern7shared
fs4=intern-multi-modal-delivery
fs5=llmit
fs6=caoweihan
fs7=songdemin
fs8=yehaochen
fs9=duanyanhui
fs10=gaozhangwei
fs11=puyullmgpu-shared

rjob delete $name
rjob submit --name $name -P $REPLICAS --gpu 8 --cpu 160  --memory 1600000 --charged-group puyullmgpunew_gpu \
    --image=registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194 \
    --mount=gpfs://gpfs1/${fs}:/mnt/shared-storage-user/${fs} \
    --mount=gpfs://gpfs1/${fs1}:/mnt/shared-storage-user/${fs1} \
    --mount=gpfs://gpfs1/${fs2}:/mnt/shared-storage-user/${fs2} \
    --mount=gpfs://gpfs1/${fs3}:/mnt/shared-storage-user/${fs3} \
    --mount=gpfs://gpfs1/${fs4}:/mnt/shared-storage-user/${fs4} \
    --mount=gpfs://gpfs1/${fs5}:/mnt/shared-storage-user/${fs5} \
    --mount=gpfs://gpfs1/${fs6}:/mnt/shared-storage-user/${fs6} \
    --mount=gpfs://gpfs1/${fs7}:/mnt/shared-storage-user/${fs7} \
    --mount=gpfs://gpfs1/${fs8}:/mnt/shared-storage-user/${fs8} \
    --mount=gpfs://gpfs1/${fs9}:/mnt/shared-storage-user/${fs9} \
    --mount=gpfs://gpfs1/${fs10}:/mnt/shared-storage-user/${fs10} \
    --mount=gpfs://gpfs1/${fs11}:/mnt/shared-storage-user/${fs11} \
    --namespace=ailab-puyullmgpunew \
    --private-machine='group' \
    --gang-start=true \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -e DISTRIBUTED_JOB=true \
    --host-network=true \
    -- bash -ecx /mnt/shared-storage-user/huanghaian/code/xtuner/sft_internvl_3p5_8b_xtuner_more.sh
