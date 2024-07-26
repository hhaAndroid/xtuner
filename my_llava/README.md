# 运行预训练

```shell
conda activate xtuner_23
CUDA=12.0 GCC=7.5 version_control
cd my_llava
bash shell/llava_pretrain_internlm2_7b.sh
```

只使用 batch packing 情况下 新版本预训练 2 小时 55 分钟，旧版本预训练3小时30分钟，占比 16.7%。

# 运行 sft

```shell
conda activate xtuner_23
CUDA=12.0 GCC=7.5 version_control
cd my_llava
bash shell/llava_sft_internlm2_7b.sh
```

只使用 batch packing 情况下 新版本预训练 6小时32分钟，旧版本预训练8小时0分钟,快了 1小时28分钟，占比 18.5%。

# 评测

```shell
MASTER_PORT=29501 srun -p llm_razor --job-name=eval --time=02:00:00 --cpus-per-task=16 --nodes=1 --gres=gpu:8 --ntasks-per-node=1 --kill-on-bad-exit=1 torchrun --nproc-per-node=8 vlmevalkit/run.py --data MMBench_DEV_EN MMStar SEEDBench_IMG MMMU_DEV_VAL ScienceQA_TEST TextVQA_VAL ChartQA_TEST AI2D_TEST DocVQA_VAL InfoVQA_VAL OCRBench RealWorldQA SEEDBench2_Plus HallusionBench --model-path work_dirs/llava_sft_internlm2_7b/20240725194745/hf-5198-of-5198
```
