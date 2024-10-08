# 运行预训练

```shell
conda activate xtuner_23
CUDA=12.0 GCC=7.5 version_control
cd my_llava
bash shell/llava_pretrain_internlm2_7b.sh
```

只使用 batch packing 情况下 新版本预训练 2 小时 55 分钟，旧版本预训练3小时30分钟，占比 16.7%。

```shell
conda activate xtuner_23
CUDA=12.0 GCC=7.5 version_control
cd my_llava
bash shell/llava_pretrain_internlm2_7b_pack.sh
```

在使用 batch packing + soft packing 情况下 新版本预训练 2 小时 57 分钟，没有提速，主要原因是数据长度非常均衡

# 运行 sft

```shell
conda activate xtuner_23
CUDA=12.0 GCC=7.5 version_control
cd my_llava
bash shell/llava_sft_internlm2_7b.sh
```

只使用 batch packing 情况下 新版本预训练 6小时32分钟，旧版本预训练8小时0分钟,快了 1小时28分钟，占比 18.5%。

```shell
conda activate xtuner_23
CUDA=12.0 GCC=7.5 version_control
cd my_llava
bash shell/llava_sft_internlm2_7b_packing.sh
```

在使用 batch packing + soft packing 情况下 新版本预训练 5 小时 40 分钟，相比于旧版本快了 2 小时 20 分钟，占比 29.2%。

# 评测

```shell
cd my_llava
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .

export LMUData='/mnt/hwfile/xtuner/huanghaian/LMUData/'
export PYTHONPATH="$(pwd):$(pwd)/../"
srun -p llm_razor --job-name=eval --time=02:00:00 --cpus-per-task=16 --nodes=1 --gres=gpu:8 --ntasks-per-node=1 --kill-on-bad-exit=1 torchrun --master_port=29501 --nproc-per-node=8 vlmevalkit/run.py --data MMBench_DEV_EN MMStar SEEDBench_IMG MMMU_DEV_VAL ScienceQA_TEST TextVQA_VAL ChartQA_TEST AI2D_TEST DocVQA_VAL InfoVQA_VAL OCRBench RealWorldQA SEEDBench2_Plus HallusionBench --model-path work_dirs/llava_sft_internlm2_7b/20240918105412/hf-0100-of-5198
```

# 基本用法说明

- 在安装了 flash attention 情况下，一定会开启 batch packing，这个可以无痛提速，推荐
- 在数据长度不均衡情况下，建议使用 soft packing，这个可以提速，但是考虑到 soft packing 情况下，总的 step 和 token 都改变了，可能 lr 超参要改一下
- 目前如果不开启 soft packing，那么数据是不需要缓存下来的，没有很大必要，这会导致启动时间变慢
- 目前已经支持 resume，只需要 --resume 即可，如果想指定某个 checkpoint，可以通过 --resume-from 参数指定
- 考虑到在保存了优化器等状态情况下，会需要非常多存储，因此默认的 --max-keep-ckpts 是 1

# 序列并行

暂时代码没有合并

# InternVL

目前仅仅支持了其中一个模型作为验证对象

## SFT 

```shell
conda activate xtuner_23
cd my_llava
CUDA=12.0 GCC=7.5 version_control
bash shell/internvl1_5_phi3_sft.sh
```

## 评测

```shell
cd my_llava

export LMUData='/mnt/hwfile/xtuner/huanghaian/LMUData/'
export PYTHONPATH="$(pwd):$(pwd)/../"

# 由于 internvl 推理代码不支持最新 transformers,因此需要额外一步
# 覆盖保存的 modeling_phi3.py
# 修改 shell/copy.sh 里面的路径后执行
bash shell/copy.sh
# 如果想多机，可以考虑把多个评测数据集分开，每个节点跑一部分评测集
srun -p llm_razor --job-name=eval --time=02:00:00 --cpus-per-task=16 --nodes=1 --gres=gpu:8 --ntasks-per-node=1 --kill-on-bad-exit=1 torchrun --master_port=29501 --nproc-per-node=8 vlmevalkit/run.py --model-flag internvl_1_5 --data MMBench_DEV_EN MMStar MME SEEDBench_IMG MMMU_DEV_VAL ScienceQA_TEST HallusionBench TextVQA_VAL ChartQA_TEST AI2D_TEST DocVQA_VAL InfoVQA_VAL OCRBench RealWorldQA SEEDBench2_Plus --model-name internvl1 --model-path work_dirs/internvl1_5_phi3_sft/20240914165647/hf-15780-of-15780

# 自动解析全部数据
srun -p llm_razor python vlmevalkit/parse_eval_csv.py internvl1
```
