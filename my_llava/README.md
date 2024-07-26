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

