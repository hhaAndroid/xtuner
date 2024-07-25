# 运行预训练

```shell
conda activate xtuner_23
CUDA=12.0 GCC=7.5 version_control
cd my_llava
bash shell/llava_pretrain_internlm2_7b.sh
```

只使用 batch packing 情况下 新版本预训练 2小时55分钟，旧版本预训练3小时30分钟。

# 运行 sft

