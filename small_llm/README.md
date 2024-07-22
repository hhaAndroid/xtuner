# 说明

- tokenizner 使用 qwen2
- model 结构使用 qwen2-1.5b
- 第一版本预训练数据使用 wanjuan 1.0 EN 部分和天工开源的部分中文数据

## 模型

https://arxiv.org/pdf/2407.10671

## 环境

## 原理记录

- dp_lazy_init 函数是用于对 fsdp 输入的一小块模型进行 参数 cuda 初始化，然后对初始化后的模型自动进行切片，才可以保证显存不会爆炸。 内部逻辑就是这样
- auto_wrap_policy 策略用于控制 fsdp 每次的通信量，太大可能会 oom,太小可能通信效率不高。模型切分策略是某一层在 rank0，某些层在 rank1,而不是 deepspeed 这种多卡均分的切分方式
- 