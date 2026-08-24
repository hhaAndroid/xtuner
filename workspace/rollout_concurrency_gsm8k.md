# `rl_grpo_gsm8k_judge.py` Rollout 并发梳理

本文只讨论 `examples/v1/config/rl_grpo_gsm8k_judge.py` 的 rollout 侧并发。训练侧流程不展开。

## 核心配置

这个配置里和 rollout 并发直接相关的值主要是：

```python
NNODE = int(os.environ.get("WORLD_SIZE", "1"))

resources.num_workers = 8 * NNODE

train_batch_size = 64
prompt_repeat_k = 5

rollout_tp_size = 1
rollout_ep_size = 1

max_prompt_length = 512
max_response_length = 1024

rollout_config = RolloutConfig(
    tensor_parallel_size=1,
    expert_parallel_size=1,
    context_length=512 + 1024,
)

produce_strategy_config = SyncProduceStrategyConfig()

judger_config = GSM8KJudgerConfig(
    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
)
```

默认 `WORLD_SIZE=1` 时，就是 8 张 GPU 做 rollout。

## 1. Rollout Engine 数

XTuner 里单个 rollout engine 使用多少 GPU，由 `tensor_parallel_size` 和 `expert_parallel_size` 决定：

```text
单个 engine 使用 GPU 数 = expert_parallel_size if expert_parallel_size > 1 else tensor_parallel_size
```

当前配置：

```text
tensor_parallel_size = 1
expert_parallel_size = 1
```

所以：

```text
单个 engine 使用 GPU 数 = 1
```

总 rollout worker 数是：

```text
rollout worker 数 = 8 * NNODE
```

因此 active rollout engine 数为：

```text
engine 数 = rollout worker 数 / 单个 engine 使用 GPU 数
          = 8 * NNODE / 1
          = 8 * NNODE
```

默认单节点时：

```text
engine 数 = 8
```

也就是说，本配置下不是一个请求跨多卡跑，而是 8 个独立的一卡推理 engine 同时服务请求。

## 2. 一轮 Rollout 的 Group 并发

这个配置使用 `SyncProduceStrategyConfig()`，所以 rollout 是同步按需生产：一次 `produce_batch()` 里生成当前需要的 rollout 数据。

传给 `produce_batch()` 的 batch size 是：

```text
train_batch_size = 64
```

在 rollout 语义里，这里的 `64` 是 64 个 prompt group，不是 64 条最终 sample。

`SyncProduceStrategy` 初始会为每个 prompt group 启动一个异步 group task：

```text
初始 group task 数 = train_batch_size
                  = 64
```

所以一轮 rollout 开始时，先有 64 个 group 并发生成。

## 3. 每个 Group 内的 Sample 并发

每个 prompt group 内会重复采样：

```text
prompt_repeat_k = 5
```

`SingleTurnAgentLoop.generate_group()` 会把这 5 条 sample 拆成 5 个独立异步任务，并发调用 `generate_sample()`。

所以每个 group 内：

```text
sample task 数 = prompt_repeat_k
               = 5
```

整个 rollout window 的初始 generation request 数为：

```text
generation request 数 = group task 数 * 每个 group 的 sample task 数
                      = train_batch_size * prompt_repeat_k
                      = 64 * 5
                      = 320
```

这里要注意：`prompt_repeat_k=5` 不是 backend 的 `n=5`。当前 `SampleParams.n` 仍然是默认值 `1`。XTuner 是在外层发出 5 个独立 generation request。

## 4. 请求如何分到各个 Engine

每条 sample task 会调用：

```text
rollout_controller.generate(...)
```

controller 再通过 `SessionRouter` 把请求路由到某个 active rollout worker/server。

默认单节点时有 8 个 engine，因此平均每个 engine 需要承接：

```text
平均每个 engine 请求数 = generation request 数 / engine 数
                       = 320 / 8
                       = 40
```

如果是多节点：

```text
平均每个 engine 请求数 = 320 / (8 * NNODE)
```

例如：

```text
WORLD_SIZE=1: 8 个 engine，平均每个 engine 约 40 个请求
WORLD_SIZE=2: 16 个 engine，平均每个 engine 约 20 个请求
WORLD_SIZE=4: 32 个 engine，平均每个 engine 约 10 个请求
```

这是平均估算值。真实分布会受 session 路由、worker 是否 active、请求完成时间、失败重试等因素影响。

## 5. XTuner Ray 层并发上限

这个配置没有显式设置 `rollout_max_batch_size_per_instance`，XTuner 会根据 `context_length` 自动推导。

当前：

```text
context_length = max_prompt_length + max_response_length
               = 512 + 1024
               = 1536
```

推导规则是：

```text
context_length <= 4096  -> rollout_max_batch_size_per_instance = 1024
context_length <= 8192  -> rollout_max_batch_size_per_instance = 512
otherwise               -> rollout_max_batch_size_per_instance = 128
```

所以当前：

```text
rollout_max_batch_size_per_instance = 1024
```

`allow_over_concurrency_ratio` 默认是 `1.2`。

单个 active rollout worker 的 Ray generate 并发上限为：

```text
max(1000, ceil(1024 * 1.2)) = 1229
```

rollout controller 的 generate 并发上限为：

```text
engine 数 * ceil(1024 * 1.2)
```

默认单节点：

```text
controller generate 并发上限 = 8 * 1229 = 9832
```

而当前正常 rollout window 只有 320 个 generation request：

```text
320 << 9832
```

因此这个配置下，XTuner Ray 层的并发上限远高于实际请求数，通常不是限制因素。

## 6. Backend Server 内部并发

XTuner 这一层会把 320 个 generation request 作为并发 HTTP 请求打到各个 rollout backend server。

backend 由环境变量选择：

```text
XTUNER_USE_LMDEPLOY=1
XTUNER_USE_VLLM=1
XTUNER_USE_SGLANG=1
```

当前配置推导到 backend 侧，大致是：

```text
vLLM:    max_num_seqs = 1024
LMDeploy: max_batch_size = 1024
SGLang:  max_running_requests = 1024
```

这表示单个 backend server 理论上可以接收/调度的请求上限很高。实际执行时，backend 会自己做排队、prefill batching、decode batching 和调度。

所以 “320 个并发 request” 更准确地说是 HTTP 请求层面的并发，不等于 GPU 上同时有 320 条完全独立的执行流。

## 7. Judger 并发

GSM8K judger 的配置是：

```text
num_workers = 1
num_cpus_per_worker = 1
```

所以 rollout 后处理阶段只有 1 个 remote judger actor。

正常无失败时，一轮 rollout 会生成 320 条 sample，因此也会有：

```text
judge 调用数 = 320
judger actor 数 = 1
```

GSM8K 的 judge 逻辑只是本地抽取答案并比较 ground truth，通常很轻。但如果以后换成远程 reward model 或更重的 reward function，这里的 `num_workers=1` 就可能成为 rollout 后处理瓶颈。

## 8. 失败、Abort 和补发

`SyncProduceStrategy` 不是只发 64 个 group 就结束，而是要拿够 64 个 completed group。

目标是：

```text
completed group 数 = 64
```

正常无失败：

```text
实际 group task 数 = 64
实际 generation request 数 = 64 * 5 = 320
```

如果有 group 失败、abort、无效响应或被过滤，那么会继续补发 group：

```text
实际 group task 数 > 64
实际 generation request 数 > 320
```

所以 320 是正常情况下的首轮/目标请求数，不是严格上限。

## 9. Eval Rollout 并发

eval 使用另一个 sampler，配置里：

```text
eval_prompt_repeat_k = 1
```

所以 eval 没有 “每个 prompt 并发采样 5 条” 这一层。

eval rollout 的请求数是：

```text
eval generation request 数 = eval prompt group 数 * 1
```

也就是 evaluator 要评多少个 prompt，就发多少个 generation request。

## 当前默认配置总结

默认 `WORLD_SIZE=1`，且没有失败补发时：

```text
rollout engine 数 = 8

group task 数 = 64
每个 group 内 sample task 数 = 5

generation request 数 = 64 * 5 = 320

平均每个 engine request 数 = 320 / 8 = 40

单个 rollout worker Ray generate 并发上限 = 1229
rollout controller Ray generate 并发上限 = 9832

judger actor 数 = 1
judge 调用数 = 320
```

核心并发图：

```text
64 个 prompt group 并发
  每个 group 内 5 条 sample 并发
    => 320 个 generation request

320 个 request 分到 8 个一卡 engine
    => 平均每个 engine 约 40 个 request
```

