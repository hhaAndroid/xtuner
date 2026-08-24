# XTuner `swap_optimizer` 实现记录

本文整理本次关于 `self.optim_cfg.swap_optimizer` 的讨论，方便后续回看实现细节。

## 结论概览

`swap_optimizer=True` 只作用在 `AdamWConfig` 上：构建优化器时不再返回 `torch.optim.AdamW`，而是返回自定义的 `SwapAdamW`。

核心定位是 optimizer state offload：

```text
optimizer state 常驻 CPU pinned memory
optimizer 计算仍在 GPU/NPU 上完成
```

它 swap 的是 Adam 的状态，例如 `exp_avg` / `exp_avg_sq`，不是模型参数、梯度或 activation。

## 入口链路

相关文件：

- `xtuner/v1/config/optim.py`
- `xtuner/v1/optim/swap_adamw.py`
- `xtuner/v1/engine/train_engine.py`

`TrainEngine.__init__` 中构建顺序大致是：

```python
self.model = self.build_model()
self.optimizer = self.build_optimizer(optim_cfg)
```

`AdamWConfig.build()` 中：

```python
if self.swap_optimizer:
    return SwapAdamW(
        params,
        lr=self.lr,
        betas=self.betas,
        eps=self.eps,
        weight_decay=self.weight_decay,
        foreach=self.foreach,
    )

return torch.optim.AdamW(...)
```

所以训练循环本身不变，仍然是：

```text
forward/backward -> clip_grad_norm -> optimizer.step() -> optimizer.zero_grad()
```

差异集中在 `SwapAdamW.step()` 内部。

## 普通 AdamW 的懒初始化

PyTorch optimizer 主要看两个变量：

```python
optimizer.param_groups
optimizer.state
```

普通 `torch.optim.AdamW(model.parameters())` 初始化后：

```text
param_groups: 有，保存模型参数对象引用
state: 空 defaultdict(dict)
```

`param_groups` 里的 `params` 只是模型参数对象的引用，不复制参数数据。因此普通 AdamW 初始化时不会因为 `param_groups` 多占一份模型显存。

AdamW 的大状态是懒初始化的：第一次 `optimizer.step()` 时，遇到有 grad 的参数，才创建：

```text
exp_avg
exp_avg_sq
step
```

这些状态通常创建在参数同设备上。参数在 GPU 上时，普通 AdamW 的 `exp_avg` / `exp_avg_sq` 也会常驻 GPU。

## `SwapAdamW._init_swap_states()` 做了什么

`SwapAdamW` 继承 `torch.optim.AdamW`，但在 `__init__` 中主动调用 `_init_swap_states()`。

这一步打破了普通 AdamW 的懒初始化：初始化 optimizer 时就为所有 trainable 参数创建 optimizer state。

状态变化可以理解为：

```text
进入 _init_swap_states 前:
    self.state == defaultdict(dict, {})

访问 self.state[param] 后:
    self.state[param] == {}

填充完成后:
    self.state[param]["exp_avg"] = CPU pinned zero tensor
    self.state[param]["exp_avg_sq"] = CPU pinned zero tensor
    self.state[param]["max_exp_avg_sq"] = None  # 非 amsgrad
    self.state[param]["step"] = CPU float32 tensor(0.0)
```

关键代码逻辑：

```python
local_param = self._to_local_tensor(param)

cpu_tensor = torch.zeros_like(local_param, memory_format=torch.preserve_format)
cpu_tensor = cpu_tensor.to(device="cpu", non_blocking=True)
cpu_tensor = cpu_tensor.pin_memory()
```

因此 `SwapAdamW` 初始化完成后，optimizer state 是真实 CPU pinned tensor，不是 meta，也不是空。

这里有一个隐含前提：`local_param` 不能是 meta tensor。如果参数还是 meta：

```python
torch.zeros_like(local_param)  # 得到 meta tensor
.to("cpu")                    # 会报 Cannot copy out of meta tensor
```

所以这个实现不是“先在 meta 上建 state，后面再 materialize”。它要求 `_init_swap_states()` 执行时参数已经有真实 storage。

## DTensor / FSDP 场景

`SwapAdamW` 里有：

```python
def _to_local_tensor(tensor):
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor
```

如果参数是 DTensor，则只对本 rank 的 local shard 创建 CPU optimizer state。也就是说，CPU 上保存的是本 rank 负责的 local optimizer state，不是 full tensor。

## `SwapAdamW.step()` 的执行方式

每次 step 时，它逐个参数处理：

```text
1. 从 CPU pinned memory 把 exp_avg / exp_avg_sq 搬到 GPU/NPU
2. 取 param 和 grad 的 local tensor
3. 调 torch.optim.adam.adam functional 在 device 上更新参数和状态
4. 把更新后的 exp_avg / exp_avg_sq copy 回 CPU pinned memory
5. device synchronize
```

简化伪代码：

```python
for param in params_list:
    if param.grad is None:
        continue

    exp_avg = cpu_exp_avg.to(device=DEVICE, non_blocking=True)
    exp_avg_sq = cpu_exp_avg_sq.to(device=DEVICE, non_blocking=True)

    torch_adam(
        [local_param],
        [local_grad],
        [local_exp_avg],
        [local_exp_avg_sq],
        ...
    )

    cpu_exp_avg.copy_(exp_avg, non_blocking=True)
    cpu_exp_avg_sq.copy_(exp_avg_sq, non_blocking=True)
```

## 为什么不直接在 CPU 上算 Adam

Adam 更新同时依赖：

```text
param
grad
exp_avg
exp_avg_sq
```

训练时 `param` 和 `grad` 都在 GPU/NPU 上。为了调用 device kernel 并原地更新参数，`exp_avg` / `exp_avg_sq` 也需要临时到同一个 device。

如果直接在 CPU 上算，就需要：

```text
GPU param -> CPU
GPU grad -> CPU
CPU 上 Adam 更新
CPU param -> GPU
```

这通常更差：

- 传输量更大，因为参数和梯度也要搬。
- CPU Adam 计算更慢。
- 参数更新路径会破坏当前 FSDP/DTensor/device mesh 下“参数在 device 上更新”的假设。

所以当前方案是：

```text
CPU 保存 optimizer state，GPU/NPU 执行 optimizer step。
```

核心原因可以简单记为：`grad` 和 `param` 已经在 GPU/NPU 上，所以 state 临时搬过去一起算更合理。

## GPU 峰值显存

当前实现没有真正分桶。optimizer-state 额外 GPU 峰值主要发生在 step 中：

```python
exp_avg = cpu_exp_avg.to(device=DEVICE, non_blocking=True)
exp_avg_sq = cpu_exp_avg_sq.to(device=DEVICE, non_blocking=True)
```

如果 `amsgrad=True`，还会多一个 `max_exp_avg_sq`。

因此额外 GPU 峰值大致是：

```text
2 * 当前单个 local_param.numel() * state_dtype_size
```

通常 XTuner 会把 trainable 参数转成 fp32，因此常见估算是：

```text
2 * 最大单个本地参数 numel * 4 bytes
```

这不是全模型 Adam state 的大小，而是“单参数更新窗口”的临时 state 峰值。

注意：虽然临时 tensor 生命周期很短，但 CUDA allocator 可能保留为 reserved memory，所以监控里可能看到 reserved memory 不马上下降。

## `self.swap_numel` 当前没有实际作用

代码里有：

```python
swap_optimizer_times: int = 16
self._swap_optimizer_times = swap_optimizer_times
self.swap_numel = swap_num // self._swap_optimizer_times
```

但全仓库没有其他地方读取 `swap_numel`。当前 `step()` 也是按参数逐个处理，而不是按 `swap_numel` 分 bucket。

所以现在的 `swap_numel` 更像遗留或预留字段。按命名推测，原设计可能想实现“把 optimizer state 切成若干 bucket，每次搬一块”，但目前没有落地。

当前真实行为：

```text
不是按 swap_numel 分桶
而是按 parameter 逐个 swap state
```

## 和 optimizer offload/onload 的关系

`TrainEngine.put_optimizer_to_device()` 中：

```python
if self.fsdp_cfg.cpu_offload or self.optim_cfg.swap_optimizer:
    return
```

所以当 `swap_optimizer=True` 时，外部调用 `offload_optimizer()` / `onload_optimizer()` 基本是 no-op。原因是 optimizer state 本来就常驻 CPU，不需要整体搬到 CPU 或 device。

## 需要额外留意的风险

`SwapAdamW` 没有 override `load_state_dict()`。

它内部除了 `self.state[param]`，还有一份：

```python
self._param_to_cpu_states_map[param]
```

如果 checkpoint resume 时 PyTorch / DCP 替换了 `self.state` 里的 tensor，但 `_param_to_cpu_states_map` 仍然指向初始化时的旧 CPU tensor，就可能出现恢复后 step 用的不是新加载的 state。

这个点需要单独验证。稳妥做法可能是在加载 optimizer state 后刷新 `_param_to_cpu_states_map`，让它重新指向 `self.state[param]` 中的 CPU state tensor。

