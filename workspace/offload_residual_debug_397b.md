# 正常训练路径 Offload Residual 调试说明

目标：排查未设置 `ONLY_CALC_MISMATCH_RATIO` 时，正常 `train_step()` 之后 offload 仍残留显存的问题。

当前已知现象：

- 35B 正常路径：offload 后约 `65 MB`。
- 397B 正常路径：offload 后约 `1958 MB`。
- 这和 `ONLY_CALC_MISMATCH_RATIO=1` 的 MTP prefetch 问题不是同一个证据形态。

## 1. 35B 现有证据

从 `20260626113647/logs/rank_0.log` 看，第一次正常训练后：

```text
[after_model_to_cpu_before_deferred_fsdp_release] CUDA allocator stats:
allocated=65.08 MB, active_large=64.00 MB, active_small=1.08 MB, active_allocs=10
```

对应 active block snapshot：

```text
01: size=32.00 MB requested=32.00 MB segment=large stream=0 <no python frames>
02: size=32.00 MB requested=32.00 MB segment=large stream=0 ... modeling_vision.py:145:forward
03: size=1.00 MB requested=1.00 MB segment=small stream=0 ... modeling_vision.py:145:forward
04: size=0.08 MB requested=0.08 MB ... aux_loss.py:_cal_tokens_per_expert
```

同时 Python live tensor 扫描只有约 `0.08 MB`：

```text
live CUDA tensors visible to Python: total=0.08 MB
```

所以 35B 的 65MB 不是 Python 可见的 FSDP `AllGatherResult`，也不是 model 参数没有 offload；它更像是 allocator 里仍 active 的非 Python tensor / 底层 op workspace / autograd 或 kernel 侧持有 block。35B 里最大栈指向 vision attention 的：

```text
xtuner/v1/model/compose/qwen3_vl/modeling_vision.py:145
self.qkv(hidden_states)
```

但不能直接把 397B 的 `1958 MB` 归因到同一处，必须让 397B 打出自己的 active block stack。

## 2. 397B 推荐运行环境变量

不要设置 `ONLY_CALC_MISMATCH_RATIO`：

```bash
unset ONLY_CALC_MISMATCH_RATIO
```

建议只打开 offload 边界 probe，不要打开全局 memory stages：

```bash
export XTUNER_DEBUG_OFFLOAD_RESIDUAL=1
export XTUNER_DEBUG_OFFLOAD_RESIDUAL_MAX_PROBES=9
export XTUNER_DEBUG_OFFLOAD_MEMORY_SNAPSHOT=1
export XTUNER_DEBUG_ACTIVE_BLOCK_OWNERS=1
export XTUNER_DEBUG_MEMORY_SNAPSHOT_BLOCK_LIMIT=30
export XTUNER_DEBUG_MEMORY_SNAPSHOT_FRAME_LIMIT=12

unset XTUNER_DEBUG_FSDP_DEFERRED
unset XTUNER_DEBUG_MEMORY_STAGES
unset XTUNER_DEBUG_UPDATE_WEIGHT_MEMORY
unset XTUNER_DEBUG_MEMORY_STAGE_LIVE_TENSORS
```

说明：

- `XTUNER_DEBUG_OFFLOAD_RESIDUAL=1`：只在 `offload_model()` 边界打详细信息。
- `MAX_PROBES=9`：每次 offload 打 3 个 probe，9 次覆盖前三次 offload，通常包括初始化、首次权重同步、第一次真实训练后残留。
- `SNAPSHOT=1`：打印 active CUDA blocks 的分配栈。
- `ACTIVE_BLOCK_OWNERS=1`：尝试把 active block 地址反查到 Python tensor storage。
- offload residual probe 会在 offload 边界强制检查 FSDP deferred，不需要全局打开 `XTUNER_DEBUG_FSDP_DEFERRED`。
- 不开 `XTUNER_DEBUG_MEMORY_STAGES`，避免每个训练和权重同步 bucket 都打 snapshot，397B 日志会非常大。

## 3. 跑完后重点看哪些日志

每个 offload 会出现三组 tag：

```text
[offload_model/before_model_to_cpu]
[offload_model/after_model_to_cpu_before_deferred_release]
[offload_model/after_deferred_release]
```

关键字段：

```text
CUDA allocator stats:
allocated=...
active_large=...
active_small=...
active_allocs=...

CUDA memory snapshot active blocks:
01: size=... requested=... segment=... stream=... <python stack>

active CUDA block Python owner probe:
01: block=... owners=...

model parameters/buffers still on CUDA:
total=... count=...

live CUDA tensors visible to Python:
total=...

deferred FSDP all-gather states:
...
```

判断顺序：

1. 如果 `model parameters/buffers still on CUDA` 非 0，说明有模型参数或 buffer 没被 offload。
2. 如果 `deferred FSDP all-gather states` 非空，说明还是 FSDP deferred state。
3. 如果 `live CUDA tensors visible to Python` 接近残留大小，说明是 Python tensor 引用，要看 referrers。
4. 如果 live tensor 很小但 `CUDA memory snapshot active blocks` 很大，说明是非 Python tensor owner 或底层 op/workspace，要根据 active block stack 判断模块。
5. 如果 397B 的最大 block stack 也指向 `modeling_vision.py:145`，再回头分析 vision qkv output / attention backend 生命周期。
6. 如果最大 block stack 指向 language model、lm_head、MTP、FSDP/DTensor convert、lmdeploy IPC 或其他路径，就按那个模块继续缩小。

## 4. 需要保留的 log 片段

如果完整 log 太大，至少保留第一次出现大残留的 offload 附近：

```bash
rg -n "offload_model/(before_model_to_cpu|after_model_to_cpu_before_deferred_release|after_deferred_release)|CUDA allocator stats|CUDA memory snapshot active blocks|active CUDA block Python owner probe|model parameters/buffers still on CUDA|live CUDA tensors visible to Python|deferred FSDP all-gather states|Offloaded model to CPU" rank_0.log
```

如果能给完整 `rank_0.log` 更好；如果只截片段，建议截第一次 `Offloaded model to CPU. Current allocate 1958...` 前后至少 200 行。
