# ONLY_CALC_MISMATCH_RATIO 下 Offload 后残留显存问题复盘

日期：2026-06-27

相关运行：

- 脚本：`/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/new_test_sh/reasoning_rl_test.sh`
- 配置：`examples/v1/config/reasoning_rl_qwen3p5vl_mtp_ep.py`
- 关键日志：
  - `work_dirs_rl/qwen3vl_8b_grpo_mixdata3-lmdeploy/20260626094241/logs/rank_0.log`
  - `work_dirs_rl/qwen3vl_8b_grpo_mixdata3-lmdeploy/20260626100721/logs/rank_0.log`
  - `work_dirs_rl/qwen3vl_8b_grpo_mixdata3-lmdeploy/20260627014307/logs/rank_0.log`

## 1. 问题是什么

这次问题只在打开诊断开关时明显暴露：

```bash
export ONLY_CALC_MISMATCH_RATIO=1
```

这个开关的预期行为是：训练侧只计算 old logprob、mismatch、rollout importance sampling 指标，然后提前 return，不进入正常 `train_step()`、backward 和 optimizer step。

预期上，这条路径虽然少跑了训练，但不应该在 train/rollout colocate 切换后多留一块长期 GPU 显存。实际现象相反：35B 上 offload 后稳定残留约 `1.6 GiB`

35B 旧日志里的典型现象：

```text
Offloaded model to CPU. Current allocate 1644.13671875 MB, reserved: 1670.0 MB
```

这个残留不是启动时天然存在。初始化阶段 `skip_load_weights=True` 触发首次权重同步后，offload 是干净的：

```text
Offloaded model to CPU. Current allocate 0.0009765625 MB, reserved: 4.0 MB
Rollout workers updated weights from train workers.
```

所以第一层结论是：问题不是“模型大所以 offload 必然留这么多”，也不是首次权重同步本身导致，而是 `ONLY_CALC_MISMATCH_RATIO=1` 的 old-logprob/mismatch 路径跑完之后留下了某个训练侧临时状态。

## 2. 先排除几个误导方向

几个环境事实先明确：

- `use_kl_loss=False`，没有 ref model。
- 用户确认没有 DeepEP。
- rollout 设置了 `skip_load_weights=True`，但首次权重同步后 offload 干净。
- 问题发生在 colocate 模式下 train worker 切回 rollout 前的 model offload 阶段。

曾经怀疑过 loss context、old logprobs、shifted labels、seq ctx 等 Python 引用。尝试在 `ONLY_CALC_MISMATCH_RATIO=1` 提前 return 前删除这些对象并执行 `gc.collect()` / `empty_cache()` 后，残留仍在，因此普通 loss/context 引用不是主因。

也怀疑过 `lm_head.weight` 的权重同步 IPC buffer，因为日志里经常能看到：

```text
handling same hf param: ['lm_head.weight'] separately
Rollout workers update weights successfully in colocate mode
Offloaded model to CPU. Current allocate 1644.x MB
```

这个方向有一定合理性，因为 lmdeploy 权重同步会创建 CUDA IPC flat tensor；但后面的 Python referrer 证明最终持有者不是 `FlattenedTensorBucket`。

## 3. 第一个关键证据：offload 后还有真实 live CUDA tensor

为了区分 allocator cache 和真实 tensor 引用，在 `offload_model()` 后加了 gated debug，扫描 Python 可见 live CUDA tensors。

日志显示，`memory_allocated()` 里的主要部分确实是一个仍被 Python 对象持有的 CUDA tensor：

```text
[offload_model] live CUDA tensors visible to Python: total=1611.13 MB unique_storages=8

01: 1611.05 MB shape=(844655104,) dtype=torch.bfloat16 device=cuda:0 type=Tensor requires_grad=False
02: 0.08 MB shape=(40, 256) dtype=torch.int64 device=cuda:0 type=Tensor requires_grad=False
...
```

这说明：

- `1644 MB` 残留里约 `1611 MB` 是真实 CUDA tensor，不是 reserved cache。
- tensor 是 1D bf16 flat buffer：`844655104 * 2 bytes = 1611.05 MiB`。
- `empty_cache()` 不可能释放它，因为还有 Python/FSDP 对象引用。

## 4. 第二个关键证据：referrer 指向 FSDP2 AllGatherResult

继续打印最大 CUDA tensor 的 referrer，看到：

```text
largest CUDA tensor referrers:

01: type=tuple indices=[6]
02: type=tuple indices=[0]
03: type=AllGatherResult indices=[0]
```

PyTorch FSDP2 的 `AllGatherResult` 定义里，第 0 个字段就是：

```python
class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    ...
```

因此可以确定：这块 `1611 MiB` 是 FSDP2 all-gather output flat buffer。

到这里还不能直接说是 `comm_ctx.all_gather_state`。早期文档里把它写成 `comm_ctx`，这是不准确的。后续专门加了 `XTUNER_DEBUG_FSDP_DEFERRED=1` 来区分：

- `comm_ctx.all_gather_state`
- `FSDPParamGroup._all_gather_result`

## 5. 第三个关键证据：真正残留在 MTP layer 的 FSDPParamGroup 上

最新日志 `20260627014307/logs/rank_0.log` 直接打出了 deferred holder 和匹配到的 param group。

在 old-logprob 前：

```text
[rollout_1/before_compute_actor_logprobs] deferred FSDP all-gather states: none
```

old-logprob forward 后立刻出现：

```text
[rollout_1/after_compute_actor_logprobs] deferred FSDP all-gather states: count=1

param_group@0x7ed2c57ff7a0 module=language_model.mtp_block.layers.0:
  holder=FSDPParamGroup,
  result=AllGatherResult,
  output=1611.05 MB shape=(844655104,) dtype=torch.bfloat16 device=cuda:0
```

在 `ONLY_CALC_MISMATCH_RATIO=1` 提前 return 前仍然存在：

```text
[rollout_1/before_only_calc_mismatch_return] deferred FSDP all-gather states: count=1
param_group@0x7ed2c57ff7a0 module=language_model.mtp_block.layers.0: ...
```

进入权重同步前也已经存在：

```text
[update_weights/begin] deferred FSDP all-gather states: count=1
param_group@0x7ed2c57ff7a0 module=language_model.mtp_block.layers.0: ...
```

这一步很重要：它说明这块残留不是 update_weights 新产生的。update_weights 只是带着这个残留继续往后走。

offload 释放时的日志也证明命中的不是 `comm_ctx`：

```text
[offload_model] released deferred FSDP all-gathers: comm_states=0, param_groups=1
FSDPCheckpointWrapper._fsdp_param_group._all_gather_result:
  1611.05 MB shape=(844655104,) dtype=torch.bfloat16 device=cuda:0

Offloaded model to CPU. Current allocate 33.0849609375 MB, reserved: 70.0 MB
[offload_model/after_deferred_release] deferred FSDP all-gather states: none
```

所以最终精确结论是：

```text
language_model.mtp_block.layers.0
  -> FSDPParamGroup._all_gather_result
     -> AllGatherResult.all_gather_output
        -> bf16 tensor, shape=(844655104,), about 1611 MiB
```

不是：

```text
comm_ctx.all_gather_state
```

## 6. 为什么会是 MTP layer

先看 old-logprob 路径。`TrainingWorker.compute_actor_logprobs()` 调：

```python
output = self._engine.forward_only(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
```

`TrainEngine.forward_only()` 实际只传 LM loss context：

```python
@torch.no_grad()
def forward_only(self, seq_ctx: SequenceContext, loss_ctx: LogProbContext):
    output = self.model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
    return output
```

也就是说 old-logprob forward 不传 `mtp` loss context。

MoE 模型 forward 里，MTP 分支只有在 `loss_ctx` 中存在 `mtp` 时才会运行：

```python
if (
    self.mtp_block is not None
    and loss_ctx is not None
    and (mtp_loss_ctx_list := loss_ctx.get("mtp")) is not None
):
    mtp_outputs = self.mtp_block(...)
    ...
```

因此 `ONLY_CALC_MISMATCH_RATIO=1` 的 old-logprob forward 会跑主 LM forward 和 LM loss，但不会真正执行 MTP block forward。

同时，FSDP shard 初始化时为了 overlap，把最后一个主 decoder layer 的 forward prefetch 目标设置成了第一个 MTP layer：

```python
if mtp_idx == 0:
    layer_next.set_modules_to_forward_prefetch([mtp_layer])
```

这里的 `layer_next` 是主语言模型的最后一个 decoder layer。

所以执行链路是：

1. old-logprob forward 跑主语言模型。
2. 跑到最后一个主 decoder layer。
3. 这个 layer 的 FSDP pre-forward hook 根据 `set_modules_to_forward_prefetch()`，显式 prefetch `language_model.mtp_block.layers.0`。
4. PyTorch FSDP2 对目标 MTP param group 调 `unshard()`，产生 `param_group._all_gather_result`。
5. 但 old-logprob 没有 `mtp` loss context，后续不会进入 MTP block forward。
6. 因此这个 prefetched MTP layer 没有机会进入自己的 `pre_forward()` / `wait_for_unshard()` 去消费并清理 `_all_gather_result`。
7. `ONLY_CALC_MISMATCH_RATIO=1` 随后提前 return，也不会进入正常 train_step/backward 路径。
8. 切回 rollout 前 model offload 只移动参数，不知道要清这个 FSDP prefetch 临时状态，于是残留留在 GPU。

这也解释了为什么“标准 FSDP forward 后应该释放”这个直觉本身没错。这里不是一个已经完整 forward 过的 FSDP module 没释放，而是一个被上游 module explicit prefetch 了、但目标 module 本次根本没 forward 的 param group。

## 7. 为什么 root post-forward 不清它

PyTorch FSDP2 root `_post_forward()` 确实会清一种 deferred all-gather：

```python
if self._state_ctx.iter_forward_root is self:
    if all_gather_state := self._comm_ctx.all_gather_state:
        ...
        self._comm_ctx.all_gather_state = None
```

这清的是 `comm_ctx.all_gather_state`，也就是某个 module 已经执行 `wait_for_unshard()`、copy-out 已经发生，只是为了 overlap 把结果延迟挂在 comm context 上。

但 explicit forward prefetch 的代码是：

```python
target_fsdp_param_group.unshard(async_op)
```

它把结果存在目标 param group 的：

```text
target_fsdp_param_group._all_gather_result
```

只有目标 module 后续真的 forward，进入 `pre_forward()` 并调用 `wait_for_unshard()`，这个 `_all_gather_result` 才会被消费并清掉。

root post-forward 不能无条件清它，否则正常 explicit prefetch 的收益和正确性都会被破坏。它不知道目标 module 是否马上要运行。

本问题的特殊性是：XTuner 配置了最后主 layer -> MTP layer 的 explicit prefetch，但 old-logprob 这条路径不跑 MTP layer。

## 8. 最终修复代码

修复放在 `TrainingWorker.offload_model()`。在 model 参数移到 CPU 后，显式释放 FSDP2 仍持有的 all-gather 临时状态，再 `empty_cache()`。

当前保留两类清理：

1. `comm_ctx.all_gather_state`
  - 覆盖已经 copy-out、为了 overlap 延迟释放的 all-gather result。
2. `FSDPParamGroup._all_gather_result`
  - 覆盖 explicit prefetch 发起了 all-gather，但目标 module 没有 forward 消费的 result。
  - 这是本次 `ONLY_CALC_MISMATCH_RATIO=1` + MTP prefetch 真正命中的路径。

核心代码位置：

```text
xtuner/v1/rl/trainer/worker.py
```

`offload_model()` 的顺序：

```python
self._maybe_log_deferred_fsdp_all_gathers("offload_model/before_model_to_cpu")
self._engine.put_model_to_device("cpu")
self._maybe_log_deferred_fsdp_all_gathers("offload_model/after_model_to_cpu_before_deferred_release")
self._release_deferred_fsdp_all_gathers("offload_model")
DEVICE_MODULE.empty_cache()
self._maybe_log_deferred_fsdp_all_gathers("offload_model/after_deferred_release")
```

`_release_deferred_fsdp_all_gathers()` 里对 `param_group._all_gather_result` 的关键逻辑：

```python
if (all_gather_result := getattr(param_group, "_all_gather_result", None)) is not None:
    record_debug_item(
        f"{module.__class__.__name__}._fsdp_param_group._all_gather_result",
        param_group,
    )
    wait_all_gather_result(all_gather_result)
    param_group._all_gather_result = None
    released_param_groups += 1
```

清理前等待 `all_gather_event` / `all_gather_work`，避免极端情况下释放仍在飞的 all-gather result。

## 9. 保留的 debug 代码怎么用

为了复现完整证据链，保留了 gated debug。

建议运行：

```bash
export ONLY_CALC_MISMATCH_RATIO=1
export XTUNER_DEBUG_FSDP_DEFERRED=1
export XTUNER_DEBUG_MEMORY_STAGES=1
export XTUNER_DEBUG_OFFLOAD_MEMORY=1
```

如果要看 allocator snapshot 和 Python owner，再额外打开：

```bash
export XTUNER_DEBUG_OFFLOAD_MEMORY_SNAPSHOT=1
export XTUNER_DEBUG_ACTIVE_BLOCK_OWNERS=1
```

最关键的判断点是：

```text
rollout_N/before_compute_actor_logprobs
  -> deferred FSDP all-gather states: none

rollout_N/after_compute_actor_logprobs
  -> count=1
  -> module=language_model.mtp_block.layers.0
  -> holder=FSDPParamGroup
  -> result=AllGatherResult

rollout_N/before_only_calc_mismatch_return
  -> 仍然 count=1

update_weights/begin
  -> 仍然 count=1，说明不是 update_weights 新产生

offload_model
  -> released deferred FSDP all-gathers: comm_states=0, param_groups=1

offload_model/after_deferred_release
  -> deferred FSDP all-gather states: none
```

这套日志能够回答三个问题：

- 残留从哪里第一次出现：`after_compute_actor_logprobs`。
- 残留挂在哪个对象上：`language_model.mtp_block.layers.0` 的 `FSDPParamGroup._all_gather_result`。
- 释放是否命中正确对象：`comm_states=0, param_groups=1`，释放后 deferred state 为空。

## 10. 后续可选的根因级优化

当前修复是在 offload 边界清理 pending FSDP all-gather，适合 colocate train/rollout 切换场景，风险较小。

如果要进一步从源头减少这个 pending prefetch，可以考虑：

- 在 old-logprob / forward-only 路径中临时禁用最后主 decoder layer 对 MTP layer 的 forward prefetch。
- 或者只在当前 forward 确认会执行 MTP loss 时，才设置主 layer -> MTP layer 的 prefetch。

但这类改动会影响 FSDP prefetch 策略和正常训练 overlap，需要单独验证正常训练吞吐和 MTP loss 路径。当前 offload cleanup 是更局部的修复。