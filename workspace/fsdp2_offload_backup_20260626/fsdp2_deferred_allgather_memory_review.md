# ONLY_CALC_MISMATCH_RATIO 下 Offload 后残留显存问题复盘

日期：2026-06-26

相关运行：

- 脚本：`/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/new_test_sh/reasoning_rl_test.sh`
- 配置：`examples/v1/config/reasoning_rl_qwen3p5vl_mtp_ep.py`
- 主要日志：
  - `work_dirs_rl/qwen3vl_8b_grpo_mixdata3-lmdeploy/20260626094241/logs/rank_0.log`
  - `work_dirs_rl/qwen3vl_8b_grpo_mixdata3-lmdeploy/20260626100721/logs/rank_0.log`

## 1. 问题背景

这次排查的触发点不是普通训练路径显存异常，而是打开：

```bash
export ONLY_CALC_MISMATCH_RATIO=1
```

之后出现了不符合预期的 offload 后显存残留。

这个 flag 的用途是只计算 old logprob / mismatch / rollout importance sampling 指标，然后提前 return。它本质上是一个诊断模式：希望在不进入后续 `train_step()`、backward、optimizer step 的情况下，观测 rollout logprob 和 train-side logprob 的差异。

因此最初的预期是：

- 仍然会走 rollout 生成、训练侧 old-logprob forward、mismatch 指标计算。
- 不会真正做训练更新。
- 在 colocate train/rollout 切换时，`offload_model()` 后显存状态不应该比正常路径多出一块长期残留的大 tensor。

实际现象和这个预期不一致：打开 `ONLY_CALC_MISMATCH_RATIO=1` 后，训练 step 结束切回 rollout 时，`Offloaded model to CPU` 后仍稳定残留约 `1.6G` 显存。

## 2. 现象

实验配置里有几个容易误导判断的点：

- rollout 设置了 `skip_load_weights=True`，启动后会立刻从 train worker 同步一次权重。
- `use_kl_loss=False`，没有 ref model。
- 用户确认没有 DeepEP。
- 使用 colocate 模式，训练和 rollout 在同一组 GPU 资源上切换。

一开始看到的问题是：在 `ONLY_CALC_MISMATCH_RATIO=1` 运行下，`Offloaded model to CPU` 后，GPU 上仍然残留一块约 `1.6G` 到 `2G` 的显存。以 35B 这次日志为例：

```text
[XTuner][RANK 0][2026-06-26 10:12:57][INFO][TrainingWorker]
Offloaded model to CPU. Current allocate 1644.13671875 MB, reserved: 1670.0 MB
```

但这个残留并不是一开始就存在。初始化阶段的 skip-load 权重同步后，offload 是干净的：

```text
[XTuner][RANK 0][2026-06-26 10:10:58][INFO][TrainingWorker]
handling same hf param: ['lm_head.weight'] separately

[XTuner][RANK 0][2026-06-26 10:11:04][INFO][TrainingWorker]
Offloaded model to CPU. Current allocate 0.0009765625 MB, reserved: 4.0 MB

[XTuner][RANK 0][2026-06-26 10:11:11][INFO][RLTrainer]
Rollout workers updated weights from train workers.
```

训练一次之后，再次同步权重并 offload，就开始稳定出现残留：

```text
[XTuner][RANK 0][2026-06-26 10:12:52][INFO][TrainingWorker]
handling same hf param: ['lm_head.weight'] separately

[XTuner][RANK 0][2026-06-26 10:12:55][INFO][RLTrainer]
Rollout workers update weights successfully in colocate mode

[XTuner][RANK 0][2026-06-26 10:12:57][INFO][TrainingWorker]
Offloaded model to CPU. Current allocate 1644.13671875 MB, reserved: 1670.0 MB
```

后续 step 中这个值持续存在，并伴随少量增长：

```text
10:13:18 Offloaded model to CPU. Current allocate 1644.29296875 MB
10:15:03 Offloaded model to CPU. Current allocate 1644.37109375 MB
10:16:03 Offloaded model to CPU. Current allocate 1644.52734375 MB
...
```

因此首先能确定的是：

- 这不是“模型一启动就天然要留 1.6G”。
- 也不是简单由 skip-load 初始化权重同步本身导致。
- 异常更像是 `ONLY_CALC_MISMATCH_RATIO=1` 下，训练侧完成 old-logprob/mismatch 后，切回 rollout/offload 时留下了某个训练侧临时状态。

## 3. 第一轮定位：offload 后还有真实 CUDA tensor

为了区分 allocator cache、碎片和真实 tensor 引用，在 `offload_model()` 后加入过一段 gated debug，扫描 Python 可见的 live CUDA tensor。

日志显示，`memory_allocated()` 中的主要部分确实是一个真实活着的 tensor，而不是 reserved cache：

```text
[XTuner][RANK 0][2026-06-26 10:12:58][INFO][TrainingWorker]
[offload_model] live CUDA tensors visible to Python: total=1611.13 MB unique_storages=8

01: 1611.05 MB shape=(844655104,) dtype=torch.bfloat16 device=cuda:0 type=Tensor requires_grad=False
02: 0.08 MB shape=(40, 256) dtype=torch.int64 device=cuda:0 type=Tensor requires_grad=False
...
```

这个结果很关键：

- `1644 MB` 的残留里，约 `1611 MB` 是一个 Python 可见 CUDA tensor。
- tensor 是 1D bf16 flat buffer：`844655104 * 2 bytes = 1689310208 bytes = 1611.05 MiB`。
- 所以问题不是 CUDA allocator 的 non-releasable memory，也不是 `empty_cache()` 不生效。

## 4. 被排除的方向

### 4.1 不是 ref model / DeepEP

配置里 `use_kl_loss=False`，没有 ref model；用户也确认没有 DeepEP。因此这两类显存来源可以排除。

### 4.2 不是 ONLY 分支里的 loss_ctx / old_logprobs 引用

曾经尝试在 `ONLY_CALC_MISMATCH_RATIO=1` 提前 return 前显式清理：

- `old_logprobs_list`
- `rollout_logprobs_list`
- `shifted_labels_list`
- `loss_ctx_list`
- `seq_ctx_list`
- `mtp_loss_ctx_list`

并执行 `gc.collect()` / `synchronize()` / `empty_cache()`。重新运行后，`offload_model` 后仍然是约 `1644 MB`，因此普通 Python loss/context 引用不是主因。

### 4.3 `lm_head` / tail module 的 reshard 方向是合理的，但不是最终残留

MoE 模型里确实有一些 module 显式设置 `reshard_after_forward=False`：

- 最后一层 decoder layer
- `lm_head`
- 最后一个 MTP layer

这会导致 forward 后 full param 暂时留在 GPU。这个方向本身是合理的。

但模型 offload 会把参数本身搬到 CPU；如果残留只是这些权重 full param，`put_model_to_device("cpu")` 后应该消失。实际残留出现在 model 已经 offload 之后，所以最终残留不只是 module 参数本身。

### 4.4 IPC flattened bucket 是中间嫌疑，但 referrer 证明不是最终持有者

因为权重同步走 lmdeploy `FlattenedTensorBucket` / CUDA IPC，且残留 tensor 是一个大 flat bf16 tensor，所以一度怀疑是 IPC send buffer。

但后续 referrer debug 显示，最大 tensor 的直接持有者不是 `FlattenedTensorBucket`，而是 PyTorch FSDP2 的 `AllGatherResult`：

```text
[XTuner][RANK 0][2026-06-26 10:12:58][INFO][TrainingWorker]
[offload_model] largest CUDA tensor referrers:

01: type=tuple indices=[6]
02: type=tuple indices=[0]
03: type=AllGatherResult indices=[0]
```

这一步把问题从“lmdeploy IPC buffer 生命周期”推进到了 “FSDP2 all-gather 临时结果生命周期”。

## 5. 关键证据链

### 5.1 残留总是出现在权重同步之后

反复出现的日志模式是：

```text
handling same hf param: ['lm_head.weight'] separately
Rollout workers update weights successfully in colocate mode
Offloaded model to CPU. Current allocate 1644.x MB
```

`handling same hf param: ['lm_head.weight'] separately` 来自 `BaseModel._get_same_hf_param()` 中对 `lm_head.weight` 的特殊路径：

```python
if (
    self.fsdp_config is not None
    and self.fsdp_config.fp32_lm_head
    and load_spec.hf_keys[0] == "lm_head.weight"
):
    log_rank0.info(f"handling same hf param: {load_spec.hf_keys} separately")
    lm_head_tensor_list = self._fsdp_foreach_allgather([local_tensor], [load_spec])
    ...
```

这说明权重同步时确实会触发 FSDP all-gather，尤其是 `lm_head.weight` 的特殊导出路径。

### 5.2 最大 live tensor 的 referrer 是 `AllGatherResult`

`AllGatherResult` 是 PyTorch FSDP2 内部对象。其第 0 个字段是：

```python
class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    ...
```

日志里的：

```text
type=AllGatherResult indices=[0]
```

对应的就是 `AllGatherResult.all_gather_output`。也就是说，offload 后残留的 `1611 MB` tensor 是 FSDP2 all-gather output。

### 5.3 FSDP2 会有意延迟释放最后一个 all-gather result

PyTorch FSDP2 的 `wait_for_unshard()` 里有这样的逻辑：

```python
if (
    not async_op
    and self._training_state == TrainingState.FORWARD
    and world_size > 1
):
    # Defer free to allow for overlap of this copy-out with next
    # all-gather collective
    self.comm_ctx.all_gather_state = AllGatherState(
        self._all_gather_result, all_gather_copy_out_event
    )
else:
    self._wait_all_gather_streams_on_event(all_gather_copy_out_event)

self._all_gather_result = None
```

也就是说，在 forward 状态下，FSDP2 会把最后一次 all-gather result 挂到 `comm_ctx.all_gather_state` 上，用于和下一次 all-gather/copy-out overlap。

这个设计在连续训练 forward/backward 时是合理的：下一次 FSDP forward 会清理前一次 deferred result。

但 colocate RL 里，训练 worker 之后会切到 rollout，并且 model 被 offload。此时可能很长时间没有下一次 FSDP forward，`comm_ctx.all_gather_state` 里的最后一个 `AllGatherResult` 就一直活着。

### 5.4 为什么初始化同步没残留，训练后同步才残留

初始化 skip-load 同步后，日志显示：

```text
Offloaded model to CPU. Current allocate 0.0009765625 MB
live CUDA tensors visible to Python: total=0.00 MB
```

训练后同步则显示：

```text
Offloaded model to CPU. Current allocate 1644.13671875 MB
live CUDA tensors visible to Python: total=1611.13 MB
largest CUDA tensor referrers:
03: type=AllGatherResult indices=[0]
```

两次都调用了 `self.train_controller.update_weights()`，区别不是“第一次没有 update”。更准确的解释是：

- 初始化同步后没有留下 deferred `AllGatherResult`。
- 在 `ONLY_CALC_MISMATCH_RATIO=1` 下，训练侧会做 old-logprob/mismatch forward，但不会进入正常 `train_step()` / backward / optimizer 那条释放路径。
- 之后经过 rollout 资源切换、再次权重同步，FSDP2 的最后一个 deferred all-gather output 没有被下一次 FSDP forward 消费。
- 之后立刻 offload model，普通 model offload 不会清理 `comm_ctx.all_gather_state`，所以残留保留在 GPU。

## 6. 推导出的原因

基于上面的现象和排除过程，可以得到原因：

`ONLY_CALC_MISMATCH_RATIO=1` 下，offload 后残留的约 `1.6G` 显存，不是 ref model、DeepEP、loss ctx、old logprob，也不是单纯的 `lm_head` full param 本身。

它是 PyTorch FSDP2 为 overlap 延迟保存的 deferred all-gather output：

```text
FSDP comm_ctx.all_gather_state
  -> AllGatherResult
     -> all_gather_output: bf16 tensor, shape=(844655104,), about 1611 MiB
```

这块显存不属于 model parameter 的普通 device move 路径，所以 `model.to("cpu")` / `put_model_to_device("cpu")` 后不会自动释放。

## 7. 最终修复代码

修复点放在 `TrainingWorker.offload_model()` 中：

1. 先把 model 参数 offload 到 CPU。
2. 如果开启 `XTUNER_DEBUG_OFFLOAD_MEMORY=1`，打印 offload 后、释放 deferred FSDP buffer 前的 live tensor/referrer，用于复现证据链。
3. 显式释放 FSDP2 deferred all-gather state。
4. `empty_cache()`。
5. 如果开启 debug，再打印释放后的 live tensor，用于验证残留是否消失。

当前代码位于：

```text
xtuner/v1/rl/trainer/worker.py
```

核心代码如下：

```python
@ray_method
def offload_model(self):
    self._engine.put_model_to_device("cpu")
    if os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY", "0") == "1":
        self._log_offload_memory_debug("after_model_to_cpu_before_deferred_fsdp_release")
    self._release_deferred_fsdp_all_gathers("offload_model")
    DEVICE_MODULE.empty_cache()
    self.logger.info(
        f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
    )
    if os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY", "0") == "1":
        self._log_offload_memory_debug("after_deferred_fsdp_release")
```

```python
def _release_deferred_fsdp_all_gathers(self, log_tag: str) -> None:
    """Free FSDP2 deferred all-gather buffers before long offload gaps.

    FSDP2 keeps the last forward all-gather result in the shared comm
    context to overlap its free with the next all-gather copy-out. When we
    switch from train to rollout and offload the model, there may be no
    next FSDP forward soon, so release that deferred temporary explicitly.
    """

    try:
        from torch.distributed._composable_state import _get_module_state
    except Exception:
        return

    released_states = 0
    released_param_groups = 0
    debug = os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY", "0") == "1"
    debug_items: list[str] = []

    def get_all_gather_result(holder):
        result = getattr(holder, "all_gather_result", None)
        if result is not None:
            return result
        result = getattr(holder, "result", None)
        if result is not None:
            return result
        if isinstance(holder, tuple) and len(holder) > 0:
            return holder[0]
        return None

    def record_debug_item(source: str, holder) -> None:
        if not debug or holder is None:
            return
        all_gather_result = get_all_gather_result(holder)
        all_gather_output = getattr(all_gather_result, "all_gather_output", None)
        if not torch.is_tensor(all_gather_output):
            return
        debug_items.append(
            f"{source}: {all_gather_output.numel() * all_gather_output.element_size() / (1024**2):.2f} MB "
            f"shape={tuple(all_gather_output.shape)} dtype={all_gather_output.dtype} "
            f"device={all_gather_output.device}"
        )

    for module in self._engine.model.modules():
        try:
            state = _get_module_state(module)
        except Exception:
            continue
        if state is None:
            continue

        comm_ctx = getattr(state, "_comm_ctx", None)
        all_gather_state = getattr(comm_ctx, "all_gather_state", None)
        if all_gather_state is not None:
            record_debug_item(f"{module.__class__.__name__}.comm_ctx.all_gather_state", all_gather_state)
            event = getattr(all_gather_state, "event", None)
            if event is not None:
                try:
                    event.synchronize()
                except Exception:
                    DEVICE_MODULE.synchronize()
            comm_ctx.all_gather_state = None
            released_states += 1

        param_group = getattr(state, "_fsdp_param_group", None)
        if getattr(param_group, "_all_gather_result", None) is not None:
            record_debug_item(f"{module.__class__.__name__}._fsdp_param_group._all_gather_result", param_group)
            param_group._all_gather_result = None
            released_param_groups += 1

    if debug and (released_states or released_param_groups):
        detail = "\n".join(debug_items) if debug_items else "<no tensor details>"
        self.logger.info(
            f"[{log_tag}] released deferred FSDP all-gathers: "
            f"comm_states={released_states}, param_groups={released_param_groups}\n{detail}"
        )
```

调试辅助代码保留为 gated debug：

```python
def _log_offload_memory_debug(self, tag: str) -> None:
    self._log_live_cuda_tensors(tag)
```

它会打印：

- 当前 Python 可见 CUDA tensor top
- 最大 tensor 的 direct referrer
- 释放前后对比

## 8. 验证方式

运行时设置：

```bash
export XTUNER_DEBUG_OFFLOAD_MEMORY=1
```

预期日志形态：

```text
[after_model_to_cpu_before_deferred_fsdp_release] live CUDA tensors visible to Python: total=1611.xx MB ...
01: 1611.05 MB shape=(844655104,) dtype=torch.bfloat16 ...

[after_model_to_cpu_before_deferred_fsdp_release] largest CUDA tensor referrers:
... type=AllGatherResult indices=[0]

[offload_model] released deferred FSDP all-gathers:
comm_states=..., param_groups=...
... all_gather_output ... 1611.05 MB shape=(844655104,) dtype=torch.bfloat16 ...

Offloaded model to CPU. Current allocate <接近 0 或显著下降> MB

[after_deferred_fsdp_release] live CUDA tensors visible to Python: total=<接近 0 或只剩小 tensor> MB
```

如果这个验证通过，就说明：

- 释放前的残留仍能复现为 `AllGatherResult.all_gather_output`。
- `_release_deferred_fsdp_all_gathers()` 命中 FSDP2 deferred state。
- 释放后 `memory_allocated()` 不再保留 `1.6G`。

## 9. 风险和注意点

这个修复使用了 PyTorch composable FSDP2 的内部状态：

```python
from torch.distributed._composable_state import _get_module_state
```

风险：

- PyTorch 内部字段名未来可能变化。
- 当前代码已做 `try/except`，如果 API 不存在，会直接跳过，不影响普通 offload。
- 真正释放前会等待 `all_gather_state.event`，避免异步 copy-out 还没完成就清引用。

为什么在 `offload_model()` 中做是合理的：

- colocate 切到 rollout 前，本来就要长时间不使用 train model GPU 参数。
- 此时保留 FSDP2 overlap 临时 buffer 没有收益。
- 普通 model offload 不覆盖 `comm_ctx.all_gather_state`，所以需要显式清理。
