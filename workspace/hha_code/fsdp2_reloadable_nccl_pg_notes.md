# FSDP2 训练 NCCL 组销毁与重建复盘

本文记录这次在 XTuner colocate RL 里尝试释放训练侧 NCCL 显存的过程、踩到的问题，以及最终采用的方案。

## 背景

在 colocate 训练/rollout 场景里，训练完成并同步权重到 rollout worker 后，训练侧模型、优化器、梯度等大块状态已经可以 offload。此时如果仍然保留 FSDP2/MoE/DeviceMesh 相关 NCCL process group，每张卡还会有几 GB 级别的残留显存。

目标是：

1. 训练一步结束。
2. 完成权重同步到 rollout。
3. offload train model。
4. 销毁训练侧不再需要的 NCCL 子组。
5. 继续 rollout。
6. 下一轮训练前恢复这些 NCCL 子组。

当前实现默认不销毁 default/world process group，只销毁可重放的非 default NCCL 子组。

相关代码：

- `xtuner/v1/utils/reloadable_process_group.py`
- `xtuner/v1/rl/trainer/worker.py`
- `xtuner/v1/rl/trainer/controller.py`
- `xtuner/v1/train/rl_trainer.py`

## 一开始的直觉和问题

最初的直觉是：既然训练模型、优化器、梯度都 offload 了，剩下最大的显存残留应该来自 NCCL communicator；那直接 `dist.destroy_process_group(group)`，下一次训练前再 `new_group`，应该就能解决。

这个直觉在 Megatron/slime 风格的通信组管理里相对容易成立，因为通信组通常由框架自己集中持有，销毁和重建可以在框架自己的 registry 里闭环。

但 FSDP2 场景里问题更复杂：PyTorch/FSDP2/DeviceMesh/functional collectives 都会缓存 process group 的对象、名字或 tag。直接销毁底层 group 后，只在我们自己的代码里重新创建一个新 group，并不会自动更新这些缓存。

## 试错中遇到的典型错误

### 1. stale ProcessGroup 不在 world map 里

报错类似：

```text
ValueError: Process group <...> is not initialized in the world group map.
Please initialize the group first.
```

根因是 FSDP2 hook 里还拿着旧的 process group 对象。我们销毁了旧对象，又新建了别的对象，但 FSDP2 缓存没有换过去。

### 2. communicator aborted / remote process exited

报错类似：

```text
torch.distributed.DistBackendError: NCCL communicator was aborted
ncclRemoteError: remote process exited or there was a network error
```

这类错误常见于多 rank 重建顺序不一致，或者某些 rank 没有参与同一批 `new_group` 调用。NCCL subgroup 创建要求所有相关 rank 的调用顺序一致，否则会连错端口或卡住。

### 3. all_gather 输出尺寸不匹配

报错类似：

```text
ValueError: output tensor size must be equal to world_size times input tensor size
```

这说明恢复后的 group 语义已经不等价，常见原因是 ranks/world_size 顺序不对，或者某些缓存仍指向旧 group。

### 4. DeviceMesh 读取不到 group name

报错类似：

```text
RuntimeError: ProcessGroup name not set
```

DeviceMesh 会读取 process group 的 `group_name`。如果 wrapper 没有正确暴露稳定的 `group_name`，DeviceMesh 初始化或后续访问会失败。

### 5. all_gather_object 走到 CPU backend 问题

报错类似：

```text
RuntimeError: No backend type associated with device type cpu
```

`all_gather_object` 内部会看 process group 的私有 `_device_types`。如果 wrapper 没代理这个属性，PyTorch 会错误判断后端能力。

### 6. functional collectives 通过名字/tag 找不到 group

报错类似：

```text
RuntimeError: Could not resolve the process group registered under the name 14
```

MoE dispatch 里会用 `torch.distributed._functional_collectives` / autograd functional collectives。这里不是只通过 Python 对象身份找 group，还会通过 c10d 注册的 group name/tag 解析。新建出来的 inner process group 会有新的 name/tag，如果不把它恢复到原来的稳定身份，就会找不到。

## 最终确认的根因

FSDP2 下不能把 process group 当成一个简单的、可随意替换的 Python 对象。

关键缓存路径包括：

1. FSDP2 hook / param group 缓存 process group 对象。
2. DeviceMesh 缓存 process group 对象和 group name。
3. `torch.distributed.distributed_c10d._world` 维护 `pg_map`、`pg_names`、`pg_group_ranks`、`pg_backend_config`、`pg_to_tag`、`tags_to_pg` 等私有 registry。
4. functional collectives 会通过 group name/tag 解析 process group。
5. 非成员 rank 的 `new_group` 调用也必须按原始顺序重放，否则下一次 NCCL subgroup 创建顺序会错。

因此，正确方向不是让 FSDP2/DeviceMesh 缓存换成新的 group 对象，而是让它们一直缓存同一个稳定 wrapper。销毁和重建只发生在 wrapper 内部的真实 NCCL process group 上。

## 最终方案

### 1. 在训练 worker 初始化早期 monkey patch

在 `TrainingWorker.__init__` 里，创建 DeviceMesh/FSDP2 之前调用：

```python
monkey_patch_reloadable_process_groups()
```

patch 的对象包括：

- `torch.distributed.new_group`
- `torch.distributed.distributed_c10d.new_group`
- `torch.distributed.distributed_c10d.split_group`
- `torch.distributed.device_mesh.new_group`
- `torch.distributed.device_mesh.split_group`
- `torch.distributed.distributed_c10d._resolve_process_group`
- `torch.distributed.device_mesh._resolve_process_group`

这样 DeviceMesh 和 FSDP2 内部创建的 NCCL 子组都会被捕获。

### 2. 用稳定 wrapper 包住非 default NCCL 子组

`ReloadableProcessGroup` 继承 `torch.distributed.ProcessGroup`，对外是一个稳定 process group 对象。

它内部持有：

```python
self.group: dist.ProcessGroup | None
```

销毁时只销毁 `self.group`，wrapper 对象本身不变。FSDP2/DeviceMesh 缓存的仍然是 wrapper。

wrapper 需要代理：

- collectives：`allreduce`、`allgather`、`_allgather_base`、`alltoall_base` 等。
- 基础属性/方法：`rank`、`size`、`name`、`get_group_store`。
- 私有后端属性：`_get_backend`、`_device_types`。
- 稳定身份：`group_name`、`group_desc`。

### 3. 记录所有可重放 subgroup 创建 spec

每次 `new_group` / `split_group` 调用都会记录：

- kind：`new_group` 或 `split_group`
- 原始 args/kwargs
- 对应 wrapper

即使当前 rank 是 `NON_GROUP_MEMBER`，只要这是一个 NCCL/cuda 相关 subgroup 创建调用，也要记录。原因是下一次 reload 时，各 rank 必须按完全一致的创建顺序重放。

### 4. destroy 阶段只销毁 wrapper 内部的 inner group

训练结束、权重同步完成、train model offload 后执行：

```python
destroy_reloadable_process_groups()
```

行为：

- 逆序销毁所有 reloadable wrapper 内的 inner NCCL group。
- 将 `wrapper.group = None`。
- wrapper 本身继续活着，FSDP2/DeviceMesh 缓存不会失效。
- 不销毁 default/world process group。

### 5. reload 阶段按原始顺序重放 subgroup 创建

下一轮训练 onload 前执行：

```python
reload_process_groups()
```

行为：

1. 按记录的 spec 顺序调用原始 `c10d.new_group` / `c10d.split_group`。
2. 对每个 wrapper 拿到新的 inner process group。
3. 将新 inner 的 group name 恢复成 wrapper 原来的 stable name。
4. 将 c10d private registry 中的 tag/name 映射恢复到稳定身份。
5. `wrapper.group = new_inner`。

其中恢复 name/tag 是解决 functional collectives 的关键，否则 MoE all-to-all autograd 路径会找不到 group。

### 6. default/world process group 暂不销毁

当前方案明确不销毁 default/world group。

原因：

- default group 被 torch distributed 控制面、Ray worker 内部逻辑和其它通用路径使用。
- 从当前层面只重放 FSDP2/DeviceMesh subgroup，不足以安全重建 default group。
- 直接销毁 default group 可能让后续控制面通信不可恢复。

所以目前释放的是最有价值、也最可控的非 default NCCL 子组。剩余的 2GB 左右基线大概率来自 default/world group、CUDA context/runtime、torch distributed 常驻状态和进程内 CUDA 库状态。

## XTuner 生命周期接入点

### 初始化

`TrainingWorker.__init__` 早期 patch，必须发生在 `init_device_mesh` 和 FSDP2 wrapping 前。

### 销毁时机

`RLColocateTrainer._sync_weights_and_save` 中：

1. rollout worker restart/onload weights。
2. bind train/rollout。
3. `train_controller.update_weights()`。
4. `train_controller.offload(target="model")`。
5. 如果开启 `XTUNER_DESTROY_TRAIN_NCCL_AFTER_SYNC=1`，销毁 train NCCL 子组。

这个时机保证第二次 rollout 可以继续跑，因为权重同步已经完成。

### 恢复时机

`BaseRLTrainer._train_one_batch` 中，在下一次 train onload 前：

1. rollout offload。
2. 如果上次销毁过 train NCCL，先 reload。
3. `train_controller.onload(target="all")`。
4. 继续 FSDP2 forward/backward/update。

## 当前配置

脚本中保留：

```bash
export XTUNER_DESTROY_TRAIN_NCCL_AFTER_SYNC=1
```

已移除：

- `XTUNER_DESTROY_TRAIN_NCCL_INCLUDE_DEFAULT`
- `XTUNER_STOP_AFTER_ROLLOUT_STEP`
- `XTUNER_DESTROY_TRAIN_NCCL_AFTER_SYNC_STEP`

## 日志策略

最初调试时每个 worker 都打印完整 JSON，controller 和 trainer 又重复打印，日志非常大，也容易被 Ray 的 repeated log 聚合截断。

现在改成：

- worker 返回完整结构化结果，但不打印。
- trainer 不打印完整结果。
- controller 只打印一行汇总。

汇总内容包括：

- rank 数。
- 每个 rank destroy/reload 的数量。
- reloadable wrapper 存活/销毁总数。
- replay spec 数。
- destroy/reload 后 used 显存范围。
- error 数。
- 未被 wrapper 捕获的 NCCL group 数。

## 验证结果

从 `mem_070806/actor_memory.json` 看，销毁后训练 worker 显存可以稳定降到约 `2.1G`，下一轮训练前 reload 后可以继续训练。

相较之前直接销毁和粗暴重建的尝试，当前方案解决了：

- FSDP2 stale process group。
- DeviceMesh group name 缓存。
- all_gather_object CPU backend 判断。
- functional collectives group name/tag 解析。
- 多 rank subgroup 创建顺序不一致。

## 仍需注意的风险

1. 该方案依赖 PyTorch c10d 的私有 registry，例如 `_world.pg_names`、`pg_to_tag`、`tags_to_pg`。PyTorch 升级后需要重新验证。
2. 当前只处理非 default NCCL 子组。要进一步降到更低显存，需要研究 default/world group 的安全重建，这涉及分布式控制面，不应混在当前方案里做。
3. wrapper 目前覆盖了当前训练路径用到的 collectives。如果未来模型引入新的 ProcessGroup 方法，可能需要继续补代理方法。
4. `new_group` / `split_group` replay 依赖所有 rank 按原始顺序调用。后续如果有条件化建组逻辑，需要特别检查各 rank 是否一致。

## 结论

这次问题的核心不是“能不能销毁 NCCL communicator”，而是“销毁后如何让 FSDP2、DeviceMesh、c10d registry 和 functional collectives 看到同一个稳定的 process group 身份”。

最终方案是：

- 让 FSDP2/DeviceMesh 永远持有稳定 wrapper。
- 销毁时只释放 wrapper 内部的 NCCL inner group。
- 恢复时按原始建组顺序重放，并把新 inner 的 name/tag 恢复到原稳定身份。
- default/world group 暂不销毁。

这和 slime 的思路一致：不要让训练框架的高层缓存直接感知 communicator 被拆掉，只在可控边界内释放和恢复底层通信资源。
