# GatedDeltaNet SP 速度优化复盘

本文复盘 PR [InternLM/xtuner#1558](https://github.com/InternLM/xtuner/pull/1558) 对 `GatedDeltaNet` linear attention 的 sequence parallel(SP) 路径优化。PR 标题是 `Optimizes the sequence parallel (SP) for GatedDeltaNet linear attention`，基于 `Support SP of Qwen3.5 (#1529)` 之后的实现，仅对 `xtuner/v1/module/attention/gated_deltanet.py` 的 SP 数据流做了重排。

PR 描述里的性能数据：

| 场景 | 吞吐 |
| --- | ---: |
| 32k | 10200 |
| old 64k sp2 | 9100 |
| new 64k sp2 | 9800 |

其中 `64k sp2` 从 9100 提升到 9800，约为 `7.7%`。

一句话总结：这次优化的关键不是单纯减少 all-to-all 调用次数，而是减少等价通信量。q/k/v 主体相关的 A2A 从 3 轮等价 qkv 通信量降到 1 轮，少了 `2/3` 的 qkv 主体通信量。

## 背景

Qwen3.5 的部分层使用 `GatedDeltaNet` 线性注意力。它的核心流程可以拆成：

1. `in_proj_qkv(hidden_states)` 得到 `mixed_qkv`。
2. 对 q/k/v 做 depthwise causal conv1d。
3. 计算 `beta` 和 `g`。
4. 调用 `chunk_gated_delta_rule(query, key, value, g, beta)`。
5. 用 `z` 做 gated RMSNorm，再 `out_proj`。

SP 下，输入序列已经按 sequence 维切到各个 rank：本地形状类似 `(batch=1, L/sp, hidden)`。但是 `chunk_gated_delta_rule` 需要在完整序列长度 `L` 上推进线性递推，所以必须把 sequence 维重新聚合起来，同时把 head/channel 维切到各个 rank 上，形成每个 rank 处理一部分 heads、但拥有完整 sequence 的布局。

这正是 Ulysses-style all-to-all 的用途：从“按 sequence 切分”转换到“按 head/channel 切分”。

## 原实现瓶颈

PR 之前的 SP 路径大致是：

```text
mixed_qkv: (1, L/sp, conv_dim)
  transpose
mixed_qkv: (1, conv_dim, L/sp)
  all_to_all conv_pre: scatter conv_dim, gather sequence
mixed_qkv: (1, conv_dim/sp, L)
  causal_conv1d_fn
mixed_qkv: (1, L, conv_dim/sp)
  all_to_all conv_post: scatter sequence, gather conv_dim
mixed_qkv: (1, L/sp, conv_dim)
  split q/k/v, reshape heads
q/k/v: (1, L/sp, heads, head_dim)
  all_to_all q/k/v: scatter heads, gather sequence
q/k/v: (1, L, heads/sp, head_dim)
  chunk_gated_delta_rule
out: (1, L, heads/sp, head_dim)
  all_to_all out: scatter sequence, gather heads
out: (1, L/sp, value_dim)
```

问题在于 q/k/v 的数据重排做了两轮互相抵消的通信：

- 第一轮 `conv_pre` 为了让 causal conv 在完整序列上做，先把 `mixed_qkv` 从 `(L/sp, conv_dim)` 变成 `(L, conv_dim/sp)`。
- `causal_conv1d_fn` 之后，`conv_post` 又把它还原回 `(L/sp, conv_dim)`。
- 后面为了进入 `chunk_gated_delta_rule`，q/k/v 再各自 all-to-all 一次，从 `(L/sp, heads)` 变回 `(L, heads/sp)`。

也就是说，在 SP 路径里，卷积后“还原到 sequence-sharded 布局”只是一个过渡状态。最终 attention 仍然要回到“完整 sequence + sharded heads”的布局。这一来一回增加了通信、transpose/contiguous 开销，也让代码里的 shape 变换更复杂。

## 优化核心

优化思路是：不要把 conv 之后的数据还原成 sequence-sharded 布局，而是让 conv 的输入/输出直接停留在 `chunk_gated_delta_rule` 需要的布局。

最终 SP 路径变成：

```text
mixed_qkv: (1, L/sp, conv_dim)
  split q/k/v
q: (1, L/sp, key_dim)
k: (1, L/sp, key_dim)
v: (1, L/sp, value_dim)
  transpose to channel-first
q/k/v: (1, dim, L/sp)
  all_to_all per q/k/v: scatter dim, gather sequence
q/k/v: (1, dim/sp, L)
  split conv weight by q/k/v, then by sp_rank
  causal_conv1d_fn per q/k/v
q/k/v: (1, dim/sp, L)
  reshape heads
q/k: (1, L, num_k_heads/sp, head_k_dim)
v:   (1, L, num_v_heads/sp, head_v_dim)
  chunk_gated_delta_rule
out: (1, L, num_v_heads/sp, head_v_dim)
  all_to_all out
out: (1, L/sp, value_dim)
```

和旧路径相比，直接消掉了：

- `mixed_qkv` 卷积后的 `conv_post all_to_all`。
- q/k/v 进入 `chunk_gated_delta_rule` 前的第二轮 all-to-all。

保留下来的通信是必要的：

- q/k/v 在 conv 前各自做一次 all-to-all：把 `L/sp` 聚成 `L`，同时把通道切成 `dim/sp`。
- `g`/`beta` 仍然需要 all-to-all：它们来自 `a`/`b` 投影，初始仍是 `(1, L/sp, num_v_heads)`，需要变成 `(1, L, num_v_heads/sp)`。
- `core_attn_out` 仍然需要 all-to-all：从 attention 内部的 head-sharded layout 回到后续 `out_proj` 需要的 sequence-sharded layout。

## 为什么这样是等价的

关键是 `conv1d` 在这里是 depthwise conv：

```python
self.conv1d = nn.Conv1d(
    in_channels=self.conv_dim,
    out_channels=self.conv_dim,
    bias=False,
    kernel_size=self.conv_kernel_size,
    groups=self.conv_dim,
    padding=self.conv_kernel_size - 1,
)
```

`groups=self.conv_dim` 表示每个通道独立卷积，不会在 q/k/v 通道之间混合。因此：

- 先把 `mixed_qkv` 整体做 all-to-all 再卷积；
- 或先拆成 q/k/v，各自 all-to-all，再用对应的 q/k/v 权重卷积；

在数学上是等价的。只要每个 rank 拿到的通道切片和权重切片一致，输出就是同一批通道的卷积结果。

PR 中对应的实现要点：

- 先 `torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)`。
- q/k/v 转成 `(1, dim, L/sp)`，分别调用 `_all_to_all_conv_pre_qk` / `_all_to_all_conv_pre_v`。
- `weight` 同样按 q/k/v split，再按 `sp_rank` chunk：

```python
query_weight, key_weight, value_weight = torch.split(
    weight,
    [self.key_dim, self.key_dim, self.value_dim],
    dim=0,
)
query_weight = query_weight.chunk(sp_size, dim=0)[sp_rank]
key_weight = key_weight.chunk(sp_size, dim=0)[sp_rank]
value_weight = value_weight.chunk(sp_size, dim=0)[sp_rank]
```

这样每个 rank 上的 q/k/v tensor 和 conv weight 都是同一个 head/channel shard。

## 代码演进

PR 中有几次有价值的迭代：

### 1. 初版：前移 q/k/v all-to-all

提交 `202046d2 optimize linear attention of sp` 做了主要重排：

- 把原来的 `forward` 拆成 `forward_for_sp` 和非 SP `forward`。
- SP 分支里先拆 q/k/v，再 all-to-all 到完整 sequence。
- 删除卷积后的 `conv_post`。
- 删除 `chunk_gated_delta_rule` 前的 q/k/v all-to-all。

这一步已经把性能优化主体完成了。

### 2. 修 `torch.compile` 缓存问题

提交 `6e163c60 fix compile` 把 q/k 和 v 的 all-to-all wrapper 拆开：

```python
def _all_to_all_conv_pre_qk(...): ...
def _all_to_all_conv_pre_v(...): ...
```

文件顶部已有注释：不同 call site 使用不同 Python 函数对象，Dynamo 会分别缓存。q 和 k 形状一致，可以共用 `_all_to_all_conv_pre_qk`；v 的 shape 可能不同，单独用 `_all_to_all_conv_pre_v`，避免 compile cache 因动态形状/签名混淆。

`ulysses_all_to_all` 内部也有 compile 相关约束：它先 `input.contiguous()`，再用 `movedim(scatter_dim, 0)`，注释里说明 `torch.compile` 下不能用普通 transpose，否则可能触发 contiguous 相关报错。

这里有一个文档层面的注意点：代码注释写的是“每个 call site 使用独立函数对象”，但最终 q/k 实际共享 `_all_to_all_conv_pre_qk`，`g/beta` 也共享 `_all_to_all_gb`。更准确的理解是：shape/trace 模式不同的 call site 要拆开，shape 完全一致的 call site 可以共享。

### 3. 简化中间 shape

提交 `5eef437b refine code` 简化了 q/k/v 的中间布局。第一版曾经把 q/k/v reshape 到 `(1, heads, L/sp, head_dim)` 再 all-to-all；后续改成直接在通道维 `(1, dim, L/sp)` 上 all-to-all。

这个改动让代码更接近真实需要：卷积的通道维本来就是 `key_dim/value_dim`，没有必要先拆 head 再 flatten 回通道。

### 4. 修 correctness 问题

review 指出了一个关键 bug：`value` reshape 曾经误用了 `self.head_k_dim`，当 `head_k_dim != head_v_dim` 时会产生错误 shape 或错误计算。

最终修成：

```python
value = value.transpose(1, 2).reshape(
    batch_size, seq_len * sp_size, -1, self.head_v_dim
)
```

这类优化很容易因为 q/k/v 维度相近而写错，文档化时要特别保留这个检查点。

### 5. 删除实验性代码和无关改动

PR 中间曾误提交过 `gated_deltanet copy.py` 和 `gated_deltanet copy 2.py`，其中包含一个更激进的实验：用独立 CUDA stream/event 将 q/k/v all-to-all 与 q/k/v conv 做流水重叠。

实验思路是：

- 在 `comm_stream` 上顺序发起 q/k/v all-to-all。
- 每个 all-to-all 后 record event。
- main stream 等 q event 后做 q conv，此时 k/v all-to-all 继续跑。
- 等 k event 做 k conv，再等 v event 做 v conv。

这个方向理论上可以继续压缩通信等待，但最终 PR 删除了这些 copy 文件，合入的是更简单的通信重排版本。原因从最终代码看很明确：先把稳定、可编译、可 review 的结构性优化合入，避免把 stream/event 生命周期、compile、autograd collective 的风险一起带进主线。

## 最终 SP 路径细节

以最终 PR 版本为准，SP 分支做了这些事：

1. 入口判断：`forward` 中如果 `seq_ctx.sequence_parallel_mesh.size() > 1`，转到 `forward_for_sp`。
2. 投影：`mixed_qkv`、`z`、`b`、`a` 仍然在本地 `L/sp` 上计算。
3. 拆 q/k/v：因为 depthwise conv 不混通道，先 split 不改变语义。
4. q/k/v all-to-all：把每个张量从 sequence-sharded 转到 channel/head-sharded。
5. 权重切片：`conv1d.weight` 先拆 q/k/v，再按 `sp_rank` 拿本 rank 通道权重。
6. q/k/v 分别卷积：每个 rank 对自己负责的通道、完整 sequence 做 causal conv。
7. reshape 成 head 布局：q/k 使用 `head_k_dim`，v 使用 `head_v_dim`。
8. q/k head 对齐：如果 `num_v_heads // num_k_heads > 1`，对 q/k 做 `repeat_interleave`。
9. `g`/`beta` all-to-all：和 v heads 对齐。
10. `chunk_gated_delta_rule` 在完整 sequence + sharded heads 上执行。
11. 输出 all-to-all：把结果转回 sequence-sharded layout。
12. norm + out_proj：继续后续 dense 投影。

## 性能收益来源

这次 PR 不是减少 FLOPs，而是减少 SP 路径上的通信和 layout 变换：

- 删除一次大张量 `mixed_qkv` 的 post-conv all-to-all。
- 删除 q/k/v 三个张量进入 attention 前的 all-to-all。
- 避免在两种分布式布局之间来回折返。
- 非 SP 路径从 SP 分支中拆出来，减少普通路径上的条件判断和无关注释/变形逻辑。

虽然 q/k/v 在 conv 前仍然需要分别 all-to-all，但旧路径的 q/k/v all-to-all 本来就存在；优化后它们承担了原来 `conv_pre` 和 attention 前 q/k/v all-to-all 两者的布局转换目的。

按等价通信量看，可以把 q/k/v 主体通信近似写成：

```text
优化前: 3 * qkv_volume
优化后: 1 * qkv_volume
减少:   2 * qkv_volume = 2/3 的 qkv 主体通信量
```

这里的 `3 * qkv_volume` 对应旧路径里的 `conv_pre`、`conv_post`、attention 前 q/k/v all-to-all 三轮主体布局转换；优化后只保留一轮 q/k/v all-to-all，并直接让卷积输出停留在 attention 需要的布局。

## 正确性和风险点

需要重点关注这些约束：

- `conv1d` 必须保持 depthwise：`groups=self.conv_dim` 是 split q/k/v 后仍然等价的前提。
- q/k/v 的 channel shard 必须和 weight shard 一致。
- `value` reshape 必须使用 `head_v_dim`，不能复用 `head_k_dim`。
- `g` 和 `beta` 仍然要 all-to-all，否则它们仍是 sequence-sharded，和 `chunk_gated_delta_rule` 的 q/k/v layout 不匹配。
- `sequence_parallel_mesh` 应在第一次 all-to-all 之前就确认非空。PR 末版是在 q/k/v all-to-all 之后才 assert，虽然正常入口由 `forward` 保证 SP 分支只在 mesh 存在时进入，但更清晰的写法是进入 `forward_for_sp` 后立即 assert 或绑定 `mesh`。
- `bias` 当前是 `bias=False`，但如果未来打开 bias，现有逻辑不能直接复用一个 chunk 后的 bias 给 q/k/v 三个 conv；应该像 weight 一样先 split q/k/v，再分别按 SP rank chunk。
- all-to-all wrapper 拆分是为了适配 Dynamo cache，不要随意合并成一个通用 wrapper。

## 推荐验证方法

复现或继续改这段逻辑时，建议至少做三类验证：

1. 数值等价：同一随机输入、同一权重下，对比 non-SP、旧 SP、新 SP 的输出。至少覆盖 `head_k_dim == head_v_dim` 和 `head_k_dim != head_v_dim`。
2. compile 路径：启用当前模型配置里的 `torch.compile`，确保 q/k/v all-to-all wrapper 不引入 graph/cache 问题。
3. 性能 profile：在 `64k sp2` 这类长序列场景下统计每层 GatedDeltaNet 的 all-to-all 数量、通信耗时、conv 耗时和总 step throughput。

## 后续可探索方向

这次 PR 最终选择了低风险的结构性重排。后续如果还要继续压 SP 速度，可以基于这版再做：

- 重新评估 CUDA stream/event 的 q/k/v 通信-计算流水，但需要严格验证 autograd collective、compile 和 stream lifetime。
- 将 q/k/v 三次 all-to-all 合并或 batched 化，前提是能处理 q/k 与 v 维度可能不同的问题。
- 对 `g`/`beta` 的 all-to-all 做融合，因为两者 shape 一致。
- 如果未来 bias 打开，先修正 q/k/v bias split，再考虑相关性能。

## 参考

- PR: https://github.com/InternLM/xtuner/pull/1558
- 核心文件: `xtuner/v1/module/attention/gated_deltanet.py`
- 通信实现: `xtuner/v1/ops/comm/all_to_all.py`
- 主提交: `202046d2 optimize linear attention of sp`
- 最终提交: `aaef1ad2 fix lint`
