# XTuner Qwen3.5 MoE 开启 EP 时的 FSDP2 切分逻辑

这份笔记只解释当前代码里实际存在的逻辑，不额外推导没有代码支撑的结论。重点代码路径：

- `xtuner/v1/config/fsdp.py`: `FSDPConfig`
- `xtuner/v1/train/trainer.py`: Trainer 初始化、data mesh、训练主循环
- `xtuner/v1/engine/train_engine.py`: build model、forward/backward、grad norm、optimizer step
- `xtuner/v1/model/moe/moe.py`: MoE 模型、EP mesh、FSDP2 fully_shard、grad 校准
- `xtuner/v1/module/decoder_layer/moe_decoder_layer.py`: MoE layer forward
- `xtuner/v1/module/grouped_linear/moe_group_linear.py`: expert 参数的 EP shard
- `xtuner/v1/module/dispatcher/*.py`: EP dispatch/combine 通信
- `xtuner/v1/model/compose/base.py` 和 `xtuner/v1/model/compose/qwen3_vl/modeling_qwen3_vl.py`: VLM compose 模型如何把 forward 交给 language model

## 1. 先说结论

Qwen3.5 MoE 在 XTuner v1 里的并行关系可以简单理解成：

```text
world ranks
  ├── FSDP 维度: 同一批 expert shard 继续做 FSDP2 参数切分
  └── EP 维度: 不同 rank 持有不同 expert，token 通过 dispatcher 发到对应 expert rank
```

如果 `world_size=64, ep_size=8`，MoE language model 的核心 mesh 是：

```text
(world_size // ep_size, ep_size) = (8, 8)
mesh_dim_names = ("default.fsdp", "default.ep")
```

也就是每个 EP group 有 8 个 rank；每个 EP group 内按 expert 分工。FSDP2 使用 `default.fsdp` 这个子 mesh 做参数 sharding。expert 参数本身先按 EP 维度 `Shard(0)`，再被 FSDP2 管理；非 expert 参数在 EP 维度上复制，然后被 FSDP2 管理。

这不是 Trainer 的 `("dp", "sp", "tp")` data mesh。Trainer 的 data mesh 用于 dataloader、sequence parallel、tensor parallel 这类外层数据/序列处理；MoE 的 `("*.fsdp", "*.ep")` mesh 是 language model 在 `fully_shard()` 里单独创建和使用的。

## 2. 配置入口

`FSDPConfig` 里和本文最相关的是：

```python
tp_size = 1
ep_size = 1
reshard_after_forward = True
recompute_ratio = 1.0
cpu_offload = False
param_dtype = torch.bfloat16
reduce_dtype = torch.bfloat16
fp32_lm_head = False
hsdp_sharding_size = None
```

代码位置：`xtuner/v1/config/fsdp.py`。

Qwen3.5 35BA3 的文本 MoE 配置在 `Qwen3_5_VLTextMoE35BA3BConfig` / `Qwen3_5_VLTextMoE35BA3BSplitConfig` 中：

- `num_hidden_layers = 40`
- `hidden_size = 2048`
- `n_routed_experts = 256`
- `n_shared_experts = 1`
- `num_experts_per_tok = 8`
- `moe_intermediate_size = 512`
- `layers_type`: 每 4 层里第 4 层是 full attention，其余是 linear attention

代码位置：`xtuner/v1/model/moe/qwen3_5_text.py`、`xtuner/v1/model/moe/qwen3_5_text_split.py`。

Trainer 会在 `_resolve_config_conflicts()` 里对齐 model config 和 FSDP config 的 `ep_size`：

- 如果 model 是 MoE 且 `model_cfg.ep_size == 1`，但 `fsdp_cfg.ep_size != 1`，会把 `model_cfg.ep_size` 改成 `fsdp_cfg.ep_size`。
- 如果 model 是 MoE 且 `fsdp_cfg.ep_size == 1`，但 `model_cfg.ep_size != 1`，会把 `fsdp_cfg.ep_size` 改成 `model_cfg.ep_size`。

代码位置：`xtuner/v1/train/trainer.py::_resolve_config_conflicts()`。

## 3. 两套 mesh 不要混淆

Trainer 初始化时会建立 data mesh：

```python
data_mesh = init_device_mesh(
    device,
    (dp_size, sp_size, tp_size),
    mesh_dim_names=("dp", "sp", "tp"),
)
```

这里的 `sp` 用于 `seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)` 和 loss ctx 构造。它不直接决定 MoE expert 如何切。

MoE language model 自己在两个地方处理 EP mesh：

1. `MoE.__init__()` 里，如果 `config.ep_size > 1`，先创建：

```python
init_device_mesh(
    DEVICE,
    (world_size // ep_size, ep_size),
    mesh_dim_names=("default.dp", "default.ep"),
)["default.ep"]
```

这个阶段主要是为了构建 MoE layer 时把 `ep_mesh` 传给 expert 和 dispatcher。

2. `MoE.fully_shard()` 调 `_init_device_mesh()` 时，会重新创建 FSDP/EP 共根 mesh：

```python
model_mesh = init_device_mesh(
    device,
    (world_size // fsdp_config.ep_size, fsdp_config.ep_size),
    mesh_dim_names=("default.fsdp", "default.ep"),
)
self.ep_mesh = model_mesh["default.ep"]
self.fsdp_mesh = model_mesh["default.fsdp"]
```

这里有一大段注释强调：FSDP 要求 DP/TP 或多个子 mesh 共享同一个 root mesh，所以 `_init_device_mesh()` 会确保新的 `ep_mesh` 和之前构造 layer 时用的 `ep_mesh` 在名字和 rank 排列上对齐。

当前代码还明确限制：`hsdp_sharding_size is not None` 时要求 `ep_size == 1`，也就是 HSDP 和 EP 现在不能一起开。

## 4. 参数在 EP + FSDP2 下怎么放

### 4.1 Expert 参数先按 EP 切

expert 参数来自 `MoEBlock`，里面有两组 grouped linear：

```python
self.fused_w1w3 = build_grouped_linear(hidden_size, 2 * moe_intermediate_size, n_routed_experts, ep_mesh=ep_mesh)
self.fused_w2 = build_grouped_linear(moe_intermediate_size, hidden_size, n_routed_experts, ep_mesh=ep_mesh)
```

`GroupedLinear` 的权重形状是：

```python
weight = torch.empty(num_routed_experts * out_features, in_features)
```

如果 `ep_mesh.size() > 1`，权重会变成：

```python
distribute_tensor(weight, ep_mesh, [Shard(0)])
```

所以它是沿 flatten 后的第 0 维切。这里容易误解：`weight` 的第 0 维确实是 `num_experts * out_features`，看起来像把 expert 维和 out feature 维混在了一起；但它的排列是 expert-major 的，也就是：

```text
expert0 的所有 out rows,
expert1 的所有 out rows,
...
expertN 的所有 out rows
```

在 Qwen3.5 35BA3 这类配置里，`n_routed_experts=256`，常见 `ep_size=8` 可以整除 expert 数。此时每个 EP shard 的第 0 维长度是：

```text
(256 * out_features) / 8 = 32 * out_features
```

shard 边界刚好落在完整 expert 边界上。因此，在进入 FSDP2 进一步切分之前，每个 EP rank 对应的是 32 个完整 expert 的权重片，而不是每个 expert 的 out feature 被 8 个 EP rank 横向切开。

换句话说，`Shard(0)` 本身只是沿 flatten row 切；它等价于按 expert 切的前提是 `n_routed_experts % ep_size == 0`，并且 shard size 是 `out_features` 的整数倍。当前 dispatcher 里也按 `n_routed_experts // ep_size` 计算每个 rank 的 expert 数，因此代码路径隐含依赖这个对齐关系。

代码位置：`xtuner/v1/module/grouped_linear/moe_group_linear.py`。

### 4.2 非 expert 参数在 EP 维度复制

`MoE.fully_shard()` 中，如果 `ep_mesh.size() > 1`，会调用：

```python
self._replicate_other_params(self)
```

这个函数递归遍历模块，但遇到 `MoEBlock` 会直接 return：

```python
if isinstance(module, MoEBlock):
    return
```

也就是说 expert 参数不被复制；其他参数，例如 embedding、attention、router、shared expert、norm、lm head 等，会在 EP mesh 上 `Replicate()`。

这样做的原因是：EP 维度在这套实现里只负责“expert 属于哪个 rank”和“token 如何发到 expert rank”，不是通用 tensor parallel 维度。除了 routed expert 本体以外，其他模块仍然需要每个 EP rank 都能本地执行完整计算：

- attention / norm / embedding / lm head 这类 dense 模块需要完整权重才能直接 forward。
- router/gate 需要看到所有 expert 的打分，当前 `MoEGate.forward()` 直接用完整 gate weight 做 `F.linear()`，再选 top-k；如果 gate 也按 EP 切，就还需要额外的跨 rank top-k 逻辑，当前代码没有走这条路径。
- shared expert 在代码里不是 `MoEBlock`，因此按 dense MLP 处理；每个 EP rank 都为自己的 token 算 shared expert 分支。

同时，FSDP2 的 shard mesh 是 `fsdp_mesh`，不包含 EP 维度。如果非 expert 参数只是普通本地 Parameter，那么 EP 维度上的多个副本 backward 后会得到各自不同的 grad，optimizer step 后可能彼此发散。把它们显式标成 EP mesh 上的 `Replicate()`，后面的 `scale_and_reduce_grad()` 才能发现这些 replicated 维度，并在 clip/optimizer 前做 all-reduce mean。

代码位置：`xtuner/v1/model/moe/moe.py::_replicate_other_params()`。

### 4.3 FSDP2 再包 layer/module

XTuner 用的是 PyTorch FSDP2 的函数式接口：

```python
from torch.distributed.fsdp import fully_shard
```

实际调用封装在 `BaseModel._fully_shard()` 里：

```python
fully_shard(
    target,
    mesh=mesh,
    mp_policy=mp_policy,
    reshard_after_forward=reshard_after_forward,
    offload_policy=offload_policy,
    ignored_params=ignored_params if ignored_params else None,
)
```

MoE 自己的 `fully_shard()` 包裹顺序是：

1. 每个 decoder layer 逐层 `fully_shard()`。
2. 相邻 layer 设置 `set_modules_to_forward_prefetch([next_layer])`。
3. `embed_tokens` 单独 shard，`reshard_after_forward` 来自 `config.embed_reshard_after_forward`。
4. `norm` 单独 shard。
5. `lm_head` 单独 shard，`reshard_after_forward=False`。
6. 如果有 MTP block，MTP layer 也逐层 shard，并设置 prefetch。
7. 最后对整个 MoE model 再 `fully_shard()` 一次。
8. patch embedding forward：如果 embedding weight 是 DTensor，就用 `weight.to_local()` 做 `F.embedding()`。

注意最后一层 decoder layer 在没有 MTP 时会设置 `reshard_after_forward=False`，lm head 也固定 `False`。这是代码里的显式策略，不是通用 FSDP 规则。

代码位置：`xtuner/v1/model/moe/moe.py::fully_shard()`。

### 4.4 checkpoint/HF loading 为什么需要 LoadSpec

代码里 `BaseModel._init_load_spec()` 的注释说得很直接：MoE 下参数布局既要服务 HF 权重加载/保存，又要服务 backward 后的梯度处理，所以不能只靠 PyTorch 的 `Shard` / `Replicate` 粗粒度信息。

对于 Qwen3.5 split 权重，`to_hf_key_list()` 会把 XTuner 的 fused expert 参数映射到 HF 的每个 expert key。例如：

```text
xtuner: fused_w1w3.weight
HF: expert_i.gate_proj.weight + expert_i.up_proj.weight

xtuner: fused_w2.weight
HF: expert_i.down_proj.weight
```

`_load_fused_hf_param()` 里注释明确写了 EP + FSDP 的二次切分加载逻辑：

1. 先根据 EP shard 取当前 rank 需要的 expert HF keys。
2. 假设 FSDP 和 EP 都沿 `FSDP_SHARD_DIM = 0` 切，再根据 FSDP shard 继续缩小要加载的 key 范围。
3. 算出当前 FSDP local tensor 对应 HF tensor 的 `start/end`，只填本地 shard。

代码位置：`xtuner/v1/model/base.py::_init_load_spec()`、`_load_fused_hf_param()`。

## 5. forward 全流程

### 5.1 VLM compose 模型先准备 language model 输入

如果用的是 `Qwen3_5_VLMoE35BA3Config` 这类 VLM compose 配置，模型结构是：

```text
BaseComposeModel
  ├── vision_tower
  ├── multi_modal_projector
  └── language_model  # Qwen3.5 MoE text model
```

`Qwen3VLForConditionalGeneration.forward()` 会先调用 `_prepare_llm_inputs()`：

1. 用 `language_model.embed_tokens(input_ids)` 得到文本 embedding。
2. 如果有图像/视频，跑 vision tower 和 projector。
3. 用 visual token mask 把 visual embedding 写回 `inputs_embeds`。
4. 把 `input_ids=None, inputs_embeds=...` 的 `SequenceContext` 交给 `language_model`。

如果是纯文本的 text MoE config，则直接进入 language model。

### 5.2 language model 的 micro-batch forward

`TrainEngine.train_step()` 会按 `intra_layer_micro_batch` 分组。如果 `intra_layer_micro_batch == 1`，模型收到单个 `seq_ctx`；否则收到 `seq_ctx_list` 和 `loss_ctx_list`。

MoE 的 `_micro_batch_forward()` 大致是：

1. 多个 micro-batch 先 concat 做 embedding：

```python
cat_input_ids = torch.cat([ctx.input_ids for ctx in seq_ctx_list], dim=1)
cat_hidden_states = self.embed_tokens(cat_input_ids)
```

如果 compose 模型已经传入 `inputs_embeds`，则 concat `inputs_embeds`。

2. concat position ids，统一算 rotary embedding。
3. 如果前面有 dense layer，就在 concat 后的 hidden states 上跑。
4. 到第一层 MoE 时，把 concat hidden states chunk 回 `hidden_states_list`。代码里还做了 `clone()`，注释说这是为了规避 `async_save_on_cpu` 和 chunk 共享 storage 导致的 nan grad norm 问题。
5. 后续每个 MoE layer 都以多个 micro-batch 的形式调用：

```python
layer_results = decoder_layer(
    *hidden_states_list,
    position_embeddings=position_embeddings_list,
    seq_ctx=seq_ctx_list,
)
```

6. 每层返回每个 micro-batch 的 hidden states、router logits、router weights。router 统计会 concat 后喂给 aux loss。
7. 所有 layer 结束后，concat hidden states，过 final norm 和 lm head，得到 loss。
8. balancing loss / z loss / tokens_per_expert_global 在 `aux_loss.finalize()` 里产出。

代码位置：`xtuner/v1/model/moe/moe.py::_micro_batch_forward()`。

### 5.3 单个 MoE decoder layer 做什么

单 micro-batch 的 `_forward()` 顺序是：

```text
input
  -> _pre_moe_forward
       input_layernorm
       self_attn
       residual add
       post_attention_layernorm
       gate/router
  -> dispatcher.dispatch_preprocess
  -> dispatcher.dispatch
  -> dispatcher.dispatch_postprocess
  -> experts grouped GEMM
  -> dispatcher.combine_preprocess
  -> dispatcher.combine
  -> dispatcher.combine_postprocess
  -> shared_experts
  -> _post_moe_forward
       routed expert output + shared expert output + residual
```

多 micro-batch 的 `_micro_batch_forward()` 顺序是为了 overlap 调整过的：

1. 对每个 micro-batch 先做 `_pre_moe_forward()` 和 `dispatch_preprocess(async_op=True)`。
2. 再逐个 micro-batch 做 `dispatch(async_op=True)`、`dispatch_postprocess(async_op=True)`、expert GEMM、`combine_preprocess(async_op=True)`。
3. 然后逐个 micro-batch 发起 `combine(async_op=True)`。
4. 最后逐个 micro-batch 做 `combine_postprocess(async_op=True)` 和 `_post_moe_forward()`。

这就是 XTuner 当前代码里和 Domino EP 思路对应的地方：让多个 micro-batch 的 dispatch/combine 通信和 expert/attention 计算尽量错开。

代码位置：`xtuner/v1/module/decoder_layer/moe_decoder_layer.py::_forward()`、`_micro_batch_forward()`。

## 6. dispatcher 如何把 token 发给 expert

dispatcher 的选择规则在 `build_dispatcher()`：

- 如果 `ep_group is None` 或 `ep_group.size() == 1`，使用 `NaiveDispatcher`。此时即使配置了 `dispatcher="deepep"`，也会 warning 并不使用 DeepEP。
- 如果 `ep_size > 1` 且 `dispatcher is None`，默认用 `all2all`。
- 如果配置 `dispatcher="deepep"`，使用 `DeepEPDispatcher`。
- 如果配置 `dispatcher="agrs"`，使用 `MoEAGRSDispatcher`，但 `MoEConfig.build()` 要求 AGRS 的 `ep_size == router_n_groups == 8`。

### 6.1 all2all dispatcher

`TorchAll2AllDispatcher` 的文件头注释列了完整步骤：

1. 先按 expert 维度 permute hidden states，并记录 row id map。
2. EP group 内交换 routing 信息，算 all-to-all 的 input/output splits。
3. dispatch 阶段调用 torch all-to-all，把 token 发到 expert 所在 rank。
4. dispatch 后再 permute 成 grouped GEMM 需要的 expert 连续顺序。
5. expert forward 后先 unpermute。
6. combine 阶段再 all-to-all，把结果发回原 rank。
7. postprocess 再 unpermute 回原 token 顺序。

异步路径里，dispatch/combine 在 `_comm_stream` 上运行，并通过 CUDA event 串依赖。

### 6.2 DeepEP dispatcher

`DeepEPDispatcher` 使用 `deep_ep` 的 `Buffer.dispatch()` / `Buffer.combine()`，并用 `EventOverlap` 表达依赖。

代码里对 backward 的注释也很清楚：

- dispatch 的 backward 实际是 combine。
- combine 的 backward 实际是 dispatch。

这和 all-to-all 版本的反向通信含义一致：forward 把 token 发出去，backward 要把对应梯度按反方向发回来。

## 7. backward 之后 grad 如何校准

训练循环的顺序是：

```text
TrainEngine.train_step()
  for each grad accumulation chunk:
    output = model(...)
    loss = total loss
    loss.backward()

Trainer.fit()
  grad_norm = engine.clip_grad_norm()
  engine.step_optimizer(grad_norm)
```

`clip_grad_norm()` 里的第一步是：

```python
self.model.scale_and_reduce_grad()
```

所以 EP/FSDP 的额外 grad 校准发生在：

```text
loss.backward() 之后
grad norm / clip 之前
optimizer.step() 之前
```

如果是 VLM compose 模型，`BaseComposeModel.scale_and_reduce_grad()` 只代理到：

```python
self.language_model.scale_and_reduce_grad()
```

这是因为这里需要特殊处理的是 language model 的 MoE/EP 参数。

### 7.1 expert 参数：不做跨 EP all_reduce，只除以 ep_size

`MoE.scale_and_reduce_grad()` 中：

```python
if ep_enabled and ".experts" in name:
    param.grad.div_(self.ep_mesh.size())
    continue
```

代码注释写明：expert 参数在 EP rank 上是唯一的，不需要跨 rank reduce；这里只除以 `ep_size`，保持有效平均尺度。

这里不要理解成所有 expert 参数都被复制了。前面已经看到 expert weight 是 `Shard(0)` 放在 EP mesh 上的；每个 EP rank 只持有自己的 expert 子集。

### 7.2 replicated 参数：按 Replicate 维度 all_reduce 求平均

对于 trainable 参数，如果它是 DTensor，代码会找出 placement 里所有 `Replicate()` 对应的 mesh dim：

```python
replicate_dim_names = tuple(
    param.device_mesh.mesh_dim_names[i]
    for i, p in enumerate(param.placements)
    if isinstance(p, Replicate)
)
```

如果存在 replicated dim：

1. 如果 replicated dim 不止一个，先 flatten 成 1D mesh。
2. 取本地 grad。
3. 先本地 `grad.div_(flat_mesh.size())`。
4. 按 process group 分桶。
5. 每个 group 用 `_coalescing_manager` 做一组 coalesced `dist.all_reduce(..., ReduceOp.SUM)`。

也就是说 replicated 参数的 grad 最终是跨 replicated ranks 的平均值。这里先除再 sum，等价于 all-reduce mean。

代码位置：`xtuner/v1/model/moe/moe.py::scale_and_reduce_grad()`。

## 8. 一步训练的完整串联

把上面串起来，一步训练可以看成：

```text
初始化:
  Trainer 对齐 model_cfg.ep_size 和 fsdp_cfg.ep_size
  Trainer 建 data_mesh(dp, sp, tp)
  TrainEngine 在 meta device 上 build model
  MoE.__init__ 创建早期 ep_mesh，并把 ep_mesh 传给 MoE layer/expert/dispatcher
  MoE.fully_shard 创建共根 model_mesh(fsdp, ep)
  expert 参数: EP Shard(0)
  非 expert 参数: EP Replicate
  decoder layer / embed / norm / lm_head / MTP: PyTorch FSDP2 fully_shard
  from_hf/init_weights 填本地 shard

forward:
  dataloader batch -> SequenceContext
  如果 sp_size > 1，seq_ctx 按 sp_mesh split
  compose VLM 准备 inputs_embeds，或者 text MoE 直接处理 input_ids
  MoE concat 多个 intra-layer micro-batch 做 embedding/rope
  到 MoE layer 后拆回多个 micro-batch
  每层:
    attention + router
    dispatch token 到 expert rank
    本地 grouped GEMM 跑 expert
    combine token 回原 rank
    shared expert + residual
  final norm + lm_head + aux loss

backward:
  loss.backward()
  dispatcher autograd function 负责反向通信
    dispatch backward = combine-like communication
    combine backward = dispatch-like communication
  异步 dispatcher 通过 CUDA event 或 DeepEP EventOverlap 控制等待和完成点

grad 校准:
  Trainer 调 engine.clip_grad_norm()
  clip 前先 model.scale_and_reduce_grad()
  expert grad: div ep_size，不跨 EP all_reduce
  replicated DTensor grad: 按 Replicate mesh group 求平均 all_reduce
  然后计算 grad norm、clip、optimizer.step()
```

## 9. review 时建议重点看这些不变量

1. `model_cfg.ep_size` 和 `fsdp_cfg.ep_size` 必须一致。Trainer 会尝试自动对齐，但 `MoE.fully_shard()` 里还有 assert。
2. `world_size` 必须能被 `ep_size` 整除，因为 MoE mesh shape 是 `(world_size // ep_size, ep_size)`。
3. HSDP 当前不能和 EP 同开，代码里要求 `hsdp_sharding_size is None` 或 `ep_size == 1`。
4. EP=1 时不会真的使用 `deepep` / `all2all`，会回退 `NaiveDispatcher`。
5. 多 micro-batch overlap 依赖 `intra_layer_micro_batch > 1`；单 micro-batch 仍是普通顺序 forward。
6. Qwen3.5 split config 的 expert 权重映射和加载依赖 `LoadSpec(FUSED)`，不要把它当成普通 dense 参数加载。
7. grad 校准发生在 backward 后、clip/optimizer 前；如果绕过 `TrainEngine.clip_grad_norm()`，就会绕过 `scale_and_reduce_grad()`。
