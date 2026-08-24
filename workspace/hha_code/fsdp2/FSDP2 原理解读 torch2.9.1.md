# FSDP2 原理解读（基于 torch 2.9.1）

本文基于当前环境的 `torch 2.9.1+cu128` 源码、PyTorch 2.9 官方文档、FSDP2 RFC，以及旧文档《FSDP 基础认知.md》重新整理。旧文档中很多说法来自早期 RFC，方向基本正确，但有些细节在 2.9.1 中需要校正，尤其是 `reshard_after_forward=None`、meta 参数初始化、root lazy init、通信流复用这些点。

本文讨论的是 FSDP2，也就是 `torch.distributed.fsdp.fully_shard`，源码主体在：

- `/mnt/shared-storage-user/huanghaian/miniconda3/envs/pt29_all_env/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fully_shard.py`
- `/mnt/shared-storage-user/huanghaian/miniconda3/envs/pt29_all_env/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_state.py`
- `/mnt/shared-storage-user/huanghaian/miniconda3/envs/pt29_all_env/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py`
- `/mnt/shared-storage-user/huanghaian/miniconda3/envs/pt29_all_env/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py`
- `/mnt/shared-storage-user/huanghaian/miniconda3/envs/pt29_all_env/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`

## 1. 一句话心智模型

FSDP2 的核心就是：

> 平时每张卡只保存每个参数的一片 `DTensor`；某个模块真正要计算时，FSDP2 临时 all-gather 出完整参数，计算完再释放完整参数；反向算出完整梯度后，reduce-scatter 回每张卡自己的梯度分片；优化器只更新本卡保存的参数分片。

和 DDP 对比：

- DDP：每张卡都有完整参数；反向后做 all-reduce，让每张卡得到一样的完整梯度。
- FSDP2：每张卡只有参数分片；前向/反向需要完整参数时 all-gather；反向后用 reduce-scatter 得到梯度分片。

所以可以把 FSDP 看成把 DDP 的“完整梯度 all-reduce”拆成两类动作：

- forward/backward 计算前：`all_gather(param_shard) -> full_param`
- backward 计算后：`reduce_scatter(full_grad) -> grad_shard`

optimizer state 为什么也省显存？因为优化器看到的是 module 上注册的 sharded `DTensor` 参数，`optimizer.step()` 只为本地参数分片维护状态，而不是为完整参数维护状态。

## 2. FSDP2 和 FSDP1 的关键差异

FSDP1 的核心抽象是 `FlatParameter`：把一组参数 flatten/concat 成一个大参数，再切分。FSDP2 不再走 flat parameter，而是“逐参数切分”：每个原始参数独立变成一个 sharded `DTensor`。

这带来几个直接变化：

- 参数 FQN 更干净：FSDP2 不包一层 `_fsdp_wrapped_module`，而是在原 module 上原地注册 hook 和替换 class，所以 `state_dict()` 里的名字基本保持原模型名字。
- 单个参数更容易理解：`linear.weight` 还是 `linear.weight`，只是它在非计算期是 `DTensor(Shard(...))`。
- sharded state dict 更自然：训练时的参数本来就是 sharded `DTensor`，保存 sharded state dict 不需要先 all-gather 成 full。
- mixed frozen/non-frozen 参数更容易放在同一个通信组里，不像 flat parameter 那样容易被 requires_grad、dtype、padding 等细节绑住。
- 通信依然按 group 聚合，不是每个参数单独发一次 collective；FSDP2 只是存储表示是 per-parameter sharding。

官方文档也明确说：FSDP2 使用 DTensor-based dim-i per-parameter sharding，保留相近吞吐，同时简化状态表示、meta 初始化和 sharded state dict。

## 3. `fully_shard()` 到底做了什么

入口是 `_fully_shard.py::fully_shard`。主流程可以概括成：

1. 检查 `module` 和 `mesh`，构造 `FSDPMeshInfo` 或 `HSDPMeshInfo`。
2. 解释 `reshard_after_forward`，得到 `post_forward_mesh_info`。
3. 用 `@contract(state_cls=FSDPState)` 给 module 绑定一个 `FSDPState`。
4. 注册 forward pre/post hook。
5. DFS 找到这个 FSDP 单元真正管理的 modules、parameters、buffers。
6. 把非 meta 的参数/缓冲移动到 mesh 对应设备。
7. 如果有参数，创建一个 `FSDPParamGroup`，里面再为每个参数创建 `FSDPParam`。
8. 动态改 module 的 Python class，让它同时是原 class 和 `FSDPModule`，从而多出 `reshard()`、`unshard()` 等 API。

对应源码：

- `_fully_shard.py:193-204`：构造 1D FSDP 或 2D HSDP mesh info。
- `_fully_shard.py:205-211`：处理 `reshard_after_forward`。
- `_fully_shard.py:217-234`：创建/初始化 state、收集参数并创建 `FSDPParamGroup`。
- `_fully_shard.py:241-249`：动态生成 `FSDP{OriginalClass}`，让 module 变成 `FSDPModule`。

这个设计不是“包一层 wrapper”，而是“原地改造 module”。所以用户手里的 `model` 对象还是同一个对象，FQN 也更稳定。

## 4. 几个核心对象的分工

FSDP2 的对象关系是：

```text
fully_shard(module)
  -> FSDPState             每次 fully_shard 调用对应一个 state
      -> FSDPParamGroup?   如果这个 state 管理了参数，则有一个 param group
          -> FSDPParam[]   group 内每个参数对应一个 FSDPParam
```

### 4.1 `FSDPModule`

`FSDPModule` 是一个轻量 mixin。它不存主要状态，只提供用户可调用 API：

- `unshard(async_op=False)`：手动 all-gather 当前 module 的参数。
- `reshard()`：手动释放完整参数，恢复 sharded 参数。
- `set_reshard_after_forward()`：运行期改前向后是否 reshard。
- `set_reshard_after_backward()`：梯度累积时可让反向后保留完整参数，减少下一轮 all-gather。
- `set_requires_gradient_sync()`：类似 FSDP1/DDP 的 `no_sync`。
- `set_modules_to_forward_prefetch()` / `set_modules_to_backward_prefetch()`：显式控制预取。
- `set_custom_all_gather()` / `set_custom_reduce_scatter()`：替换通信实现。

一个常见坑：如果你直接调用 `model.forward(x)`，PyTorch 的 module hooks 不会触发。FSDP2 文档要求调用 `model(x)`。如果确实有 `generate()` 这类自定义前向方法，要用 `register_fsdp_forward_method(module, "generate")` 或手动 `unshard()`。

### 4.2 `FSDPState`

`FSDPState` 管“模块级生命周期”，包括：

- 注册 forward pre/post hook。
- 在第一次 forward 时做 lazy init。
- 判断 root FSDP state，并为整个 root module tree 建立共享上下文。
- 管 root pre-forward / root post-forward / root post-backward final callback。
- 注册 pre-backward hook 到 forward outputs 上。
- 保存 forward/backward prefetch 目标。

源码重点：

- `_fsdp_state.py:103-118`：注册 forward pre/post hooks；如果传入的是 `list[nn.Module]`，则注册 group hooks。
- `_fsdp_state.py:153-193`：lazy init，第一次从 root forward 进入时确定所有 FSDP state。
- `_fsdp_state.py:195-201`：所有 state 共享同一个 `FSDPCommContext`，所以不是每个 layer 永远各用各的通信流。
- `_fsdp_state.py:228-280`：pre-forward / post-forward 主逻辑。
- `_fsdp_state.py:282-318`：pre-backward 和 final callback。
- `_fsdp_state.py:336-350`：在 output tensor 上注册 backward hook，并通过 autograd engine queue final callback。

### 4.3 `FSDPParamGroup`

`FSDPParamGroup` 管“一个通信组”。它只在当前 `fully_shard()` 管理了非空参数时存在。

它负责：

- group 内参数一起 unshard：一次打包 all-gather。
- group 内梯度一起 reduce-scatter。
- 维护 group 的 `SHARDED / SHARDED_POST_FORWARD / UNSHARDED` 状态。
- 管通信流、event、prefetch、state_dict pre-hook。
- 管 HSDP 的 reduce-scatter 后 all-reduce。

源码重点：

- `_fsdp_param_group.py:123-149`：为 group 内每个参数创建 `FSDPParam`。
- `_fsdp_param_group.py:249-266`：lazy init，检查 meta、CPU offload、mixed precision dtype。
- `_fsdp_param_group.py:299-410`：unshard 和 wait_for_unshard。
- `_fsdp_param_group.py:419-429`：reshard。
- `_fsdp_param_group.py:431-450`：forward 的 group-level hook。
- `_fsdp_param_group.py:459-570`：backward 的 group-level unshard、reshard、reduce。
- `_fsdp_param_group.py:701-722`：保存/加载 state_dict 前强制切回 sharded。

### 4.4 `FSDPParam`

`FSDPParam` 管一个参数从原始 tensor 到 sharded DTensor、再到临时 unsharded parameter 的变换。

它脑子里同时维护几类 tensor：

- original parameter：调用 FSDP 前 module 上的参数对象。
- sharded parameter：切分后的 `DTensor`，非计算期注册在 module 上。
- all-gather input：发给 collective 的本地输入，通常就是 sharded local tensor 的 1D view，可能经过 dtype cast 或扩展转换。
- all-gather output：collective 结果 buffer。
- unsharded parameter：forward/backward 真正计算用的完整参数，计算期注册在 module 上。
- sharded post-forward parameter：`reshard_after_forward=int` 时前向后保留的中间切分形态。

源码注释把这几类 tensor 写得很清楚：`_fsdp_param.py:31-65`。

## 5. 参数怎么被切成 shard

FSDP2 默认 `Shard(0)`，也就是按参数第 0 维切。对 `weight` 这种二维矩阵来说，通常就是按行切；对 embedding 也是按 vocab/row 切。

源码路径在 `_fsdp_param.py::_init_sharded_param`：

- `_fsdp_param.py:264-271`：要求参数已经在目标 device 或 meta 上，并且当前不支持 non-contiguous 参数。
- `_fsdp_param.py:272-279`：默认 `Shard(0)`，也允许 `shard_placement_fn` 指定别的维度。
- `_fsdp_param.py:331-341`：普通 tensor 构造 DTensorSpec；HSDP 用 `(Replicate(), Shard(...))`。
- `_fsdp_param.py:351-360`：非 0 维切分目前要求能整除 world size。
- `_fsdp_param.py:362-383`：用 `torch.chunk` 切出当前 rank 的 shard，并为了 uneven dim0 sharding 做 padding。
- `_fsdp_param.py:389-394`：把本地 shard 包成 `nn.Parameter(DTensor(...))`。
- `_fsdp_param.py:397-405`：把 module 上的参数替换成 sharded 参数。

注意：FSDP2 是“每个参数自己切”，但通信时会把同一个 param group 内的多个参数打包一起通信。所以不要把“per-parameter sharding”误解成“每个参数独立 collective”。

## 6. `reshard_after_forward` 的准确语义

`reshard_after_forward` 控制 forward 结束后 module 上注册什么形态的参数：

- `True`：forward 后立刻 reshard 回完整 FSDP shard。省显存，但 backward 前要再 all-gather 一次。
- `False`：forward 后保留 unsharded 参数。多占显存，但 backward 可以少一次 all-gather。
- `None`：2.9.1 的默认值。非 root 按 True；root 在 lazy init 中自动改成 False。
- `int`：forward 后不是切回整个 FSDP shard world size，而是切到一个较小 world size。这样 backward 的 all-gather 可以只在较小组里做，但显存比 True 更高。这个值必须是 shard mesh size 的 factor。

源码细节：

- `_fully_shard.py:89-90`：2.9.1 函数签名里默认是 `reshard_after_forward=None`。
- `_fully_shard.py:205-211`：如果用户没传，先按 True 构造 post-forward mesh info。
- `_fsdp_state.py:183-186`：lazy init 发现当前 state 是 root 且该值是自动值，才把 root 的 `post_forward_mesh_info` 改成 None，也就是 root 不 forward 后 reshard。
- `_fsdp_init.py:20-63`：`int` 情况会构造一个新的 HSDP-like post-forward mesh。
- `_fsdp_param_group.py:419-429`：forward 中 post hook 根据 `post_forward_mesh_info` 决定 reshard 形态。

这里容易误解：root 自动改成 `False` 只发生在用户没有显式传值，也就是 `reshard_after_forward=None`。如果你显式传 `True`，`_auto_reshard_after_forward=False`，lazy init 不会帮你改。

和 ZeRO 类比：

- `True` 类似 ZeRO-3：参数 forward 后释放，backward 前再 gather。
- `False` 类似 ZeRO-2 的参数行为：forward 后参数仍完整保留，梯度仍 sharded/reduced。
- `int` 接近 ZeRO++ hpZ 的层级思想：用更小 group 保留一部分完整度，少一些跨大组通信，多一些显存。

## 7. 初始化阶段：哪些模块和参数归当前 FSDP 单元管

`fully_shard()` 不会盲目拿整个 module tree 所有参数。它从传入 module 做 DFS，遇到下面情况会停：

- 子模块已经被另一个 `fully_shard()` 管理。
- 子模块注册了不可和 FSDP 组合的 API，目前源码里主要是 `replicate`。
- 参数在 `ignored_params` 中。

源码：

- `_fsdp_init.py:133-167`：DFS 收集 managed modules。
- `_fsdp_init.py:147-153`：碰到 non-composable API 或 nested FSDP state 就停止递归。
- `_fsdp_common.py:_is_composable_with_fsdp`：如果 composable registry 里有 `replicate`，返回 False。
- `_fsdp_init.py:183-208`：只收集当前 managed modules 的 non-recursive parameters/buffers，并去重。

这也是为什么官方和源码都强调 bottom-up：

```python
for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)
```

先 shard 每层，会让每层形成自己的通信组；最后 shard root，只管理剩下没被子 FSDP 管的参数，比如 embedding、lm_head、final norm 等。这样既能省显存，又能让下一层 all-gather 和当前层 compute overlap。

如果只对 root 调一次 `fully_shard(model)`，功能上可能能跑，但整个模型只有一个大 param group：forward 前一次性 all-gather 所有参数，forward 后再释放。这样峰值显存和 overlap 都很差，基本失去 layer-by-layer FSDP 的意义。

如果只 shard 子层而不 shard root，root 上剩余参数没有被 FSDP/DDP 管理，它们的梯度不会自动跨 DP 同步。正确做法通常是“子层 bottom-up + root 再 fully_shard”。

## 8. meta 初始化的正确理解

旧文档里有一个重要校正：FSDP2 不会在 `fully_shard()` 时自动把 meta 参数 materialize 成真实 CUDA 参数。

2.9.1 源码里：

- `_fsdp_init.py:211-223`：移动 states 到 device 时，如果 tensor 是 meta，直接 `continue`。
- `_fsdp_param.py:264-267`：`FSDPParam` 允许参数仍在 meta 上。
- `_fsdp_param_group.py:249-266`：第一次 forward 的 lazy init 会调用 `_validate_no_meta_params()`。
- `_fsdp_param_group.py:768-780`：如果 sharded 参数还在 meta，会直接报错，并提示调用 `module.to_empty(device)` 和 `reset_parameters()`。

所以正确流程通常是：

```python
with torch.device("meta"):
    model = Transformer(...)

for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

model.to_empty(device="cuda")
model.reset_parameters()  # 或 load sharded/DTensor state dict
```

这样做的好处是：`fully_shard()` 已经把参数对象替换成 sharded DTensor，`to_empty()` materialize 的是每个 rank 的本地 shard，而不是完整参数。峰值显存更接近 sharded 参数大小。不过真实峰值还取决于 checkpoint 加载方式、dtype、临时 buffer、optimizer 初始化、是否有 CPU offload、是否有额外 broadcast/all-gather，不能简单永远写成 `参数量 * dtype_size / fsdp_size`。

## 9. Forward 运行时：参数什么时候完整、什么时候释放

一次普通 forward 的顺序是：

```text
root pre-forward
  lazy init
  all-gather streams 等待 optimizer/current stream
  输入搬到目标 device

每个 FSDP state pre-forward
  cast forward inputs
  param group unshard: all-gather 参数
  wait_for_unshard: copy-out，module 上切到完整参数
  可选 forward prefetch 下一个 group

用户 forward compute

param group post-forward
  根据 reshard_after_forward 决定是否释放完整参数
  记录 post_forward_order，供 backward prefetch 使用

state post-forward
  在 outputs 上注册 pre-backward hook
  root post-forward 清理最后一个延迟释放的 all-gather result
  cast forward outputs
```

关键源码：

- `_fsdp_state.py:120-151`：root pre-forward，等待 optimizer 后的 event/current stream，并移动输入到 device。
- `_fsdp_state.py:228-252`：state pre-forward，做输入 cast、param group pre-forward、forward prefetch。
- `_fsdp_param_group.py:431-440`：param group pre-forward 调 `unshard()` 和 `wait_for_unshard()`。
- `_fsdp_param_group.py:443-457`：post-forward 调 `reshard()` 并记录 forward 顺序。
- `_fsdp_state.py:254-280`：state post-forward 注册 pre-backward hook，root 清理最后一个 all-gather result。

重要点：FSDP2 认为“参数的 live interval”默认就是 module 的 `forward()`。如果你在父模块里直接访问子模块参数、或者绕过 `__call__` 直接调用 `forward()`，hook 不触发，module 上可能还是 sharded DTensor，行为就不符合预期。

## 10. Backward 运行时：为什么需要两个 hook

Backward 里 FSDP2 要做两件事：

1. 在当前 module 的参数梯度计算前，保证参数是完整的。
2. 在当前 module 的参数梯度计算后，释放完整参数，并把完整梯度 reduce-scatter 成 shard。

源码使用了两类 hook：

- `FSDPState` 在 forward output tensor 上注册 hook：当输出梯度出现时，触发 pre-backward。
- `FSDPParamGroup` 在 forward input tensor 上通过自定义 autograd Function 注册 post-backward：当这个模块的输入梯度计算完时，说明这个模块的参数梯度也算完了，可以 reduce-scatter。

上面是当前环境普通 eager 的默认路径，因为 `torch._dynamo.config.skip_fsdp_hooks=True` 且未启用 compiled autograd 时，`_register_post_backward_hook()` 会插入 `RegisterPostBackwardFunction`。如果配置或 compiled autograd 路径不同，源码会避免插入这个自定义 Function，并依赖下一个 `pre_backward()` 或 root final callback 补齐尚未执行的 `post_backward()`。所以读源码时不要把 `RegisterPostBackwardFunction` 理解成所有模式下唯一的 post-backward 入口。

顺序简化为：

```text
output grad ready
  FSDPState._pre_backward
    queue root final callback
    param_group.pre_backward
      unshard 参数
      backward prefetch 前一个 group

autograd 计算本模块参数梯度和输入梯度

RegisterPostBackwardFunction.backward
  param_group.post_backward
    收集 unsharded_param.grad
    reshard 参数
    foreach_reduce: reduce-scatter 梯度

整轮 backward 结束
  root final callback
    补跑没触发的 post_backward
    等 reduce-scatter/all-reduce event
    清理 post_forward_order 等状态
```

源码：

- `_fsdp_state.py:282-291`：pre-backward。
- `_fsdp_param_group.py:459-476`：param group pre-backward。
- `_fsdp_param_group.py:478-570`：post-backward 收集梯度、reshard、reduce。
- `_fsdp_param_group.py:672-699`：在普通 eager 默认路径下给输入 tensor 包自定义 autograd Function；配置/compiled autograd 路径会跳过。
- `_fsdp_state.py:293-318`：root post-backward final callback。

为什么 backward 前还要 all-gather 参数？因为 PyTorch autograd 计算很多参数梯度或输入梯度时需要 forward 时的完整参数值。例如 linear backward 需要 `weight` 算 `grad_input`，也需要 input 算 `grad_weight`。如果 forward 后释放了 full weight，backward 前必须恢复。

## 11. FSDPParam 的三种 sharded state

`_fsdp_param.py:146-161` 定义了三种状态：

- `SHARDED`：module 上注册的是完整 FSDP mesh 上的 sharded `DTensor`。这是正常空闲态。
- `UNSHARDED`：module 上注册的是完整参数。forward/backward compute 期间处于这个状态。
- `SHARDED_POST_FORWARD`：`reshard_after_forward=int` 时的中间态，前向后切到较小 group 的 shard。它不能作为普通计算参数长期使用，原地修改前应该手动 `reshard()`。

状态转移大致是：

```text
SHARDED
  -- pre-forward/pre-backward all-gather -->
UNSHARDED
  -- post-forward True/post-backward -->
SHARDED

UNSHARDED
  -- post-forward False -->
UNSHARDED

UNSHARDED
  -- post-forward int -->
SHARDED_POST_FORWARD
  -- next backward pre-unshard -->
UNSHARDED
```

源码：

- `_fsdp_param.py:537-540`：切回 sharded 并释放 full。
- `_fsdp_param.py:542-574`：切到 post-forward smaller mesh。
- `_fsdp_param.py:576-587`：切到 unsharded，并把 module 参数替换为 full 参数。

## 12. “释放完整参数”为什么用 storage resize

FSDP2 不是简单 `del full_param`。原因是 autograd 在 forward 中可能保存了参数或参数 view 的引用用于 backward。如果直接换对象或删对象，别名关系会被破坏。

源码注释 `_fsdp_param.py:58-65` 说明：FSDP 动态释放/分配 unsharded parameter 时，用 storage resizing 保持 aliasing。默认 torch.Tensor 路径里 all-gather output 和 unsharded parameter 共享底层 data，所以释放就是把相关 tensor 的 storage resize 到 0。

对应实现：

- `_fsdp_param.py:648-672`：`free_unsharded_param()`。
- `_fsdp_param.py:450-527`：`init_unsharded_param()`，从 all-gather output 构造/恢复 full parameter。

可以把它理解成：

> Python 对象和 autograd 引用还在，但它背后的仓库临时清空；backward 真要用之前，FSDP2 再把仓库补回来。

这也是为什么这个逻辑很依赖 hook 调度正确。如果绕过 FSDP hook 使用参数，就可能在“仓库为空”或“参数还是 shard”的状态下计算。

## 13. Collective 怎么打包

FSDP2 的存储是 per-parameter shard，但通信是 per-param-group batched collective。

### 13.1 all-gather

all-gather 主流程在 `_fsdp_collectives.py::foreach_all_gather`：

1. 每个 `FSDPParam` 生成自己的 all-gather input，通常是本地 sharded tensor 的 1D view。
2. 如果 group 内 dtype 不一致，会用 `uint8` view 打包。
3. 分配一个大的 1D `all_gather_output`，大小是 `sum(local_inputs) * world_size`。
4. 当前 rank 的 input 先 copy 到 output 中属于本 rank 的 slice。
5. 调 `dist.all_gather_into_tensor()`。
6. copy-out 时把大 buffer split 成每个参数自己的 `all_gather_outputs`。
7. 每个 `FSDPParam` 用 `torch.as_strided` 把 1D output view 成原始 ND shape，作为 unsharded parameter。

源码：

- `_fsdp_collectives.py:235-289`：copy-in、all-gather。
- `_fsdp_collectives.py:344-430`：copy-out、split。
- `_fsdp_param.py:450-527`：从 output 初始化 unsharded param。

这里为什么有 copy-in/copy-out？因为每个参数独立存 shard，但 NCCL 对一堆小 tensor 单独发 collective 效率差；打包成连续大 buffer 带宽更好，代价是多了内存拷贝。

### 13.2 reduce-scatter

反向 reduce 主流程在 `_fsdp_collectives.py::foreach_reduce`：

1. 收集 group 内各参数的完整梯度 `unsharded_grad`。
2. 如果 `shard_placement_fn` 不是 `Shard(0)`，先变换布局以适配 reduce-scatter。
3. 为每个 grad 计算 padding 后的 full size。
4. 分配一个大的 reduce-scatter input buffer。
5. 用 `chunk_cat` 按 rank shard 顺序把多个完整梯度拼进去。
6. 调 `dist.reduce_scatter_tensor()`。
7. 可选 HSDP：reduce-scatter 后再跨 replicate 维 all-reduce。
8. 把 reduce output view 成各参数的 sharded grad，并注册到 sharded DTensor 参数上。

源码：

- `_fsdp_collectives.py:447-535`：构造 reduce-scatter input 并发起 reduce-scatter。
- `_fsdp_collectives.py:536-562`：HSDP all-reduce。
- `_fsdp_collectives.py:575-623`：把 reduce output 切回每个参数的 sharded grad。
- `_fsdp_collectives.py:638-646`：`chunk_cat` copy-in。

梯度平均也在这里处理：`_get_gradient_divide_factors()` 会根据 dtype、DP size、用户设置的 `gradient_divide_factor` 决定 pre-divide、post-divide 和 collective op。对 fp32/bf16，优先用 NCCL AVG 或 premul sum；对 fp16 等有 overflow 风险的 dtype，会拆成 pre/post scaling。

## 14. 通信流和为什么避免 `record_stream`

FSDP2 在 lazy init 后，root 下所有 FSDP state 共享一个 `FSDPCommContext`。它创建几条 stream：

- `all_gather_copy_in_stream`：把参数 shard copy 到 all-gather staging buffer。
- `all_gather_stream`：发 all-gather collective。
- `reduce_scatter_stream`：发 reduce-scatter，并做梯度除法、view-out 等后处理。
- `all_reduce_stream`：HSDP 的 replicate 维 all-reduce。

源码：

- `_fsdp_param_group.py:48-86`：stream 和 shared communication states。
- `_fsdp_param_group.py:88-98`：implicit prefetch 时使用独立 copy-in/all-gather stream；否则回到 current stream。
- `_fsdp_state.py:195-201`：lazy init 后所有 states 共用 root 的 comm context。

为什么不用 `record_stream`？PyTorch dev-discuss 和 FSDP2 RFC 解释了背景：跨 stream 使用 tensor 时，如果让 CUDACachingAllocator 通过 `record_stream` 自动追踪生命周期，CPU 侧释放并不意味着显存马上可复用，显存峰值可能变得不确定。FSDP2 选择手动 event/wait，把同步责任交给 stream 之间，而不是让 CPU 或 allocator 兜底。

源码中能看到很多 event/wait：

- root pre-forward 等 optimizer stream：`_fsdp_state.py:130-138`。
- all-gather copy-in stream 和 all-gather stream 同步：`_fsdp_collectives.py:247-281`。
- forward 中延迟释放上一个 all-gather result 来 overlap：`_fsdp_param_group.py:340-410`。
- post-forward 最后清理 pending all-gather result：`_fsdp_state.py:264-273`。
- backward 结束等 reduce-scatter/all-reduce event：`_fsdp_state.py:311-318`、`_fsdp_param_group.py:571-597`。

可以把 stream 设计理解成一条流水线：

```text
默认 stream:              layer i compute ---------------- layer i+1 compute
all_gather_copy_in:                      copy-in layer i+1
all_gather_stream:                              all-gather layer i+1
reduce_scatter_stream:    reduce-scatter layer i-1
```

FSDP2 想达到的是：当前层计算时，下层参数通信已经在路上；当前层反向计算时，上一层梯度 reduce-scatter 已经在路上。

## 15. MixedPrecisionPolicy

`MixedPrecisionPolicy` 在 `_fsdp_api.py` 中定义：

- `param_dtype`：unsharded 参数 dtype，也就是 all-gather dtype 和 forward/backward compute dtype。如果为 None，就用原始 dtype。
- `reduce_dtype`：梯度 reduce-scatter dtype。如果 None 但 `param_dtype` 非 None，则通常继承 compute dtype；如果想 bf16 compute 但 fp32 reduce，可以设置 `param_dtype=torch.bfloat16, reduce_dtype=torch.float32`。
- `output_dtype`：forward output 中浮点 tensor 的 cast dtype，用于不同模块 mixed precision policy 衔接。
- `cast_forward_inputs`：是否把 forward 输入浮点 tensor cast 到 `param_dtype`，默认 True。

源码：

- `_fsdp_state.py:238-246`：pre-forward cast inputs。
- `_fsdp_state.py:274-279`：post-forward cast outputs。
- `_fsdp_param.py:404-420`：lazy init 时设置每个参数的 orig/param/reduce dtype。
- `_fsdp_param_group.py:225-247`：要求 trainable params 的 original dtype 和 reduce dtype 在一个 group 内一致。
- `_fsdp_param.py:738-743`：all-gather input 根据 `param_dtype` cast。

和 autocast 的区别：autocast 是 op-level；FSDP2 的 mixed precision 是 module/parameter communication boundary-level。FSDP2 仍保留 sharded original dtype 参数给 optimizer，用 low precision full param 做 compute。

## 16. HSDP：2D mesh 时发生什么

如果传入 2D `DeviceMesh`，FSDP2 把第 1 维作为 shard 维，第 0 维作为 replicate 维：

```text
mesh shape = (replicate_size, shard_size)
placement = (Replicate(), Shard(0))
```

这就是 HSDP：节点内或小组内做 FSDP shard，节点间或更外层做 replicate。反向时：

1. shard 维做 reduce-scatter。
2. replicate 维再 all-reduce。

源码：

- `_fully_shard.py:196-203`：1D 是 FSDP，2D 是 HSDP。
- `_fsdp_param.py:331-336`：普通 tensor 在 HSDP 下 placement 是 `(Replicate(), Shard(...))`。
- `_fsdp_param_group.py:524-560`：如果 `_is_hsdp`，reduce-scatter 后接 all-reduce。

`reshard_after_forward=int` 也会构造一个 HSDP-like post-forward mesh：把原 shard mesh reshape 成 `(-1, int)`，相当于前向后保留较小 shard group 的参数，减少 backward all-gather 范围。

## 17. State dict 和 checkpoint

FSDP2 默认训练态参数就是 sharded `DTensor`，所以 sharded state dict 是最自然的表示。官方文档明确说 FSDP2 不直接支持 FSDP1 那种 full/local/sharded 三套 state dict API；如果要 full state dict，可以用 DTensor API 或 Distributed Checkpoint 的 higher-level API 转换。

源码层面：

- `_fsdp_param_group.py:701-722`：为有 FSDP 参数的 module 注册 pre-save 和 pre-load hook，在保存/加载前先 `_to_sharded()`。
- `_fsdp_param.py:249-254`：load_state_dict 后调用 `reset_sharded_param()`，修复 padding、本地 tensor 引用等元数据。
- `_fully_shard.py:253-256`：FSDP2 不支持 deepcopy，建议用 state dict 序列化。

这意味着：

- `model.state_dict()` 看到的是 sharded DTensor 参数。
- 保存前 FSDP2 会尽量确保 module 上注册的是 sharded 参数，而不是计算期留下的 unsharded 参数。
- 如果你手动加载普通 full tensor state dict 到 FSDP2 模型，不能假设它会自动按你想要的方式 scatter；更推荐使用 DCP 或预先构造匹配的 DTensor state dict。

## 18. CPU offload

`CPUOffloadPolicy` 表示 sharded 参数、sharded 梯度、optimizer state 在 CPU 上。all-gather 前会把 sharded 参数从 CPU copy 到 device；backward reduce 后 sharded grad 可以 copy 回 CPU。

源码：

- `_fsdp_api.py::CPUOffloadPolicy`：策略定义。
- `_fsdp_param.py:237-240`：记录是否 CPU offload 和 pin memory。
- `_fsdp_param.py:385-388`：初始化时把 padded sharded param 放 CPU/pinned memory。
- `_fsdp_param.py:738-742`：all-gather input 时 copy 到 device。
- `_fsdp_collectives.py:591-606`：reduce output view 成 sharded grad 后 copy 到 CPU，必要时记录 event。
- `_fsdp_param_group.py:782-790`：启用 CPU offload 时 lazy init 要求 sharded params 在 CPU。

## 19. 和 `replicate` / DDP 的组合边界

FSDP2 是 composable API 体系的一部分，但不是任何组合都能嵌套。`_is_composable_with_fsdp()` 当前明确把 `replicate` 视为 non-composable，因此 FSDP DFS 遍历遇到已经 replicate 的模块会停止。

含义是：一个参数不能同时被 FSDP 分片和被 DDP/replicate 复制同步。否则同一个参数的所有权冲突：到底应该保持完整复制，还是按 FSDP mesh 切片？

但不要误解成“每个 module 必须显式 fully_shard 或 replicate”。通常 root `fully_shard(model)` 会接管没有被子 FSDP 接管的剩余参数。真正危险的是：某些有参数的模块既不在任何 FSDP 管理范围内，也没有 DDP/replicate 管理，那它们在 DP ranks 间不会自动同步梯度。

## 20. 常见误区校正

### 20.1 “FSDP2 会自动初始化 meta 参数”

不会。2.9.1 会保留 meta shard，并在 lazy init 时检查到 meta 参数后报错。需要用户调用 `to_empty()` + 初始化，或加载合适的 sharded/DTensor checkpoint。

### 20.2 “root 一定会被强制 `reshard_after_forward=False`”

不准确。只有 `reshard_after_forward=None` 的自动模式会让 root 在 lazy init 时改成 False。用户显式传 True 时不会被覆盖。

### 20.3 “FSDP2 每个参数发一次 all-gather”

不准确。每个参数独立 sharding，但一个 `FSDPParamGroup` 内参数会打包进一个大 buffer，做一次 all-gather / reduce-scatter。

### 20.4 “只包 root 就是标准用法”

功能上可能可跑，但不是推荐用法。官方明确建议 bottom-up 包 transformer block，再包 root。只包 root 会让通信组过大，缺少 layer-wise 显存释放和 overlap。

### 20.5 “FSDP2 的参数在 forward 里一直是 DTensor”

不准确。非计算期 module 上通常是 sharded DTensor；compute 期 pre-forward/pre-backward 会替换成 unsharded 参数。这个 unsharded 参数对普通 tensor 原始参数来说就是普通 `torch.Tensor`/`nn.Parameter` 形态；如果原始参数已经是 DTensor，例如 TP/EP 组合，则 unsharded 参数可能仍是 TP/EP 维度上的 DTensor。

### 20.6 “forward 后释放完整参数就是删除对象”

不准确。FSDP2 需要保留 autograd alias 关系，所以通过 storage resize/free 来释放底层 storage，而不是简单删掉 Python parameter 对象。

### 20.7 “每个 FSDP layer 都有独立 stream，不能 overlap”

初始化时每个 group 先有自己的 `FSDPCommContext` 占位，但 root lazy init 会把 root 下所有 state 的 `comm_ctx` 替换为同一个共享上下文。因此正常从 root forward 进入时，layer 之间共享通信流和 post-forward order，才能做 prefetch/overlap。

## 21. 推荐阅读源码顺序

如果想系统学习，可以按这个顺序读：

1. `_fully_shard.py:86-250`：先看入口和 API 语义。
2. `_fsdp_init.py:20-80,133-223`：看 mesh、managed modules、device/meta 处理。
3. `_fsdp_state.py:73-201`：看 state、hook 注册、lazy init、shared context。
4. `_fsdp_param.py:31-65,146-161,257-405`：看参数状态和 shard 初始化。
5. `_fsdp_param_group.py:299-570`：看 forward/backward runtime。
6. `_fsdp_collectives.py:235-430,447-646`：看 all-gather/reduce-scatter 打包通信。
7. `_fsdp_param.py:450-672`：回头看 unsharded param 的构造和释放。
8. `_fsdp_param_group.py:701-790`：最后看 state_dict、meta、CPU offload 校验。

## 22. 一个简化但准确的完整时间线

假设模型按每个 transformer block bottom-up fully_shard，且 root 也 fully_shard，`reshard_after_forward=None`：

```text
初始化期
  layer0 fully_shard -> layer0 参数变 DTensor shard，形成 group0
  layer1 fully_shard -> layer1 参数变 DTensor shard，形成 group1
  ...
  root fully_shard   -> embedding/lm_head 等剩余参数变 DTensor shard，形成 root group

第一次 model(x)
  root pre-forward
    lazy init:
      找到 root 下所有 FSDP states
      root 自动 reshard_after_forward=False
      所有 states 共享 comm context
      检查 meta 参数、dtype、state_dict hooks

  root group pre-forward
    all-gather embedding/lm_head 等 root group 参数

  layer0 pre-forward
    all-gather layer0 参数
  layer0 compute
  layer0 post-forward
    reshard layer0 参数，释放 full layer0 参数
    记录 post_forward_order

  layer1 pre-forward
    all-gather layer1 参数，可能和 layer0 compute/释放流水 overlap
  layer1 compute
  layer1 post-forward
    reshard layer1 参数
    记录 post_forward_order

  root post-forward
    root group 默认不 reshard，保留完整 root 参数
    给 output 注册 pre-backward hook

backward
  最后一个 layer output grad ready
    layerN pre-backward all-gather 参数
    默认按 post_forward_order 反向 prefetch layerN-1

  layerN grad compute 完
    post_backward:
      收集 full grad
      reshard full param
      reduce-scatter full grad -> sharded grad

  ...

  root final callback
    补齐没触发的 post_backward
    等待 pending reduce-scatter/all-reduce
    清理本轮状态

optimizer.step()
  优化器更新 sharded DTensor 参数
  optimizer state 也是 sharded
```

## 23. 外部资料阅读结论

已参考：

- PyTorch 2.9 FSDP2 官方文档：`https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html`
- PyTorch FSDP2 getting-started tutorial：`https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html`
- FSDP2 RFC / Per-Parameter-Sharding FSDP：`https://github.com/pytorch/pytorch/issues/114299`
- DeepSpeed ZeRO：`https://www.deepspeed.ai/tutorials/zero/`
- DeepSpeed ZeRO++：`https://www.deepspeed.ai/tutorials/zeropp/`
- PyTorch dev-discuss 关于 FSDP 和 CUDA caching allocator / record_stream：`https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486`

旧文档里的飞书链接 `https://aicarrier.feishu.cn/wiki/EYiNwv21ni0HFekr9zrcHEaenSg` 和知乎链接 `https://zhuanlan.zhihu.com/p/1943202817247519535` 在当前环境无法可靠打开，因此本文没有把它们作为事实来源；相关主题均回到 torch 2.9.1 本地源码和 PyTorch 官方/RFC资料核对。
