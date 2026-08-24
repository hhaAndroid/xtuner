# Domino EP overlap 学习笔记

本文围绕 Domino EP 的核心思想学习 XTuner 里的 EP 通信和计算重叠。

Domino EP 的想法很朴素：不用把一个 micro-batch 内部的通信切得很细，而是利用两个相互独立的 micro-batch，让一个 micro-batch 的通信自然 overlap 另一个 micro-batch 的计算。

配套 demo：

```bash
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_basic_demo.py --demo all
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_noop_demo.py --demo all
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_xtuner_style_demo.py
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_xtuner_event_demo.py
```

相关源码：

- Demo 1/2: `workspace/hha_code/domino_ep_basic_demo.py`
- No-op demo: `workspace/hha_code/domino_ep_noop_demo.py`
- XTuner style 简化 hook demo: `workspace/hha_code/domino_ep_xtuner_style_demo.py`
- XTuner v1 event/comm-stream demo: `workspace/hha_code/domino_ep_xtuner_event_demo.py`
- old XTuner no-op 实现: `/mnt/shared-storage-user/huanghaian/code/temp/old_xtuner/xpuyu/xpuyu/modelings/deepseekv3/modeling_deepseek_v3.py`
- XTuner v1 MoE layer: `xtuner/v1/module/decoder_layer/moe_decoder_layer.py`
- XTuner v1 dispatcher: `xtuner/v1/module/dispatcher/torch_all2all.py`

## 1. 先理解要 overlap 什么

一个 MoE EP block 可以简化成：

```text
Pre -> Dispatch all-to-all -> Expert -> Combine all-to-all -> Post
```

其中：

```text
Dispatch:
  token 从原始 rank 发到 expert 所在 rank

Expert:
  每个 rank 计算自己负责的 expert token

Combine:
  expert 输出再发回原始 rank
```

如果完全同步执行，两个 micro-batch 是：

```text
MB0: Pre0 -> Dispatch0 -> Expert0 -> Combine0 -> Post0 然后执行
MB1: Pre1 -> Dispatch1 -> Expert1 -> Combine1 -> Post1
```

所有通信都会阻塞后面的计算。Domino EP 想做的是：

```text
MB0 的通信  overlap  MB1 的计算
MB1 的通信  overlap  MB0 的计算
```

这里的关键不是通信本身变快，而是通信被藏到另一个 micro-batch 的计算时间里。

## 2. demo_1：完全没有 overlap

运行：

```bash
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_basic_demo.py --demo 1
```

对应代码是 `demo_1_no_overlap`：

```python
for x in inputs:
    h = block.pre_compute(x)
    dispatched = a2a_sync(h)
    expert_out = block.expert_compute(dispatched)
    combined = a2a_sync(expert_out)
    losses.append(block.post_compute(combined))
```

这个版本里 `a2a_sync` 的 forward 和 backward 都是同步通信：

```text
forward:
  all_to_all_single(async_op=False)

backward:
  all_to_all_single(async_op=False)
```

所以执行形态是：

```text
MB0: Pre0 -> Dispatch0(wait) -> Expert0 -> Combine0(wait) -> Post0
MB1: Pre1 -> Dispatch1(wait) -> Expert1 -> Combine1(wait) -> Post1
```

这个版本只作为 reference，逻辑最直观，但没有通信计算重叠。

## 3. demo_2：只有 forward overlap

运行：

```bash
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_basic_demo.py --demo 2
```

对应代码是 `demo_2_forward_overlap`。

核心变化是 forward all-to-all 改成：

```python
handle = dist.all_to_all_single(out, x, async_op=True)
```

通信发起后不马上 wait，而是先去做另一个 micro-batch 的计算：

```python
h0 = block.pre_compute(inputs[0])
d0, d0_handle = a2a_forward_async_backward_sync(h0)

h1 = block.pre_compute(inputs[1])
d1, d1_handle = a2a_forward_async_backward_sync(h1)
```

理想 forward 图：

```text
## forward 相同位置表示两个 micro-batch 可以 overlap
#  MB0: Pre0 ── Dispatch0 ── Expert0 ── Combine0 ── Post0
#                    │           │         │           │
#  MB1:          Pre1 ── Dispatch1 ── Expert1 ── Combine1 ── Post1
```

这个版本只解决 forward：

```text
Dispatch0 forward 可以 overlap Pre1
Dispatch1 forward 可以 overlap Expert0
Combine0 forward 可以 overlap Expert1
Combine1 forward 可以 overlap Post0
```

但 backward 还是同步的：

```python
class AttachSyncBackward(torch.autograd.Function):
    def backward(ctx, grad_out):
        dist.all_to_all_single(grad_x, grad_out, async_op=False)
```

所以它适合用来单独理解 forward overlap，不涉及 no-op trick。

## 4. no-op 如何实现 backward overlap

forward overlap 很容易写，因为 forward 代码就是我们手写的执行顺序：

```text
launch async a2a
做另一个 micro-batch 的 compute
wait a2a
```

backward overlap 难在：我们通常不会手写 backward 执行顺序，而是只调用：

```python
loss.backward()
```

后面的 backward 节点由 PyTorch autograd 按 forward graph 的反向拓扑顺序执行。

所以 no-op 版本的核心问题是：

```text
如果不能直接手写 backward 顺序，怎么控制 backward 里某个通信在哪里发起、在哪里等待？
```

答案是：在 forward graph 里提前埋两个 autograd node。

```text
NoOpWait:
  forward 什么都不做
  backward 等待某个 async comm handle

LaunchBackwardA2A:
  forward 什么都不做，只返回 extra_input
  backward 发起 reverse async all-to-all，并把 handle 存起来
```

这两个节点在 forward 看起来都很奇怪，因为它们几乎不改变 forward 数值。但它们真正有用的地方在 backward。

### 4.1 一个通信对应两个 backward 控制点

以 `dispatch0` 为例，forward 通信本身是：

```text
h0 --Dispatch0 forward all-to-all--> d0
```

为了让 `dispatch0` 的 backward 也异步执行，no-op 版本会在 forward graph 里为它埋两个点：

```text
NoOp_D0 ............................. Launch_D0
   │                                      │
   │ forward 什么都不做                    │ forward 什么都不做
   │ backward wait dispatch0_bwd          │ backward launch dispatch0_bwd
```

注意 forward 和 backward 的顺序是反过来的：

```text
forward:
  NoOp_D0 -> ... -> Launch_D0

backward:
  Launch_D0 -> ... -> NoOp_D0
```

所以这两个点的职责刚好变成：

```text
Launch_D0:
  backward 中先发起 async all-to-all

NoOp_D0:
  backward 中更晚才 wait 这个 all-to-all
```

中间的 `...` 就是可以拿来 overlap 的 backward compute。

### 4.2 为什么它能 overlap

假设 forward 中有这样一段：

```text
NoOp_D0 -> Dispatch0 -> Pre1 -> Launch_D0
```

那么 backward 大致反过来：

```text
Launch_D0 -> Pre1 -> Dispatch0 -> NoOp_D0
```

在 NoOp_D0  和 Launch_D0  中的计算模块 Pre1 通过这种固定范式就可以自动实现 forward 和 backward overlap。理解了这个核心就基本上懂了 no op 实现过程。剩下的就是如何编排 launch 顺序就行。



其中：

```text
Launch_D0:
  发起 dispatch0 的 reverse all-to-all，async_op=True

Pre1:
  做 Pre1 的 backward compute

NoOp_D0:
  wait dispatch0 reverse all-to-all 完成
```

理想执行图是：

```text
CPU/autograd launch:
  Launch_D0 -> Pre1 -> ... -> NoOp_D0

CUDA compute stream:
              Pre1 backward compute ─────────────

CUDA comm stream:
  Dispatch0 backward a2a ─────────────────
```

这里最关键的是：

```text
Launch_D0 只负责发起通信，不等待通信。
NoOp_D0 才负责等待通信。
```

只要 `Launch_D0` 和 `NoOp_D0` 之间隔着足够多的 backward compute，通信就有机会被藏到这些 compute 下面。

### 4.3 handle_table 和 key 的作用

`LaunchBackwardA2A` 和 `NoOpWait` 是两个不同的 autograd node。它们需要在 backward 时找到同一个通信 handle。

demo 里用的是：

```python
handles: dict[str, AsyncComm] = {}
```

再用 key 把 launch 和 wait 配对：

```text
key = "dispatch_mb0"

LaunchBackwardA2A.backward:
  handles["dispatch_mb0"] = AsyncComm(handle=...)

NoOpWait.backward:
  handle = handles.pop("dispatch_mb0")
  handle.wait()
```

所以 key 非常重要。它表达的是：

```text
这个 NoOpWait 等待哪一个 LaunchBackwardA2A 发起的通信
```

如果 key 对不上，wait 就找不到正确的 handle；如果插入点不对，通信虽然能发起，但 overlap 窗口就不对。

### 4.4 为什么叫 no-op

它叫 no-op，不是因为 backward 没事做，而是因为 forward 数值上近似什么都不做。

例如：

```python
h0 = no_op_wait(h0, handles, "dispatch_mb0")
```

forward 返回的还是 `h0`，所以对 forward 计算结果没有影响。

但 autograd graph 里多了一个节点：

```text
h0 -> NoOpWait -> 后续计算
```

到了 backward，这个节点就会执行：

```python
handle = handles.pop("dispatch_mb0")
handle.wait()
```

同理：

```python
_, d0 = a2a_fwd_bwd_overlap(
    h0,
    handles,
    "dispatch_mb0",
    is_forward=False,
    extra_input=d0,
)
```

forward 返回的还是 `d0`，但 backward 会发起 `dispatch_mb0` 的 reverse all-to-all。

所以 no-op 的真实含义是：

```text
forward 数值 no-op
backward 侧-effect：launch 或 wait async communication
```

### 4.5 为什么“只写 forward 插入点”就能控制 backward

PyTorch autograd 的基本规律是：

```text
forward:
  A -> B -> C -> D

backward:
  D -> C -> B -> A
```

所以如果希望某个 backward 通信更早 launch，就把它对应的 `LaunchBackwardA2A` 在 forward 里放得更靠后。

如果希望某个 backward 通信更晚 wait，就把它对应的 `NoOpWait` 在 forward 里放得更靠前。

这就是 Domino / old XTuner no-op trick 最核心的地方：

```text
forward graph 决定 backward 顺序
no-op node 在 forward 中占位置
这些位置反过来变成 backward 中的通信 launch/wait 位置
```

一旦理解这一点，后面的代码主要就是在看：

```text
每个通信的 LaunchBackwardA2A 插在哪里
每个通信的 NoOpWait 插在哪里
这两个点之间隔着哪些 backward compute
```

## 5. no-op 版本的三个核心组件

运行：

```bash
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_noop_demo.py --demo all
```

### 5.1 AsyncComm

```python
@dataclass
class AsyncComm:
    handle: dist.Work
    keep_alive: tuple[Any, ...]

    def wait(self) -> None:
        self.handle.wait()
        self.keep_alive = ()
```

这里保存两个东西：

```text
handle:
  NCCL async all-to-all 返回的 dist.Work，用来 wait。

keep_alive:
  保持输入和输出 tensor 的 Python 引用，避免 async comm 还没结束时 tensor 生命周期结束。
```

基础 forward-only demo 里为了简单可以直接返回 handle；no-op demo 里因为 backward async 通信也要跨 autograd node 保存，所以保留 `AsyncComm` 更接近 old XTuner。

### 5.2 LaunchBackwardA2A

```python
class LaunchBackwardA2A(torch.autograd.Function):
    def forward(ctx, x_for_grad, handle_table, key, extra_input):
        ctx.input_shape = tuple(x_for_grad.shape)
        ctx.handle_table = handle_table
        ctx.key = key
        return extra_input

    def backward(ctx, grad_out):
        grad_x = torch.empty(ctx.input_shape, device=grad_out.device, dtype=grad_out.dtype)
        handle = dist.all_to_all_single(grad_x, grad_out, async_op=True)
        ctx.handle_table[ctx.key] = AsyncComm(handle=handle, keep_alive=(grad_out, grad_x))
        return grad_x, None, None, None
```

这个节点的语义是：

```text
forward:
  不发通信，直接返回 extra_input

backward:
  发起 reverse all-to-all
  把 async handle 存到 handle_table[key]
```

它对应 old XTuner 里的 `_AllToAll_BWD`。

### 5.3 NoOpWait

```python
class NoOpWait(torch.autograd.Function):
    def forward(ctx, x, handle_table, key):
        ctx.handle_table = handle_table
        ctx.key = key
        return x

    def backward(ctx, grad_out):
        handle = ctx.handle_table.pop(ctx.key)
        handle.wait()
        return grad_out, None, None
```

这个节点的语义是：

```text
forward:
  什么都不做，只保存 handle_table 和 key

backward:
  根据 key 找到之前 LaunchBackwardA2A 发起的 async comm
  等待通信结束
```

所以 no-op trick 的本质是：

```text
LaunchBackwardA2A 控制 backward 中何时 launch 通信
NoOpWait 控制 backward 中何时 wait 通信
```

## 6. no-op 版本 A：c1_e1_c0

运行：

```bash
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_noop_demo.py --demo c1_e1_c0
```

这个版本的目标 backward 顺序是：

```text
combine1 -> expert1 -> combine0
```

也就是让 `Combine0` 的 backward launch 点更晚一些，放到 `Expert1` 后面执行。

代码里对应函数是：

```python
demo_noop_c1_e1_c0(...)
```

forward 图：

```text
## forward 相同位置表示两个 micro-batch 可以 overlap
#  MB0: Pre0 ── Dispatch0 ── Expert0 ── Combine0 ── Post0
#                    │           │         │           │
#  MB1:          Pre1 ── Dispatch1 ─────── Expert1 ── Combine1 ── Post1
```

backward 图：

```text
## backward 相同位置表示两个 micro-batch 可以 overlap
#   MB0:        Post0 ─────── Combine0 ─────── Expert0 ── Dispatch0 ── Pre0
#                   │            │             │             │
#   MB1: Post1 ── Combine1 ── Expert1 ── Dispatch1 ────── Pre1
```

这张图里最关键的是：

```text
Post0 和 Combine1 可以 overlap
Combine0 和 Expert1 可以 overlap
Expert0 和 Dispatch1 可以 overlap
Dispatch0 和 Pre1 可以 overlap
```

这里说的 overlap 不是 CPU autograd 节点同时执行。CPU 上 autograd 仍然按顺序 launch backward 节点。

真正的 overlap 来自：

```text
compute kernel enqueue 到 compute stream
all-to-all enqueue 到 communication stream
CPU 继续向后 launch
CUDA 上两个 stream 可以并行执行
```

所以即使 CPU launch 顺序是：

```text
Post0 backward launch
Combine1 backward launch async all-to-all
```

只要 `Post0` 的 compute kernel 还在跑，`Combine1` 的通信就可以在通信流上和它 overlap。

## 7. no-op 版本 B：c1_c0_e1

运行：

```bash
torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_noop_demo.py --demo c1_c0_e1
```

这个版本的目标 backward 顺序是：

```text
combine1 -> combine0 -> expert1
```

也就是把 `Combine0` 的 backward launch 点提前，让两个 reverse combine 更靠近。

代码里对应函数是：

```python
demo_noop_c1_c0_e1(...)
```

forward 图：

```text
## forward 相同位置表示两个 micro-batch 可以 overlap
#
#  MB0: Pre0 ── Dispatch0 ── Expert0 ── Combine0 ─── Post0
#                    │           │         │           │
#  MB1:          Pre1 ── Dispatch1 ── Expert1 ── Combine1 ── Post1
```

backward 图：

```text
## backward 相同位置表示两个 micro-batch 可以 overlap
#   MB0:        Post0 ── Combine0 ───────── Expert0 ── Dispatch0 ── Pre0
#                   │         │                │             │
#   MB1: Post1 ── Combine1 ── Expert1 ── Dispatch1 ────── Pre1
```

和版本 A 的差异只在 `Combine0` backward launch 点：

```text
c1_e1_c0:
  Combine1 -> Expert1 -> Combine0

c1_c0_e1:
  Combine1 -> Combine0 -> Expert1
```

这个 demo 想表达的是：no-op 插入点不是固定的。只要数值依赖正确，forward graph 里插在哪里，backward 中通信就会按反向位置发起。

## 8. 为什么结果仍然是对的

no-op demo 不只看 loss，也会比较梯度。

同步 reference：

```python
demo_reference_sync(...)
```

no-op 版本：

```python
demo_noop_c1_e1_c0(...)
demo_noop_c1_c0_e1(...)
```

最终比较：

```python
loss_diff = (result.loss - ref.loss).abs().item()
grad_max_diff = (result.grads - ref.grads).abs().max().item()
grad_l2_diff = torch.linalg.vector_norm(result.grads - ref.grads).item()
```

如果 no-op 插入点写错，最容易出问题的是：

```text
loss 可能看起来还差不多
但参数梯度或输入梯度不一致
```

所以 demo 用同步版本作为 reference，同时验证：

```text
loss 一致
所有参数梯度一致
两个 input tensor 的梯度一致
```

这是学习 no-op trick 时非常重要的一点：看起来能 overlap 不够，还必须证明 autograd 依赖没有被破坏。

## 9. no-op trick 的关键心智模型

可以把整个机制压缩成四句话：

```text
1. forward async all-to-all 负责 forward overlap。
2. LaunchBackwardA2A 负责在 backward 中发起 reverse async all-to-all。
3. NoOpWait 负责在 backward 中等待 reverse async all-to-all。
4. 改变 forward graph 里的插入点，就能改变 backward 中 launch/wait 的位置。
```

因此 no-op 版本最难理解的不是 all-to-all 本身，而是：

```text
forward 里看到的是 no-op
backward 里它们变成了通信 launch / wait 节点
```

forward graph 和 backward 执行顺序的关系是：

```text
forward:
  A -> B -> C -> D

backward:
  D -> C -> B -> A
```

所以如果希望 backward 中某个通信更早发起，就要把它对应的 forward node 放得更靠后。

如果希望 backward 中某个通信更晚等待，就要把它对应的 no-op wait forward node 放得更靠前。

## 10. XTuner event/comm-stream 版本

当前学习顺序建议先固定为：

```text
1. demo_1_no_overlap
2. demo_2_forward_overlap
3. demo_noop_c1_e1_c0
4. demo_noop_c1_c0_e1
5. domino_ep_xtuner_style_demo
6. domino_ep_xtuner_event_demo
7. XTuner v1 真实代码
```

前四步解决的是 old XTuner / no-op 思路：

```text
通过 forward graph 里的 no-op autograd node 控制 backward overlap
```

`domino_ep_xtuner_style_demo.py` 是更简单的 hook/token 版本，用来理解：

```text
不额外插 NoOp Function，也可以把 wait 点挂到已有 grad_fn 上
```

这个 demo 里的 `hook_handle` 只是保存 `register_prehook` 返回的 removable handle：

```python
token.hook_handle = x.grad_fn.register_prehook(wait_before_this_node)
```

hook 是否生效不依赖保存这个 handle。注册完成后 hook 已经挂到 `x.grad_fn` 上了；保存 handle 只是为了以后可以手动：

```python
token.hook_handle.remove()
```

当前 demo 不会 remove hook，所以 `hook_handle` 不参与 overlap 逻辑。真实 XTuner v1 里也是直接 `register_hook` / `register_prehook`，不需要额外保存 handle。

`x.grad_fn.register_prehook(fn)` 的调用时机是：

```text
backward 执行到 x.grad_fn 这个 autograd node 时
先调用 prehook
再执行 x.grad_fn 自己的 backward 计算
```

可以理解成：

```text
下游 backward 已经把 grad_outputs 传到了 x.grad_fn
x.grad_fn 准备继续向上游计算 grad_inputs
prehook 在 grad_inputs 计算前被调用
```

所以它适合做 wait：

```python
def wait_before_this_node(grad_outputs):
    token.wait()
```

语义是：

```text
在这个节点真正使用 grad_outputs 往前算梯度前，
先等待通信流上的 reverse all-to-all 完成。
```

### 10.1 event 版本和 no-op 版本的对应关系

`domino_ep_xtuner_event_demo.py` 更接近当前 XTuner v1，用来理解：

```text
通信流 comm_stream
forward_finished_event
backward_previous_event
backward_finished_event
hook / prehook 如何用 event 控制 backward overlap
```

event 版本和 no-op 版本解决的是同一个问题：

```text
backward 中通信什么时候可以 launch？
backward 中什么时候必须 wait 通信完成？
通信如何和中间 compute overlap？
```

但实现方式不一样：

```text
old no-op:
  LaunchBackwardA2A.backward 发起通信
  NoOpWait.backward 等待通信 handle
  用 handle_table[key] 把 launch 和 wait 配对

XTuner event:
  AsyncA2A.backward 在 comm_stream 上发起通信
  register_hook 记录 grad_output ready event
  register_prehook 等待 backward_finished_event
  用 CUDA event 把 compute stream 和 comm stream 串起来
```

可以粗略对应成：

```text
LaunchBackwardA2A.backward  ~=  AsyncA2AWithEvents.backward / _AsyncDispatch.backward
NoOpWait.backward           ~=  grad_fn.register_prehook(wait_event)
handle_table[key]           ~=  A2AEvents / CUDA event 字段
async handle.wait()         ~=  current_stream.wait_event(...)
```

所以 event 版本不再显式保存 `dist.Work` handle。它把同步关系交给 CUDA event：

```text
comm stream 上 record event
compute stream 上 wait event
compute stream 上 record event
comm stream 上 wait event
```

### 10.2 四个 event 分别表示什么

demo 里每个 all-to-all 都有一组 `A2AEvents`：

```python
@dataclass
class A2AEvents:
    name: str
    forward_previous_event: torch.cuda.Event
    forward_finished_event: torch.cuda.Event
    backward_previous_event: torch.cuda.Event
    backward_finished_event: torch.cuda.Event
```

这四个 event 可以按 forward / backward 分开理解。

forward 侧：

```text
forward_previous_event:
  记录输入 x 在 compute stream 上已经 ready。
  comm stream 必须等它，才能开始 forward all-to-all。

forward_finished_event:
  记录 forward all-to-all 在 comm stream 上已经完成。
  compute stream 必须等它，才能使用 all-to-all 输出。
```

backward 侧：

```text
backward_previous_event:
  记录 grad_output 在 compute stream 上已经 ready。
  comm stream 必须等它，才能开始 reverse all-to-all。

backward_finished_event:
  记录 reverse all-to-all 在 comm stream 上已经完成。
  compute stream 必须等它，才能继续上游 backward compute。
```

把这四个 event 画成一条 all-to-all 的生命周期：

```text
forward:
  compute stream:  produce x ── record forward_previous_event
                                         │
  comm stream:                wait ── forward a2a ── record forward_finished_event
                                                               │
  compute stream:                                      wait ── consume y

backward:
  compute stream:  produce grad_y ── record backward_previous_event
                                             │
  comm stream:                  wait ── backward a2a ── record backward_finished_event
                                                                 │
  compute stream:                                        wait ── consume grad_x
```

这就是 event 版本的完整同步闭环。

### 10.3 forward：通信流负责 all-to-all，计算流只等 event

demo 里的 forward 通信在 `AsyncA2AWithEvents.forward`：

```python
with torch.cuda.stream(comm_stream):
    comm_stream.wait_event(events.forward_previous_event)
    dist.all_to_all_single(out, x, async_op=False)
    x.record_stream(comm_stream)
    out.record_stream(comm_stream)
    events.forward_finished_event.record(comm_stream)
```

这里的关键是：

```text
all-to-all 被放到 comm_stream 上
comm_stream 先等 forward_previous_event，确保输入 x 已经 ready
通信完成后 record forward_finished_event
```

计算流使用通信输出前，只做：

```python
wait_forward_comm(d0_events)
e0 = block.expert_compute(d0)
```

其中：

```python
def wait_forward_comm(events):
    torch.cuda.current_stream().wait_event(events.forward_finished_event)
```

所以 forward overlap 的形态是：

```text
compute stream:  Pre0 ───────────── Pre1 ───────────── wait D0 ── Expert0
                    │                 │
comm stream:        wait Pre0 ── Dispatch0 ─────────── record D0 done
                                      wait Pre1 ── Dispatch1 ───────────
```

也就是：

```text
Dispatch0 在 comm stream 上跑
Pre1 在 compute stream 上跑
两个 stream 没有互相等待，所以可以 overlap
```

### 10.4 backward：hook 记录 ready，prehook 等待完成

backward 侧有两个关键函数。

第一个是 `record_backward_ready`：

```python
def record_backward_ready(x, events):
    def record_event(_grad):
        events.backward_previous_event.record(torch.cuda.current_stream())
        return None

    x.register_hook(record_event)
    return x
```

它的含义是：

```text
x 的 grad 已经被下游 backward 算出来了
在当前 compute stream 上 record backward_previous_event
通知 comm stream：reverse all-to-all 的输入 grad_output ready 了
```

第二个是 `make_a2a_events` 里的 prehook：

```python
def wait_before_upstream_backward(_grad_outputs):
    torch.cuda.current_stream().wait_event(events.backward_finished_event)
    return None

x.grad_fn.register_prehook(wait_before_upstream_backward)
```

它的含义是：

```text
backward 准备执行 x.grad_fn 自己的 backward 前
先等待 reverse all-to-all 完成
确保 grad_x 已经 ready
```

中间真正发起 reverse all-to-all 的地方在 `AsyncA2AWithEvents.backward`：

```python
with torch.cuda.stream(ctx.comm_stream):
    ctx.comm_stream.wait_event(ctx.events.backward_previous_event)
    dist.all_to_all_single(grad_x, grad_out, async_op=False)
    grad_out.record_stream(ctx.comm_stream)
    grad_x.record_stream(ctx.comm_stream)
    ctx.events.backward_finished_event.record(ctx.comm_stream)
```

所以 backward 的顺序是：

```text
1. 下游 compute backward 产生 grad_out
2. tensor hook record backward_previous_event
3. AsyncA2A.backward 在 comm_stream 上 wait backward_previous_event
4. comm_stream 发起 reverse all-to-all
5. comm_stream record backward_finished_event
6. 上游 grad_fn prehook wait backward_finished_event
7. 上游 compute backward 继续执行
```

画成流图：

```text
compute stream:  downstream backward ── record backward_previous_event ── other compute ── wait backward_finished_event ── upstream backward
                                      │                                      ▲
comm stream:                          wait ── reverse all-to-all ── record ─┘
```

这就是 event 版的 backward overlap。

### 10.5 为什么 prehook 等价于 no-op wait

no-op 版本里，等待点是一个显式 autograd node：

```text
NoOpWait.forward:
  return x

NoOpWait.backward:
  handle.wait()
  return grad_out
```

event 版本里，不再创建这个 `NoOpWait` 节点，而是把等待动作挂到已有 node 前面：

```python
x.grad_fn.register_prehook(wait_before_upstream_backward)
```

调用时机是：

```text
准备执行 x.grad_fn.backward 前
先执行 wait_before_upstream_backward
```

所以效果等价于：

```text
在 upstream backward compute 真正开始前 wait comm done
```

区别只是：

```text
no-op:
  用额外 autograd Function 占一个 graph 位置

event:
  不增加新 Function，把 wait 挂到已有 grad_fn 上
```

### 10.6 为什么 register_hook 等价于告诉通信可以 launch

反向通信的输入是 `grad_out`。它不能太早 launch，因为 `grad_out` 还没算出来。

所以 event 版本要在 `grad_out` ready 的地方 record event：

```python
x.register_hook(record_event)
```

这个 hook 的意义不是等待，而是发信号：

```text
这个 tensor 的 grad 已经 ready
comm stream 可以开始用它做 reverse all-to-all
```

对应到真实 XTuner v1：

```python
global_input_tokens.grad_fn.register_hook(
    get_backward_hook(dispatched["backward_previous_event"], ...)
)
```

`get_backward_hook` 做的事情就是：

```python
backward_finished_event.record()
```

虽然名字叫 `backward_finished_event`，但在 `dispatch_postprocess` 这里传进去的是 `dispatched["backward_previous_event"]`。它表达的是：

```text
dispatch_postprocess backward 已经结束
dispatch backward 的 grad_output ready
可以让 _AsyncDispatch.backward 开始等这个 event 后发通信
```

### 10.7 2 micro-batch 的 event 版流程图

forward 理想图仍然是：

```text
## forward 相同位置表示两个 micro-batch 可以 overlap
#  MB0: Pre0 ── Dispatch0 ── Expert0 ── Combine0 ── Post0
#                    │           │         │           │
#  MB1:          Pre1 ── Dispatch1 ── Expert1 ── Combine1 ── Post1
```

event 版本里，这张图背后的实际 stream/event 是：

```text
compute stream:
  Pre0 ── record D0.forward_previous
       ── Pre1 ── record D1.forward_previous
       ── wait D0.forward_finished ── Expert0 ── record C0.forward_previous
       ── wait D1.forward_finished ── Expert1 ── record C1.forward_previous
       ── wait C0.forward_finished ── Post0
       ── wait C1.forward_finished ── Post1

comm stream:
  wait D0.forward_previous ── Dispatch0 ── record D0.forward_finished
  wait D1.forward_previous ── Dispatch1 ── record D1.forward_finished
  wait C0.forward_previous ── Combine0 ── record C0.forward_finished
  wait C1.forward_previous ── Combine1 ── record C1.forward_finished
```

backward 理想图可以先按 no-op 版本理解：

```text
## backward 相同位置表示两个 micro-batch 可以 overlap
#   MB0:        Post0 ─────── Combine0 ─────── Expert0 ── Dispatch0 ── Pre0
#                   │            │             │             │
#   MB1: Post1 ── Combine1 ── Expert1 ── Dispatch1 ────── Pre1
```

event 版本里，通信不是通过 `handle.wait()` 控制，而是：

```text
compute stream:
  Post1/Post0 backward
  record C1.backward_previous
  other backward compute
  wait C1.backward_finished before Expert1 backward consumes grad

comm stream:
  wait C1.backward_previous
  Combine1 reverse all-to-all
  record C1.backward_finished
```

所以看 event 版本时，不要只找“哪里 wait”。要同时找：

```text
谁 record backward_previous_event？
谁 wait backward_previous_event 后发通信？
谁 record backward_finished_event？
谁 wait backward_finished_event 后继续 compute？
```

这四个问题回答清楚，一个通信的 backward overlap 就清楚了。

### 10.8 demo 和真实 XTuner v1 的对应关系

demo 文件：

```text
workspace/hha_code/domino_ep_xtuner_event_demo.py
```

真实源码：

```text
xtuner/v1/module/dispatcher/torch_all2all.py
xtuner/v1/module/decoder_layer/moe_decoder_layer.py
```

对应关系：

```text
demo: A2AEvents
real: TorchAll2AllPreDispatchResult / DispatchResult / PreCombineResult / CombineResult 里的 event 字段

demo: AsyncA2AWithEvents.forward/backward
real: _AsyncDispatch.forward/backward, _AsyncCombine.forward/backward

demo: record_backward_ready
real: get_backward_hook + register_hook

demo: make_a2a_events 里的 register_prehook
real: get_backward_pre_hook + register_prehook

demo: wait_forward_comm
real: wait_comm_stream
```

真实代码里 dispatch 的拆分更细：

```text
dispatch_preprocess:
  做 permute
  record forward_finished_event
  register_prehook 等 dispatch backward 完成

dispatch:
  comm_stream 等 preprocess forward event
  发 forward dispatch all-to-all
  backward 中发 reverse dispatch all-to-all

dispatch_postprocess:
  wait dispatch forward event
  做 postprocess permute
  register_hook，backward 后 record event，通知 dispatch backward 可以开始
```

combine 也是同样结构：

```text
combine_preprocess:
  做 unpermute
  record forward_finished_event
  register_prehook 等 combine backward 完成

combine:
  comm_stream 等 combine_preprocess forward event
  发 forward combine all-to-all
  backward 中发 reverse combine all-to-all

combine_postprocess:
  wait combine forward event
  做 final unpermute
  register_hook，backward 后 record event，通知 combine backward 可以开始
```

### 10.9 阅读真实源码的顺序

建议先按这个顺序看：

```text
1. moe_decoder_layer.py::_micro_batch_forward
   先只看 micro-batch 的 forward 排布。

2. torch_all2all.py::_AsyncDispatch
   看 forward / backward 如何都在 comm_stream 上跑。

3. dispatch_preprocess
   看 forward_previous_event 怎么创建，prehook 怎么注册。

4. dispatch_postprocess
   看 forward_finished_event 怎么 wait，backward hook 怎么 record event。

5. _AsyncCombine / combine_preprocess / combine_postprocess
   按 dispatch 的方式再看一遍 combine。
```

这部分不要一开始就陷入 `tokens_per_expert`、`row_id_map`、`input_splits`、`output_splits`。那些是 all-to-all 的数据排布细节。

学习 overlap 时先只抓住：

```text
哪个 tensor ready 后 record event？
哪个 stream wait 这个 event？
哪个通信完成后 record event？
哪个 backward node prehook wait 这个完成 event？
```

这四件事就是当前 XTuner v1 event/comm-stream 写法的主线。
