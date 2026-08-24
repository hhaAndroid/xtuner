# PyTorch Activation Offload 学习笔记

本文围绕 XTuner 里的 `async_save_on_cpu` 学习 activation offload。

配套 demo：

```bash
python workspace/hha_code/activation_offload_demo.py --demo 1
python workspace/hha_code/activation_offload_demo.py --demo 5
```

相关源码：

- Demo: `workspace/hha_code/activation_offload_demo.py`
- XTuner 核心实现: `xtuner/v1/utils/activation_offload.py`
- XTuner MoE 调用: `xtuner/v1/model/moe/moe.py`

## 1. 先理解 activation 为什么占显存

训练时 GPU 显存大致包括：

```text
参数 parameters
梯度 gradients
优化器状态 optimizer states
activation / intermediate tensors
临时 tensor / workspace
```

activation 是 forward 中间结果。它们之所以不能随便释放，是因为 backward 需要它们计算梯度。

一个简化流程：

```text
forward:
  x -> block0 -> block1 -> block2 -> loss
       ^         ^         ^
       |         |         |
       backward 需要这些 saved tensors

backward:
  loss -> block2 backward -> block1 backward -> block0 backward
```

如果所有 saved activation 都留在 GPU 上，长序列、大 batch、深层模型时显存会很高。

activation offload 的核心思路是：

```text
forward 保存 activation 时:
  GPU activation -> CPU
  释放 GPU storage

backward 真要用 activation 时:
  CPU activation -> GPU
  继续 backward
```

收益是降低 forward 后到 backward 前这一段的 GPU activation 占用。代价是多了 CPU-GPU 拷贝。

## 2. activation offload 和 checkpointing 的区别

这两个技术都能省显存，但思路不同：

```text
activation checkpointing:
  forward 时不保存部分 activation
  backward 时重新跑一遍 forward 计算回来
  省显存，增加计算

activation offload:
  forward 时保存 activation，但保存到 CPU
  backward 时从 CPU 拷回 GPU
  省显存，增加 PCIe/NVLink 拷贝
```

offload 是否划算，取决于：

- activation 有多大。
- CPU-GPU 带宽有多高。
- D2H/H2D 能否和计算重叠。
- backward 取回 activation 时会不会频繁等待。

## 3. PyTorch 最小机制：saved_tensors_hooks

PyTorch autograd 里，某些 op 的 backward 需要 forward 的输入或输出。以自定义 autograd function 为例：

```python
ctx.save_for_backward(x)
```

这句话不是“立刻复制一份 x”，更准确地说是：

```text
登记 x 会被 backward 用到
```

如果外层启用了：

```python
from torch.autograd.graph import saved_tensors_hooks
```

那么 PyTorch 会在保存 saved tensor 时调用 `pack`，在 backward 读取 saved tensor 时调用 `unpack`。

最小形式：

```python
def pack(t):
    return t

def unpack(t):
    return t

with saved_tensors_hooks(pack, unpack):
    y = some_forward(x)

y.backward()
```

语义是：

```text
forward 阶段:
  PyTorch 准备保存 tensor t
  调用 pack(t)
  autograd graph 保存 pack(t) 的返回值

backward 阶段:
  PyTorch 需要这个 saved tensor
  调用 unpack(saved_payload)
  backward 使用 unpack(...) 返回的 tensor
```

所以 offload 可以这么做：

```text
pack:
  CUDA tensor -> CPU payload
  return CPU payload

unpack:
  CPU payload -> CUDA tensor
  return CUDA tensor
```

但是要注意：`saved_tensors_hooks` 只决定 autograd graph 保存什么，不会自动释放原 CUDA tensor 的 storage。真正释放 GPU 显存还需要去掉引用，或者像 XTuner 一样手动把原 tensor 的 storage 缩到 0。

## 4. 关于 PyTorch 显存统计

demo 里打印了：

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
```

含义：

```text
allocated:
  当前活 tensor 实际占用的 GPU 显存

reserved:
  PyTorch caching allocator 从 CUDA 申请到的内存池
  reserved 通常 >= allocated

peak:
  历史最高 allocated
```

`torch.cuda.empty_cache()` 只释放 PyTorch 缓存池里“已经空闲”的块。它不会释放还被活 tensor 引用的 storage。

所以：

```python
x = torch.randn(..., device="cuda")
torch.cuda.empty_cache()
```

`allocated` 不会下降，因为 `x` 还活着。

如果：

```python
x = x.cpu()
torch.cuda.empty_cache()
```

旧 CUDA tensor 如果没有其他引用，就可以释放，`allocated` 会下降。

但训练计算图中，autograd graph 可能仍然持有 tensor 或中间结果，所以不能简单认为 `x = x.cpu()` 一定能释放所有相关显存。

## 5. demo_01：普通 saved tensor

代码入口：

```python
python workspace/hha_code/activation_offload_demo.py --demo 1
```

`make_hidden` 里：

```python
x = torch.randn(n, n, device="cuda", requires_grad=True)
return x * 1.0
```

当 `n=2048` 时，一个 float32 tensor 大约：

```text
2048 * 2048 * 4 bytes = 16 MB
```

所以刚创建 hidden 后是 32 MB：

```text
x leaf tensor        16 MB
hidden = x * 1.0     16 MB
合计                 32 MB
```

forward 后是 48 MB：

```text
x leaf tensor                 16 MB
hidden                        16 MB
out = hidden * hidden         16 MB
合计                          48 MB
```

`ctx.save_for_backward(hidden)` 不会复制一份 16 MB tensor，它只是让 autograd graph 持有这个 tensor。显存增加主要来自 `out`。

backward 后常见是 64 MB：

```text
x leaf tensor                 16 MB
hidden                        16 MB
out                           16 MB
x.grad                        16 MB
合计                          64 MB
```

峰值可能到 80 MB，是因为 backward 中还有临时梯度 tensor。

## 6. demo_02：观察 pack/unpack 时机

运行：

```bash
python workspace/hha_code/activation_offload_demo.py --demo 2
```

典型输出：

```text
[forward:observe_hooks] called ctx.save_for_backward(x)
[forward:observe_hooks] custom Function forward is about to return
[hook:pack] PyTorch wants to save tensor
[main] forward context exited; backward starts
[hook:unpack] PyTorch needs saved tensor for backward
[backward:observe_hooks] loaded saved x
```

关键点：

```text
ctx.save_for_backward(x)
  只是登记 x 会被 backward 用到

forward 返回后
  PyTorch 调用 pack(x)

backward 需要 x 时
  PyTorch 调用 unpack(payload)
```

所以不要把 `ctx.save_for_backward(x)` 理解成“已经把 tensor 保存好了”。最终保存什么，由 `pack` 的返回值决定。

## 7. demo_03：同步 CPU offload，但显存不下降

运行：

```bash
python workspace/hha_code/activation_offload_demo.py --demo 3
```

核心：

```python
def pack(t):
    cpu_tensor = t.detach().cpu()
    return cpu_tensor, t.device

def unpack(payload):
    cpu_tensor, device = payload
    return cpu_tensor.to(device)
```

这确实让 autograd graph 保存了 CPU payload。

但是 forward 后 GPU 显存仍然可能是 48 MB：

```text
x leaf tensor                 16 MB
hidden                        16 MB
out                           16 MB
合计                          48 MB
```

原因是：

```text
pack -> t.cpu()
  只是复制一份 CPU tensor
  不会销毁原 CUDA tensor

原 CUDA hidden 仍然被 Python 变量 / 计算图引用
所以 allocated 不下降
```

`empty_cache()` 也没用，因为这些 tensor 不是缓存空闲块，而是 live tensor。

这也是为什么 XTuner 不能只写一个简单的 `pack(t): return t.cpu()`。

## 8. demo_04：异步 D2H，但仍然不释放 GPU storage

运行：

```bash
python workspace/hha_code/activation_offload_demo.py --demo 4
```

demo_04 引入了三件事：

```text
pinned CPU memory
side CUDA stream
CUDA event
```

核心流程：

```python
cpu_tensor = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
d2h_done = torch.cuda.Event()

d2h_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(d2h_stream):
    cpu_tensor.copy_(t.detach(), non_blocking=True)
    d2h_done.record(d2h_stream)
```

含义：

```text
1. 当前 stream 负责 forward 计算 t
2. d2h_stream 等当前 stream 把 t 算完
3. d2h_stream 异步执行 GPU -> CPU copy
4. copy 结束后记录 event
```

backward 中：

```python
torch.cuda.current_stream().wait_event(payload.d2h_done)
return payload.cpu_tensor.to(payload.original_device, non_blocking=True)
```

含义：

```text
backward 真要读 saved tensor 时
先等 D2H 完成
再把 CPU payload 拷回 GPU
```

但是 demo_04 仍然不会让 forward 后 `allocated` 下降，因为它仍然没有释放原 CUDA tensor 的 storage。

demo_04 解决的是：

```text
怎么异步 D2H
怎么用 event 保证正确性
```

它还没有解决：

```text
怎么释放原 GPU storage
```

这个问题放在 demo_05。

## 9. demo_05：接近 XTuner 的 release + restore

运行：

```bash
python workspace/hha_code/activation_offload_demo.py --demo 5
```

demo_05 多了一个 `SimpleSwapTensor`，它类似 XTuner 的 `SwapTensor`。

它保存：

```text
self.tensor:
  原来的 CUDA tensor 对象

self.cpu_tensor:
  D2H 后保存 activation 内容的 CPU pinned buffer

self.storage_size:
  原 CUDA storage 大小

self.d2h_done:
  D2H 完成事件
```

### 9.1 为什么不是简单新建 CUDA tensor

demo_04 的 backward 是：

```python
return cpu_tensor.to(cuda)
```

这会新建一个 CUDA tensor。

demo_05 是：

```python
self.tensor.storage().resize_(self.storage_size)
self.tensor.copy_(self.cpu_tensor)
return self.tensor
```

它恢复的是原 tensor 对象背后的 storage。

工程上这样更接近 saved tensor 的语义，也方便做生命周期管理：

```text
forward 后:
  原 tensor 对象仍在
  storage 被 resize 到 0

backward 前:
  同一个 tensor 对象恢复 storage
  CPU 内容 copy 回来
```

简单新建 tensor 在教学上能跑，但可能不保留 view/storage/alias 等关系，也更难和释放、预取、record_stream 等工程细节配合。

### 9.2 demo_05 时序图

两条线：默认计算 stream 和 D2H stream。

```text
时间向下
│
│  Default / compute stream                 D2H stream
│
│  block0 forward
│    ctx.save_for_backward(block0_input)
│    pack(block0_input)
│    swap0.launch_d2h()
│    ├─ 提交 D2H 任务 ───────────────────────► wait compute stream
│    │                                        copy block0_input GPU -> CPU
│    │                                        record d2h_done event
│
│  block0 forward 返回
│  block1 forward 开始
│
│  block1 forward
│    ctx.save_for_backward(block1_input)
│    pack(block1_input)
│
│    release_previous_swaps()
│      wait swap0.d2h_done event ◄─────────── block0 D2H 完成
│      swap0.tensor.storage().resize_(0)
│      block0_input GPU storage 释放
│
│    swap1.launch_d2h()
│    ├─ 提交 D2H 任务 ───────────────────────► wait compute stream
│    │                                        copy block1_input GPU -> CPU
│    │                                        record d2h_done event
│
│  block1 forward 返回
│
│  forward 全部结束
│    release_previous_swaps()
│      swap0 已释放，跳过
│      wait swap1.d2h_done event ◄─────────── block1 D2H 完成
│      swap1.tensor.storage().resize_(0)
│      block1_input GPU storage 释放
│
│  backward 开始
│
│  backward block1
│    unpack(swap1)
│      swap1.tensor.storage().resize_(original_size)
│      copy swap1.cpu_tensor CPU -> GPU
│      return swap1.tensor
│
│  backward block0
│    unpack(swap0)
│      swap0.tensor.storage().resize_(original_size)
│      copy swap0.cpu_tensor CPU -> GPU
│      return swap0.tensor
│
▼
```

### 9.3 为什么 block0 不立刻释放，而是下个 block 释放

因为 D2H 是异步的。

在 block0 的 `pack` 里：

```python
swap.launch_d2h()
return swap
```

`launch_d2h()` 只是提交 copy 任务，不代表 CPU copy 已经完成。

如果马上：

```python
self.tensor.storage().resize_(0)
```

可能出现：

```text
D2H stream 还没读完 GPU tensor
GPU storage 已经被缩到 0
CPU copy 得到错误或不完整数据
```

正确顺序必须是：

```text
发起 D2H
等待 D2H 完成
释放 GPU storage
```

如果在 block0 里同步等待：

```text
block0 forward
block0 D2H
wait D2H 完成
release
block1 forward
```

显存释放更早，但 forward 被 D2H 卡住。

XTuner 风格是：

```text
block0 forward
发起 block0 D2H
block1 forward 和 block0 D2H 尽量重叠
进入 block1 pack 时再等 block0 D2H 完成并释放
```

这是用一层的释放滞后来换计算和拷贝重叠。

### 9.4 为什么 demo_05 backward 后显存变多

你看到的典型输出：

```text
[mem] after releasing saved GPU storages   allocated=    32.0 MB
[mem] after backward                       allocated=    80.0 MB
```

这是符合预期的。

forward 结束并释放 saved storage 后，约 32 MB：

```text
x leaf tensor                 16 MB
final hidden                  16 MB
合计                          32 MB
```

backward 时需要恢复两个 saved tensor：

```text
block1 saved input            16 MB
block0 saved input            16 MB
```

还会产生 leaf gradient：

```text
x.grad                        16 MB
```

所以：

```text
32 + 16 + 16 + 16 = 80 MB
```

demo 里还有 `pending_swaps` 引用着 restored tensor，所以 backward 后 allocated 偏高。清掉教学引用后再看：

```python
pending_swaps.clear()
del loss, hidden
torch.cuda.empty_cache()
```

显存会下降。真实 XTuner 通过 `OffloadManager` 做更完整的生命周期管理。

## 10. demo_06：一个 op 保存多个 tensor

运行：

```bash
python workspace/hha_code/activation_offload_demo.py --demo 6
```

这个 demo 用来回答一个细节问题：

```python
ctx.save_for_backward(x, y)
```

这种情况下，`pack/unpack` 是触发一次，还是对每个 tensor 分别触发？

结论是：**分别触发**。

`SaveTwoTensors.forward` 里：

```python
ctx.save_for_backward(x, y)
```

典型输出：

```text
[forward:save_two] called ctx.save_for_backward(x, y)
[hook:pack #1] tensor device=cuda:0, shape=(2048, 2048)
[hook:pack #2] tensor device=cuda:0, shape=(2048, 2048)
```

说明 forward 保存两个 tensor 时，PyTorch 会分别调用：

```text
pack(x)
pack(y)
```

backward 里：

```python
x, y = ctx.saved_tensors
```

典型输出：

```text
[hook:unpack #1] from pack #1
[hook:unpack #2] from pack #2
[backward:save_two] loaded x
[backward:save_two] loaded y
```

说明读取 `ctx.saved_tensors` 时，PyTorch 也会分别调用：

```text
unpack(payload_for_x)
unpack(payload_for_y)
```

顺序也保持保存顺序：

```text
ctx.save_for_backward(x, y)
  -> pack x
  -> pack y

ctx.saved_tensors
  -> unpack x
  -> unpack y
```

这里要区分两种顺序：

```text
不同 block/layer 之间:
  forward: block0 -> block1
  backward: block1 -> block0
  这是反的

同一个 Function 里多个 saved tensor:
  save_for_backward(x, y)
  ctx.saved_tensors -> x, y
  这个保持保存顺序
```

显存上，`n=2048` 时：

```text
after creating x and y: 64 MB
```

因为 `make_hidden(n)` 会产生 leaf 和 hidden 两个 16 MB tensor。创建两个 hidden：

```text
x 的 leaf + hidden: 32 MB
y 的 leaf + hidden: 32 MB
合计: 64 MB
```

backward 后常见：

```text
after backward: 112 MB
```

大致是：

```text
x 的 leaf + hidden: 32 MB
y 的 leaf + hidden: 32 MB
out: 16 MB
两个 leaf grad: 32 MB
合计: 112 MB
```

这个 demo 对理解 XTuner 很有用：`async_save_on_cpu` 里同一个 block 可能 pack 多次时，`GetCnt` 的 `tensor_idx` 就是这样一点点递增出来的。

## 11. 回到 XTuner：async_save_on_cpu 的结构

核心类：

```python
class async_save_on_cpu(saved_tensors_hooks):
```

它就是 PyTorch `saved_tensors_hooks` 的工程化包装。

初始化时传入：

```text
h2d_stream:
  CPU -> GPU 时使用的 stream

d2h_stream:
  GPU -> CPU 时使用的 stream

block_idx:
  当前第几个 block/layer

group:
  区分 text、vision 等不同 offload 域

custom_check_fn:
  只 offload 满足条件的 tensor

prefetch:
  backward 时是否预取前一层，基本上都是 true，否则 backward 肯定太慢了

reserve_pin_memory:
  是否复用 pinned CPU buffer
```

### 11.1 base_check_fn

`base_check_fn` 会跳过不适合 offload 的 tensor：

```python
if isinstance(tensor._base, torch.nn.parameter.Parameter) or isinstance(tensor, torch.nn.parameter.Parameter):
    return False
if tensor.untyped_storage().size() <= 0:
    return False
return True
```

含义：

```text
不要 offload Parameter
不要 offload 空 storage tensor
```

activation offload 主要针对中间激活，不是参数。

### 11.2 SwapTensor

XTuner 的 `SwapTensor` 对应 demo_05 的 `SimpleSwapTensor`，但更完整。

关键字段：

```text
self.tensor:
  原 CUDA tensor

self.storage_size:
  原 storage 大小

self.tensor_cpu:
  CPU pinned buffer

self.is_slice_tensor:
  是否是 slice/view 类 tensor

self.stat:
  当前状态，device 或 host

self.h2d_event:
  H2D 完成事件
```

关键方法：

```text
launch_d2h:
  异步 GPU -> CPU

wait_d2h_finished:
  等 D2H 完成，然后 tensor.storage().resize_(0)

launch_h2d:
  backward 需要时恢复 storage，并 CPU -> GPU

prefetch_launch_h2d:
  backward 前预取
```

最关键的释放语句：

```python
self.tensor.storage().resize_(0)
```

这才是真正让原 activation 的 GPU storage 下降的地方。

### 11.3 OffloadManager

XTuner 不是只管理一个 tensor，而是管理很多层、很多 tensor。

所以有 `OffloadManager`：

```text
items:
  当前还在 offload 管理中的 SwapTensor

getcnt:
  每个 group 下的 block/tensor 计数

may_npu_tensors:
  backward restore 后、可能还在 device 上的 tensor

pin_memory_cache:
  reserve_pin_memory=True 时复用 CPU pinned buffer
```

`GetCnt` 负责生成 key：

```text
block_idx_tensor_idx
```

再加上 group 变成：

```text
text_3_0
vision_5_2
```

这样可以区分不同模块、不同层、不同 saved tensor。

### 11.4 为什么需要 group

`group` 的核心原因很简单：**不同模块的 layer index 可能重复**。

比如 Qwen3 VL 这种 compose model 里，至少有两段 transformer-like stack：

```text
vision_tower:
  layer 0, 1, 2, ...

language_model:
  layer 0, 1, 2, ...
```

如果 offload key 只用：

```text
{block_idx}_{tensor_idx}
```

那么就会撞：

```text
vision layer 0 saved tensor -> 0_0
text   layer 0 saved tensor -> 0_0
```

加上 group 后就变成：

```text
vision_0_0
text_0_0
```

所以 `group` 本质上是 namespace，用来隔离不同模块的：

```text
block/tensor 计数
items key
pin_memory_cache
release / prefetch 逻辑
```

否则就需要手动给不同模块的 layer index 加 offset，例如 vision 从 0 开始、text 从 10000 开始，或者维护多个 manager。`group` 是更直接的做法。

Qwen3 VL 的 vision encoder 里使用：

```python
with async_save_on_cpu(
    h2d_stream=self.offload_stream,
    d2h_stream=self.offload_stream,
    block_idx=int(layer_num),
    group="vision",
    custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
):
    hidden_states = blk(...)
```

而 text / MoE 侧也有自己的 layer index。它们都可能从 0 开始计数，所以需要通过 group 避免 key 冲突。

### 11.5 pack 阶段

XTuner 的 `_pack_to_cpu` 大致流程：

```text
1. base_check_fn 过滤
2. custom_check_fn 过滤
3. 通过 OffloadManager 生成 key
4. 如果进入了新 block，释放前一个 block 已完成 D2H 的 tensor storage
5. 创建或复用 CPU pinned buffer
6. 创建 SwapTensor
7. 发起 D2H
8. OffloadManager 保存 SwapTensor
9. return swap_tensor
```

对应关键代码：

```python
if after_block and (prev_block_idx is not None):
    OffloadManager().del_npu_tensor(f"{group}_{prev_block_idx}_", d2h_stream)
```

这和 demo_05 的：

```python
release_previous_swaps()
```

是同一个思想。

发起 D2H：

```python
d2h_stream.wait_stream(working_stream)
swap_tensor.launch_d2h(d2h_stream)
```

保存 payload：

```python
OffloadManager().put(full_key, swap_tensor)
return swap_tensor
```

因为 `async_save_on_cpu` 继承了 `saved_tensors_hooks`，所以这里 return 的 `swap_tensor` 就是 autograd graph 最终保存的 payload。

### 11.6 unpack 阶段

XTuner 的 `_unpack_from_cpu` 大致流程：

```text
1. 如果 payload 本来就是普通 tensor，直接返回
2. 当前 backward stream 与 h2d_stream 做同步
3. 从 swap_tensor.key 解析 block_idx/tensor_idx
4. 清理下一层可能已经 restore 到 device 的 tensor
5. 当前 tensor H2D restore
6. 如果 prefetch=True，预取前一层的 tensor
7. return swap_tensor.tensor
```

关键：

```python
swap_tensor.launch_h2d(h2d_stream, True, working_stream)
return swap_tensor.tensor
```

这里不是简单：

```python
return tensor_cpu.to("cuda")
```

而是恢复原 `swap_tensor.tensor` 的 storage。

prefetch：

```python
if prefetch and block_idx != 0:
    OffloadManager().prefetch_get(...)
```

backward 是反向逐层走的。当前层 backward 时，可以提前把前一层需要的 activation 从 CPU 拷回 GPU，以减少下一步等待。

### 11.7 prefetch 的核心作用

`prefetch` 的核心作用是：**提前把下一步 backward 将要用到的 activation 从 CPU 搬回 GPU，尽量隐藏 H2D 拷贝时间**。

它主要是性能优化，不是 correctness 必需项。

#### 简单情况：每个 block 只有一个 tensor

先看最简单的情况。假设 forward 顺序是：

```text
block0 -> block1 -> block2
```

backward 顺序会反过来：

```text
block2 backward -> block1 backward -> block0 backward
```

如果没有 prefetch，时序更像：

```text
backward block2:
  等 block2 activation CPU -> GPU
  计算 block2 backward

backward block1:
  等 block1 activation CPU -> GPU
  计算 block1 backward

backward block0:
  等 block0 activation CPU -> GPU
  计算 block0 backward
```

有 prefetch 后，目标是：

```text
backward block2:
  恢复 block2 activation
  同时/提前调度 block1 activation CPU -> GPU
  计算 block2 backward

backward block1:
  block1 activation 最好已经在 GPU 上
  同时/提前调度 block0 activation CPU -> GPU
  计算 block1 backward

backward block0:
  block0 activation 最好已经在 GPU 上
  计算 block0 backward
```

也就是：

```text
当前 block backward 计算
  和
前一个 block activation H2D

尽量重叠
```

在这种情况下，`get_prefetch_keys` 退化成很直观的逻辑：

```text
当前 unpack block2 tensor0 -> prefetch block1 tensor0
当前 unpack block1 tensor0 -> prefetch block0 tensor0
```

#### Micro-batch 情况：一个 block 有多个 tensor

多 micro-batch / domino EP 路径里，`hidden_states_list` 里有多个 hidden states：

```text
hidden_states_list = [
  mb0_hidden,
  mb1_hidden,
  mb2_hidden,
]
```

同一个 block 可能为多个 micro-batch 各 offload 一个 tensor：

```text
block0:
  text_0_0  -> block0, micro-batch 0 的 hidden
  text_0_1  -> block0, micro-batch 1 的 hidden
  text_0_2  -> block0, micro-batch 2 的 hidden

block1:
  text_1_0
  text_1_1
  text_1_2
```

backward 处理 block1 时，当前 block1 的 tensor 会由 autograd 按需 unpack。比如访问 `text_1_0` 时：

```text
必须做:
  restore text_1_0
  因为 block1 backward 现在就要用它

可选 prefetch:
  提前 restore text_0_0
  因为 block1 backward 后面会进入 block0 backward
```

所以 prefetch 不是为了预取当前 block 的下一个 tensor，比如 `text_1_1`。`text_1_1` 属于当前 block，autograd 很快会自己 unpack 它。

一定要理解这个逻辑，因为他不是 text_1_0 perfetch text_1_1，而是 text_0_0。原因是 text_1_0 执行后会自动调用 text_1_1 的，不需要你 perfetch。你需要 perfetch 的是 text_0_0。否则上一个 block 不会有 perfetch 了。虽然这样会有可能同时存在 6 个激活层，但是通常都不大，能接受。

prefetch 关心的是下一步 backward block：

```text
当前正在 backward block1
  -> 提前准备 block0

当前正在 backward block2
  -> 提前准备 block1
```

如果每个 block 都有 3 个 tensor，映射就是：

```text
unpack text_1_0 -> prefetch text_0_0
unpack text_1_1 -> prefetch text_0_1
unpack text_1_2 -> prefetch text_0_2
```

注意两个顺序不要混淆：

```text
不同 block 之间:
  backward 顺序和 forward 相反，block1 -> block0

同一个 Function 内多个 saved tensor:
  ctx.save_for_backward(x, y)
  unpack 顺序仍然是 x -> y
```

#### 复杂情况：前后 block tensor 数不一致

XTuner 的 `get_prefetch_keys` 写得更泛化。它不是固定预取“前一层第 0 个 tensor”，而是根据当前 `tensor_idx` 映射到前一个 block 的对应区间：

```python
start = tensor_idx * prefetch_block_tensor_nums // block_tensor_nums
end = (tensor_idx + 1) * prefetch_block_tensor_nums // block_tensor_nums
```

如果前后 block 数量一样，比如都是 3 个：

```text
prev_num = 3
curr_num = 3

tensor_idx 0 -> prefetch [0]
tensor_idx 1 -> prefetch [1]
tensor_idx 2 -> prefetch [2]
```

如果前一个 block 有 4 个 tensor，当前 block 有 2 个：

```text
prev_num = 4
curr_num = 2

tensor_idx 0 -> prefetch [0, 1]
tensor_idx 1 -> prefetch [2, 3]
```

如果前一个 block 有 2 个 tensor，当前 block 有 4 个：

```text
prev_num = 2
curr_num = 4

tensor_idx 0 -> prefetch []
tensor_idx 1 -> prefetch [0]
tensor_idx 2 -> prefetch []
tensor_idx 3 -> prefetch [1]
```

这个公式的本质是：

```text
把前一个 block 的 tensor 区间 [0, prev_num)
按比例分摊到当前 block 的 curr_num 次 unpack 触发点上
```

#### 显存 tradeoff

prefetch 会增加峰值显存风险。

因为当前 block backward 期间，GPU 上可能同时有：

```text
当前 block 正在用的 saved tensor
已经 prefetch 回来的前一个 block saved tensor
backward 临时 tensor / grad
```

所以：

```text
prefetch=False:
  backward 需要哪个 tensor，现场 H2D 哪个 tensor
  显存压力更低，但更容易等待 copy

prefetch=True:
  当前层 backward 时，提前 H2D 前一层 activation
  等待更少，但峰值显存可能更高
```

## 12. XTuner MoE 里怎么用

MoE 单 micro-batch 路径：

```python
if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
    with async_save_on_cpu(
        h2d_stream=self.offload_stream,
        d2h_stream=self.offload_stream,
        block_idx=int(idx),
        group="text",
        custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
    ):
        layer_results = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )
```

关键是：

```python
custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr()
```

含义：

```text
虽然 saved_tensors_hooks 会看到当前 context 内很多 saved tensor
但 XTuner 只 offload 当前层输入 hidden_states
```

这样做更保守，避免把参数、小 tensor、router 中间结果等都 offload，降低副作用。

这里的 `group="text"` 可以理解成 text language model 的 namespace。对于 Qwen3 VL 这类模型，vision tower 也可能启用 activation offload，并且 vision layer index 也会从 0 开始。如果没有 group，vision layer 0 和 text layer 0 的 offload key 就容易混在一起。

### 12.1 单 batch + checkpoint_wrapper + activation offload 的运行流程

XTuner 里某些 layer 会先套一层 checkpoint wrapper：

```python
layer = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.REENTRANT)
```

这会改变 `saved_tensors_hooks` 看到的保存对象。

不套 checkpoint 时：

```text
decoder_layer 内部很多 op 都可能保存 activation
saved_tensors_hooks 可能看到 layer 内部 op 保存的 tensor
custom_check_fn 再从中筛出 hidden_states
```

套 reentrant checkpoint 后：

```text
checkpoint wrapper 是一个 autograd Function
真实 decoder_layer forward 通常在 no_grad 下执行
内部 op 不保存普通 activation
checkpoint wrapper 主要保存边界输入 hidden_states
```

所以可以把 offload 理解成：

```text
checkpoint:
  不保存 layer 内部 activation，backward 时重算 decoder_layer forward

activation offload:
  把 checkpoint 边界输入 hidden_states 搬到 CPU
```

用单 batch、两层模型举例：

```text
h0 -> layer0(checkpoint_wrapper) -> h1 -> layer1(checkpoint_wrapper) -> h2
```

forward 阶段：

```text
进入 layer0 的 async_save_on_cpu context

layer0 checkpoint wrapper forward:
  保存 wrapper backward 重算需要的输入 h0
  内部真实 layer0 forward 在 no_grad 下执行
  得到 h1

layer0 wrapper forward 返回后:
  PyTorch 对 wrapper 保存的 h0 触发 pack
  custom_check_fn 判断 data_ptr == h0.data_ptr()
  匹配成功
  发起 h0 D2H: GPU -> CPU
  注意：这时 h0 GPU storage 还没有立刻 resize_(0)

进入 layer1 的 async_save_on_cpu context

layer1 checkpoint wrapper forward:
  保存 wrapper backward 重算需要的输入 h1
  内部真实 layer1 forward 在 no_grad 下执行
  得到 h2

layer1 pack(h1) 触发时:
  发现进入了新 block
  先等待 h0 D2H 完成
  h0.storage().resize_(0)
  释放 h0 的 GPU storage
  再发起 h1 D2H
```

forward 结束后，如果没有下一层触发释放，最后一层保存的 h1 需要在后续清理点等待 D2H 完成并释放。

backward 阶段：

```text
先进入 layer1 backward

checkpoint wrapper 需要重算 layer1 forward:
  读取 saved input h1
  触发 unpack(h1)
  h1.storage().resize_(原大小)
  CPU -> GPU 拷回 h1
  用 h1 重新跑 layer1 forward
  再计算 layer1 backward

然后进入 layer0 backward

checkpoint wrapper 需要重算 layer0 forward:
  读取 saved input h0
  触发 unpack(h0)
  h0.storage().resize_(原大小)
  CPU -> GPU 拷回 h0
  用 h0 重新跑 layer0 forward
  再计算 layer0 backward
```

所以在 checkpoint wrapper 场景下，可以把 `pack` 时刻近似理解成：

```text
checkpoint wrapper forward 返回后，
PyTorch 保存 wrapper 输入时触发 pack。
```

这也是为什么 `custom_check_fn` 匹配 `hidden_states` 很自然：checkpoint wrapper 最重要的 saved tensor 正是这个 layer 的输入 hidden_states。

多 micro-batch / domino EP 路径：

```python
with async_save_on_cpu(
    h2d_stream=self.offload_stream,
    d2h_stream=self.offload_stream,
    block_idx=layer_idx - self.config.first_k_dense_replace,
    group="text",
    custom_check_fn=lambda x: x.data_ptr()
    in [hidden_states.data_ptr() for hidden_states in hidden_states_list],
    prefetch=True,
    reserve_pin_memory=True,
):
    layer_results = decoder_layer(...)
```

这里有几个点：

```text
hidden_states_list:
  多个 micro-batch 的 hidden_states

custom_check_fn:
  只 offload 这些 hidden_states

prefetch=True:
  backward 时预取前面 block 的 activation

reserve_pin_memory=True:
  复用 CPU pinned buffer，减少反复分配 pin memory 的成本
```

## 13. XTuner 里的两个重要 caveat

### 13.1 chunk 共享 storage 问题

MoE 里有：

```python
hidden_states_list = [i.clone() for i in cat_hidden_states.chunk(len(seq_ctx_list), dim=1)]
```

注释说明：当前 offload 实现对 `chunk` 这类共享 storage 的 tensor 不友好，可能导致 nan grad norm。

原因可以按这个思路理解：

```text
chunk 出来的多个 tensor 可能共享同一块底层 storage
offload 会对某个 tensor.storage().resize_(0)
如果其他 tensor 也依赖这块 storage，就可能出问题
```

所以这里用 `clone()` 让每个 hidden state 拥有自己的 storage。

### 13.2 inputs_embeds clone 问题

MoE `_forward` 里：

```python
hidden_states = seq_ctx.inputs_embeds.clone()
```

原因是当前 activation offload 会原地修改 tensor storage：

```python
tensor.storage().resize_(0)
```

如果直接拿 `seq_ctx.inputs_embeds` 做 hidden_states，后续还访问 `inputs_embeds` 时可能发现 storage 已经空了。

所以 clone 一份作为可被 offload 修改 storage 的 working tensor。

## 14. 和 worker.py 里的 offload_model 有什么不同

`xtuner/v1/rl/trainer/worker.py` 里：

```python
def offload_model(self):
    self._engine.put_model_to_device("cpu")
    DEVICE_MODULE.empty_cache()
```

这和 activation offload 不是同一种机制。

模型 offload 是：

```text
model 参数 / buffer 从 CUDA 迁移到 CPU
model 不再持有 CUDA 参数 tensor
旧 CUDA storage 没有引用后可释放
empty_cache 清掉空闲缓存
```

optimizer offload 类似：

```python
state[key] = val.to(device, non_blocking=True)
```

它把 optimizer state dict 里的 CUDA tensor 替换成 CPU tensor。

而 demo_03 这种：

```python
cpu_tensor = t.cpu()
return cpu_tensor
```

只是多了一份 CPU payload，原 CUDA activation 仍可能被计算图引用，所以显存不降。

activation offload 更难，是因为 activation 仍然是 backward 所需要的 saved tensor。不能简单把引用替换掉就完事，还要保证 backward 前恢复。

## 15. 总结

`async_save_on_cpu` 的核心可以压缩成一句话：

```text
用 saved_tensors_hooks 接管 autograd saved tensor；
forward 保存时把指定 activation 异步 D2H 到 CPU；
D2H 完成后把原 CUDA storage resize 到 0；
backward 需要时再恢复原 storage 并 H2D 拷回。
```

几个关键结论：

```text
1. ctx.save_for_backward 只是登记，pack 的返回值才是 autograd graph 最终保存的 payload。

2. t.cpu() 只复制到 CPU，不会自动释放原 CUDA tensor。

3. empty_cache 只能释放空闲缓存，不能释放 live tensor。

4. 真正让 activation GPU 显存下降的是 storage().resize_(0) 或去掉所有 CUDA tensor 引用。

5. 异步 D2H 不能立刻释放 storage，必须等 D2H 完成。

6. 延迟到下一个 block 释放，是为了让 D2H 和下一层 forward 尽量重叠。

7. backward 后显存升高是正常的，因为 saved activation 要恢复，leaf grad 和临时 tensor 也会出现。

8. XTuner 通过 custom_check_fn 只 offload hidden_states，避免扩大影响面。
```

