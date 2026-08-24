"""
Small demos for learning PyTorch activation offload.

Run:
    python workspace/hha_code/activation_offload_demo.py

Run one demo:
    python workspace/hha_code/activation_offload_demo.py --demo 3

Make tensors larger or smaller:
    python workspace/hha_code/activation_offload_demo.py --n 4096

The demos intentionally go from simple to more realistic:
1. A custom autograd Function saves a tensor for backward.
2. saved_tensors_hooks shows when PyTorch packs/unpacks saved tensors.
3. A minimal synchronous CPU offload hook.
4. An asynchronous D2H hook using pinned memory, CUDA stream, and event.
5. An XTuner-like version that releases GPU storage after D2H and restores it in backward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.graph import saved_tensors_hooks


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires CUDA. Please run it in a CUDA-enabled environment.")


def mem(label: str, *, sync: bool = True) -> None:
    if sync:
        torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[mem] {label:<36} allocated={allocated:8.1f} MB  reserved={reserved:8.1f} MB  peak={max_allocated:8.1f} MB")


def reset_memory() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_hidden(n: int) -> torch.Tensor:
    x = torch.randn(n, n, device="cuda", requires_grad=True)
    # Make a non-leaf activation. Real model hidden_states are usually non-leaf activations,
    # not Parameter objects.
    return x * 1.0  # x * 1.0 是为了让 hidden 变成 non-leaf activation，更接近模型里的 hidden_states


class SquareThatSaves(torch.autograd.Function):
    """A tiny op whose backward needs the forward input."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, name: str) -> torch.Tensor:
        out = x * x
        ctx.name = name
        # 这里只是登记 x 会被 backward 用到。
        # 如果外层启用了 saved_tensors_hooks，PyTorch 会在 forward 返回后调用 pack(x)。
        # autograd graph 最终保存的是 pack(x) 的返回值，而不一定是原始 CUDA tensor。
        ctx.save_for_backward(x)
        print(
            f"[forward:{name}] called ctx.save_for_backward(x): "
            f"device={x.device}, shape={tuple(x.shape)}"
        )
        print(f"[forward:{name}] custom Function forward is about to return")
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        print(f"[backward:{ctx.name}] loaded saved x: device={x.device}, shape={tuple(x.shape)}")
        return grad_out * 2 * x, None # 返回 x 的梯度


def square_that_saves(x: torch.Tensor, name: str) -> torch.Tensor:
    return SquareThatSaves.apply(x, name)


class SaveTwoTensors(torch.autograd.Function):
    """Show that save_for_backward(x, y) packs/unpacks each tensor separately.

    重点：ctx.save_for_backward(x, y) 不是把 (x, y) 当成一个整体 payload。
    PyTorch 会对 x 和 y 分别走 saved tensor hook：
      pack(x)
      pack(y)
    backward 读取 ctx.saved_tensors 时也会分别触发：
      unpack(payload_for_x)
      unpack(payload_for_y)
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = x * y
        # 正确的多 tensor 保存方式：把多个 tensor 作为多个参数传入。
        # hooks 会逐 tensor 处理，而不是一次性把它们打包成 list/tuple 处理。
        ctx.save_for_backward(x, y)
        print("[forward:save_two] called ctx.save_for_backward(x, y)")
        print("[forward:save_two] custom Function forward is about to return")
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 访问 ctx.saved_tensors 时，PyTorch 会先对每个 saved payload 调用 unpack。
        # 所以下面真正拿到的 x/y 已经是 unpack 返回的 tensor。
        x, y = ctx.saved_tensors
        print(f"[backward:save_two] loaded x: device={x.device}, shape={tuple(x.shape)}")
        print(f"[backward:save_two] loaded y: device={y.device}, shape={tuple(y.shape)}")
        return grad_out * y, grad_out * x


def save_two_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return SaveTwoTensors.apply(x, y)


def demo_01_normal_saved_tensor(n: int) -> None:
    print("\n=== demo_01: normal autograd saved tensor ===")
    reset_memory()
    
    # n=2048时候，hidden 的 shape 是 (2048, 2048)，理论上是 16m，但是因为内部有两行代码，激活值是 2 倍，因此是 32m
    hidden = make_hidden(n)
    # [mem] after creating hidden                allocated=    32.0 MB  reserved=    32.0 MB  peak=    32.0 MB
    mem("after creating hidden")
    
    # 内部会存一份，所以现在变成 48m
    out = square_that_saves(hidden, "normal")
    loss = out.sum()
    # [mem] after forward                        allocated=    48.0 MB  reserved=    50.0 MB  peak=    48.0 MB
    mem("after forward")
    
    # PyTorch 会给 leaf tensor x 写入梯度 x.grad 16 MB，所以最终是 64m
    # peak=80 MB 是 backward 过程中短暂出现过额外临时 tensor。比如这个 backward 公式：return grad_out * 2 * x, None 中间会产生临时结果
    print("[main] backward starts")
    loss.backward()
    # [mem] after backward                       allocated=    64.0 MB  reserved=    82.0 MB  peak=    80.0 MB
    mem("after backward")

    #中间会产生临时结果，再加上最后写入的 x.grad，所以峰值比最终 allocated 更高：
    #  backward 前存活              48 MB
    #  临时梯度/中间结果             16 MB
    #  x.grad                      16 MB
    #  峰值约                      80 MB


def demo_02_observe_saved_tensors_hooks(n: int) -> None:
    """
    [forward:observe_hooks] called ctx.save_for_backward(x): device=cuda:0, shape=(2048, 2048)
    [forward:observe_hooks] custom Function forward is about to return
    [hook:pack] PyTorch wants to save tensor: device=cuda:0, shape=(2048, 2048)
    [main] forward context exited; backward starts
    [hook:unpack] PyTorch needs saved tensor for backward: device=cuda:0, shape=(2048, 2048)
    [backward:observe_hooks] loaded saved x: device=cuda:0, shape=(2048, 2048)
    [mem] after backward                       allocated=    64.0 MB  reserved=    82.0 MB  peak=    80.0 MB
    """
    print("\n=== demo_02: observe saved_tensors_hooks ===")
    reset_memory()

    def pack(t: torch.Tensor) -> torch.Tensor:
        # forward 阶段：PyTorch 准备把 t 保存进 autograd graph 时调用。
        # 返回值会替代原始 tensor 被保存。
        print(f"[hook:pack] PyTorch wants to save tensor: device={t.device}, shape={tuple(t.shape)}")
        return t

    def unpack(t: torch.Tensor) -> torch.Tensor:
         # backward 阶段：PyTorch 需要读取 saved tensor 时调用。
        # 返回值必须是 backward 真正可用的 tensor。
        print(f"[hook:unpack] PyTorch needs saved tensor for backward: device={t.device}, shape={tuple(t.shape)}")
        return t

    hidden = make_hidden(n)
    with saved_tensors_hooks(pack, unpack):
        # 可以看出，执行完成 forward 后，会调用 pack 函数
        out = square_that_saves(hidden, "observe_hooks")
        # backward 阶段，因为要读取，所有是先调用 unpack 函数，然后才执行对应 op 的 backward 。
        loss = out.sum()

    print("[main] forward context exited; backward starts")
    loss.backward()
    mem("after backward")


def demo_03_sync_cpu_offload(n: int) -> None:
    """
    [mem] after creating hidden                allocated=    32.0 MB  reserved=    32.0 MB  peak=    32.0 MB
    [forward:sync_offload] called ctx.save_for_backward(x): device=cuda:0, shape=(2048, 2048)
    [forward:sync_offload] custom Function forward is about to return
    [hook:pack] D2H now: cuda:0 -> cpu
    [mem] after forward; saved tensor copy is on CPU allocated=    48.0 MB  reserved=    50.0 MB  peak=    48.0 MB
    [main] backward starts
    [hook:unpack] H2D now: cpu -> cuda:0
    [backward:sync_offload] loaded saved x: device=cuda:0, shape=(2048, 2048)
    [mem] after backward                       allocated=    64.0 MB  reserved=    98.0 MB  peak=    96.0 MB
    """
    print("\n=== demo_03: minimal synchronous CPU offload ===")
    reset_memory()
    
    # 这中写法显存其实并没有任何下降。
    def pack(t: torch.Tensor) -> tuple[torch.Tensor, torch.device]:
        print(f"[hook:pack] D2H now: {t.device} -> cpu")
        cpu_tensor = t.detach().cpu()
        # pack -> t.cpu() 只是让 autograd graph 保存 CPU payload，并不会自动销毁原来的 CUDA tensor. empty_cache 也没用。
        # worker.offload_model() 有效，是因为它系统性地把 model 参数引用替换成 CPU tensor。
        return cpu_tensor, t.device

    def unpack(payload: tuple[torch.Tensor, torch.device]) -> torch.Tensor:
        cpu_tensor, device = payload
        print(f"[hook:unpack] H2D now: cpu -> {device}")
        return cpu_tensor.to(device)
    
    # [mem] after creating hidden                allocated=    32.0 MB  reserved=    32.0 MB  peak=    32.0 MB
    hidden = make_hidden(n)
    mem("after creating hidden")

    with saved_tensors_hooks(pack, unpack):
        out = square_that_saves(hidden, "sync_offload")
        loss = out.sum()
    # [mem] after forward; saved tensor copy is on CPU allocated=    48.0 MB  reserved=    50.0 MB  peak=    48.0 MB
    print("[note] pack returned a CPU payload, but the original CUDA hidden tensor still has live references.")
    print("[note] So demo_03 shows hook-based offload mechanics, not GPU storage release yet.")
    mem("after forward; CPU payload saved, CUDA storage still live")

    print("[main] backward starts")
    loss.backward()
    # [mem] after backward                       allocated=    64.0 MB  reserved=    98.0 MB  peak=    96.0 MB
    mem("after backward")


@dataclass
class AsyncCpuPayload:
    # pack 返回给 autograd graph 的 payload。
    # 注意：这里保存的是 CPU 副本 + 原始设备 + D2H 完成事件，而不是直接保存 CUDA tensor。
    cpu_tensor: torch.Tensor
    original_device: torch.device
    d2h_done: torch.cuda.Event


def demo_04_async_d2h_without_freeing_storage(n: int) -> None:
    """
    [forward:async_no_free] called ctx.save_for_backward(x): device=cuda:0, shape=(2048, 2048)
    [forward:async_no_free] custom Function forward is about to return
    [hook:pack] schedule async D2H on a side stream
    [mem] after forward; original GPU tensor still exists allocated=    48.0 MB  reserved=    50.0 MB  peak=    48.0 MB
    [main] backward starts
    [hook:unpack] wait D2H event, then copy CPU tensor back to GPU
    [backward:async_no_free] loaded saved x: device=cuda:0, shape=(2048, 2048)
    [mem] after backward                       allocated=    64.0 MB  reserved=    98.0 MB  peak=    96.0 MB
    """
    print("\n=== demo_04: async D2H with pinned memory, but no GPU storage free ===")
    reset_memory()

    # 额外创建一个 CUDA stream 专门做 device-to-host copy。
    # 这样当前默认 stream 可以继续做后续计算，D2H copy 尽量在旁路 stream 上异步进行。
    d2h_stream = torch.cuda.Stream()

    def pack(t: torch.Tensor) -> AsyncCpuPayload:
        # pinned memory 是页锁定 CPU 内存，适合 GPU DMA 拷贝。
        # 没有 pin_memory=True 时，non_blocking=True 很多情况下不能真正异步。
        cpu_tensor = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)

        # event 用来记录“D2H copy 已经完成”。
        # backward 的 unpack 阶段必须等这个 event，否则可能读到还没拷完的 CPU tensor。
        d2h_done = torch.cuda.Event()

        print("[hook:pack] schedule async D2H on a side stream")

        # 保证 D2H stream 必须等当前 stream 上产生 t 的计算完成后，才能开始 copy t。
        # 否则可能拷贝到尚未写完的数据。
        d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(d2h_stream):
            # 这里只是把 t 的内容拷贝到 CPU pinned buffer。
            # detach() 是为了说明这份 CPU 副本本身不参与 autograd。
            cpu_tensor.copy_(t.detach(), non_blocking=True)

            # 在 d2h_stream 上记录事件。这个事件完成后，说明 cpu_tensor 内容可用。
            d2h_done.record(d2h_stream)

        # pack 返回 payload 后，autograd graph 保存的是这个 payload。
        # 但是原始 CUDA tensor 的 storage 此时还没有释放，所以 demo_04 的 GPU allocated 不会下降。
        return AsyncCpuPayload(cpu_tensor=cpu_tensor, original_device=t.device, d2h_done=d2h_done)

    def unpack(payload: AsyncCpuPayload) -> torch.Tensor:
        print("[hook:unpack] wait D2H event, then copy CPU tensor back to GPU")
        # backward 真正需要 saved tensor 时，先确保 forward 阶段的异步 D2H 已经完成。
        torch.cuda.current_stream().wait_event(payload.d2h_done)

        # 把 CPU payload 拷回原设备。这里会新建一个 CUDA tensor 返回给 backward 使用。
        # demo_05 会进一步演示 XTuner 风格：恢复原 tensor storage，而不是简单新建 tensor。
        return payload.cpu_tensor.to(payload.original_device, non_blocking=True)

    hidden = make_hidden(n)
    with saved_tensors_hooks(pack, unpack):
        out = square_that_saves(hidden, "async_no_free")
        loss = out.sum()

    # sync=False 是故意的：这里想观察“forward 刚结束时”的状态。
    # 但因为 demo_04 没有 resize_(0)，即使 D2H 完成，原始 CUDA hidden 仍然占显存。
    mem("after forward; original GPU tensor still exists", sync=False)
    print("[main] backward starts")
    loss.backward()
    mem("after backward")


class SimpleSwapTensor:
    """A small, teaching-only version of XTuner's SwapTensor.

    这个对象可以理解为“autograd graph 里保存的占位符”：
    - self.tensor: 原来的 CUDA tensor 对象。
    - self.cpu_tensor: D2H 后保存 activation 内容的 CPU pinned buffer。
    - self.storage_size: 原 CUDA storage 的大小，backward 前要靠它 resize_ 回来。
    - self.d2h_done: D2H 完成事件，释放/恢复前要用它做同步。
    """

    def __init__(self, tensor: torch.Tensor, name: str, d2h_stream: torch.cuda.Stream) -> None:
        # 保存原 tensor 对象本身。demo_05 的重点就是后面恢复这个 tensor 的 storage，
        # 而不是像 demo_04 那样新建一个 CUDA tensor 返回给 backward。
        self.tensor = tensor
        self.name = name
        self.d2h_stream = d2h_stream

        # resize_(0) 后 storage 会变空；backward 前需要知道原大小才能 resize_ 回来。
        self.storage_size = tensor.storage().size()

        # CPU 侧保存 activation 内容。pin_memory=True 让 D2H/H2D 更适合异步拷贝。
        self.cpu_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)

        # 记录 D2H 是否完成，避免 CPU buffer 还没写完就释放 GPU storage 或 backward 读取。
        self.d2h_done = torch.cuda.Event()
        self.released = False

    def launch_d2h(self) -> None:
        print(f"[swap:{self.name}] schedule D2H")
        # D2H 必须等当前 stream 上产生 self.tensor 的计算完成。
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.d2h_stream):
            # 把 activation 内容拷到 CPU。这里没有释放 GPU storage，只是发起异步 copy。
            self.cpu_tensor.copy_(self.tensor.detach(), non_blocking=True)
            self.d2h_done.record(self.d2h_stream)

    def release_gpu_storage_after_d2h(self) -> None:
        if self.released:
            return
        print(f"[swap:{self.name}] wait D2H, then resize original GPU storage to 0")

        # 必须先等 D2H 完成，否则 resize_(0) 可能发生在 copy 读完 GPU 数据之前。
        torch.cuda.current_stream().wait_event(self.d2h_done)
        torch.cuda.current_stream().synchronize()

        # 这是 demo_05 和 demo_04 的关键区别：真正让 allocated 下降的是这一句。
        # tensor 对象还在，但它背后的 CUDA storage 被缩到 0。
        self.tensor.storage().resize_(0)
        self.released = True

    def restore_to_gpu(self) -> torch.Tensor:
        print(f"[swap:{self.name}] restore original GPU storage and copy CPU -> GPU")
        if self.released:
            # backward 要用 saved tensor 时，先把原 CUDA storage 恢复到原大小。
            self.tensor.storage().resize_(self.storage_size)

        # 再把 CPU pinned buffer 里的 activation 内容拷回这个原 tensor。
        self.tensor.copy_(self.cpu_tensor, non_blocking=True)
        return self.tensor


def demo_05_xtuner_like_release_and_restore(n: int) -> None:
    """
    [mem] after creating hidden                allocated=    32.0 MB  reserved=    32.0 MB  peak=    32.0 MB
    [forward:block0] called ctx.save_for_backward(x): device=cuda:0, shape=(2048, 2048)
    [forward:block0] custom Function forward is about to return
    [swap:block0] schedule D2H
    [mem] after block0 forward                 allocated=    48.0 MB  reserved=    48.0 MB  peak=    48.0 MB
    [forward:block1] called ctx.save_for_backward(x): device=cuda:0, shape=(2048, 2048)
    [forward:block1] custom Function forward is about to return
    [swap:block0] wait D2H, then resize original GPU storage to 0
    [swap:block1] schedule D2H
    [mem] after block1 forward; block0 can be freed allocated=    48.0 MB  reserved=    64.0 MB  peak=    64.0 MB
    [main] forward is done; release last pending saved tensor before backward
    [swap:block1] wait D2H, then resize original GPU storage to 0
    [mem] after releasing saved GPU storages   allocated=    32.0 MB  reserved=    64.0 MB  peak=    64.0 MB
    [main] backward starts; block1 restores first, then block0
    [swap:block1] restore original GPU storage and copy CPU -> GPU
    [backward:block1] loaded saved x: device=cuda:0, shape=(2048, 2048)
    [swap:block0] restore original GPU storage and copy CPU -> GPU
    [backward:block0] loaded saved x: device=cuda:0, shape=(2048, 2048)
    [mem] after backward                       allocated=    80.0 MB  reserved=   114.0 MB  peak=   112.0 MB
    [mem] after clearing demo references       allocated=     0.0 MB  reserved=     0.0 MB  peak=   112.0 MB
    """
    print("\n=== demo_05: XTuner-like async offload, release, restore ===")
    reset_memory()

    # demo_05 仍然只用一个 side stream 做 D2H。
    # XTuner 里 h2d_stream/d2h_stream 可以分别传入，这里为了教学保持简单。
    d2h_stream = torch.cuda.Stream()

    # 保存每个 block 的 SwapTensor。真实 XTuner 用 OffloadManager 管理 key、引用计数和清理。
    pending_swaps: list[SimpleSwapTensor] = []

    def release_previous_swaps() -> None:
        # 模拟 XTuner 的策略：进入新 block 时，释放前面已经 D2H 完成的 saved tensor storage。
        for swap in pending_swaps:
            swap.release_gpu_storage_after_d2h()

    def make_hooks(block_name: str):
        def pack(t: torch.Tensor) -> SimpleSwapTensor:
            # pack 在 forward 阶段触发。
            # 新 block 保存自己的 tensor 之前，先尝试释放旧 block 的 GPU storage。
            release_previous_swaps()

            # 为当前 block 的 saved tensor 创建一个 SwapTensor 占位符。
            swap = SimpleSwapTensor(t, block_name, d2h_stream)
            # 为啥在 block0 时候不能直接释放 device 的，而是在下一个 block 时候才释放？
            # block0 的 D2H 是异步的，如果在 to cpu 后释放，就需要 wait event 等待 D2H 完成。
            # 那么就无法实现 copy 和 计算重叠，性能交差，虽然显存会少一点。
            swap.launch_d2h()
            pending_swaps.append(swap)

            # autograd graph 保存 swap；backward 时会把 swap 传给 unpack。
            return swap

        def unpack(swap: SimpleSwapTensor) -> torch.Tensor:
            # unpack 在 backward 真正需要 saved tensor 时触发。
            # 这里恢复原 tensor storage，并把 CPU 内容拷回 GPU。
            return swap.restore_to_gpu()

        return pack, unpack

    hidden = make_hidden(n)
    mem("after creating hidden")

    # block0 forward：会保存 block0 的输入 hidden，并调度 D2H。
    pack0, unpack0 = make_hooks("block0")
    with saved_tensors_hooks(pack0, unpack0):
        hidden = square_that_saves(hidden, "block0")
    # [mem] after block0 forward                 allocated=    48.0 MB  reserved=    48.0 MB  peak=    48.0 MB
    mem("after block0 forward", sync=False)

    # block1 forward：进入 block1 的 pack 时，会先释放 block0 已完成 D2H 的 GPU storage。
    pack1, unpack1 = make_hooks("block1")
    with saved_tensors_hooks(pack1, unpack1):
        hidden = square_that_saves(hidden, "block1")
    mem("after block1 forward; block0 can be freed", sync=False)

    print("[main] forward is done; release last pending saved tensor before backward")
    # forward 结束后没有下一个 block 触发 release，所以手动释放最后一个 block 的 saved tensor storage。
    release_previous_swaps()
    # [mem] after releasing saved GPU storages   allocated=    32.0 MB  reserved=    64.0 MB  peak=    64.0 MB
    mem("after releasing saved GPU storages")

    loss = hidden.sum()
    print("[main] backward starts; block1 restores first, then block0")
    # backward 顺序和 forward 相反：先恢复 block1 saved tensor，再恢复 block0 saved tensor。
    # [mem] after backward                       allocated=    80.0 MB  reserved=   114.0 MB  peak=   112.0 MB
    loss.backward()
    mem("after backward")

    # 教学 demo 为了展示状态，把 swap 对象都留在 pending_swaps 里。
    # pending_swaps 里又引用了 restored 的 CUDA tensor，所以 backward 后 allocated 会偏高。
    # 清理这些教学引用后，再看一次显存，能区分“机制需要的显存”和“demo 变量还活着的显存”。
    pending_swaps.clear()
    del loss, hidden
    torch.cuda.empty_cache()
    # [mem] after clearing demo references       allocated=     0.0 MB  reserved=     0.0 MB  peak=   112.0 MB
    mem("after clearing demo references")


def demo_06_save_multiple_tensors(n: int) -> None:
    """
    Observe how hooks behave when one op saves multiple tensors.

    Expected order:
      forward calls ctx.save_for_backward(x, y)
      pack is called once for x
      pack is called once for y
      backward reads ctx.saved_tensors
      unpack is called once for x payload
      unpack is called once for y payload

    [mem] after creating x and y               allocated=    64.0 MB  reserved=    64.0 MB  peak=    64.0 MB
    [forward:save_two] called ctx.save_for_backward(x, y)
    [forward:save_two] custom Function forward is about to return
    [hook:pack #1] tensor device=cuda:0, shape=(2048, 2048), data_ptr=139927356440576
    [hook:pack #2] tensor device=cuda:0, shape=(2048, 2048), data_ptr=139926830055424
    [main] forward done; total pack calls = 2
    [main] backward starts
    [hook:unpack #1] from pack #1, tensor device=cuda:0, shape=(2048, 2048), data_ptr=139927356440576
    [hook:unpack #2] from pack #2, tensor device=cuda:0, shape=(2048, 2048), data_ptr=139926830055424
    [backward:save_two] loaded x: device=cuda:0, shape=(2048, 2048)
    [backward:save_two] loaded y: device=cuda:0, shape=(2048, 2048)
    [main] backward done; total unpack calls = 2
    [mem] after backward                       allocated=   112.0 MB  reserved=   130.0 MB  peak=   128.0 MB
    """
    print("\n=== demo_06: save_for_backward with two tensors ===")
    reset_memory()

    pack_count = 0
    unpack_count = 0

    def pack(t: torch.Tensor) -> tuple[int, torch.Tensor]:
        nonlocal pack_count
        pack_count += 1
        # save_for_backward(x, y) 会让 pack 被调用两次：
        # 第一次处理 x，第二次处理 y。
        print(
            f"[hook:pack #{pack_count}] tensor device={t.device}, "
            f"shape={tuple(t.shape)}, data_ptr={t.data_ptr()}"
        )
        # 返回一个 payload。这里为了教学仍然返回 CUDA tensor 本身，方便观察触发次数。
        return pack_count, t

    def unpack(payload: tuple[int, torch.Tensor]) -> torch.Tensor:
        nonlocal unpack_count
        unpack_count += 1
        pack_id, t = payload
        # backward 里访问 ctx.saved_tensors 时，unpack 也会被调用两次：
        # 一次恢复 x，一次恢复 y。
        print(
            f"[hook:unpack #{unpack_count}] from pack #{pack_id}, "
            f"tensor device={t.device}, shape={tuple(t.shape)}, data_ptr={t.data_ptr()}"
        )
        return t

    # make_hidden(n) 自身会产生 leaf x 和 non-leaf hidden 两个 16 MB tensor。
    # 这里创建 x 和 y 两个 hidden，所以初始 allocated 约为 4 * 16 MB = 64 MB。
    x = make_hidden(n)
    y = make_hidden(n)
    mem("after creating x and y")
    
    # 如果存入 x y，那么 backward 时候调用顺序也是 x y，而不是 y x，这和不同 block 是不一样的
    with saved_tensors_hooks(pack, unpack):
        out = save_two_tensors(x, y)
        loss = out.sum()

    print(f"[main] forward done; total pack calls = {pack_count}")
    print("[main] backward starts")
    loss.backward()
    print(f"[main] backward done; total unpack calls = {unpack_count}")
    # n=2048 时常见 allocated 约 112 MB：
    # x 的 leaf + hidden: 32 MB
    # y 的 leaf + hidden: 32 MB
    # out: 16 MB
    # 两个 leaf grad: 32 MB
    # 合计约 112 MB。peak 更高通常来自 backward 临时 tensor。
    mem("after backward")


DEMOS = {
    "1": demo_01_normal_saved_tensor,
    "2": demo_02_observe_saved_tensors_hooks,
    "3": demo_03_sync_cpu_offload,
    "4": demo_04_async_d2h_without_freeing_storage,
    "5": demo_05_xtuner_like_release_and_restore,
    "6": demo_06_save_multiple_tensors,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["all", *DEMOS.keys()], default="all")
    parser.add_argument("--n", type=int, default=2048, help="Tensor side length. Memory is roughly n*n*4 bytes per tensor.")
    args = parser.parse_args()

    print(f"torch={torch.__version__}")
    print(f"n={args.n}; one float32 tensor is about {args.n * args.n * 4 / 1024**2:.1f} MB")

    if args.demo == "all":
        for fn in DEMOS.values():
            fn(args.n)
    else:
        DEMOS[args.demo](args.n)


if __name__ == "__main__":
    main()
