"""
Basic 2-GPU EP communication demos without the no-op backward trick.

Run:
    torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_basic_demo.py --demo 1
    torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_basic_demo.py --demo 2
    torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_basic_demo.py --demo all

Demo 1:
    Two micro batches run one by one.  Both dispatch and combine all-to-all
    wait immediately.  No overlap.

Demo 2:
    Forward dispatch/combine use async all-to-all.  Backward is still sync.
    This isolates forward-only overlap before introducing the no-op trick.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_dist() -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires CUDA.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Please run with torchrun.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    except TypeError:
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("This demo expects exactly 2 GPUs.")
    return rank, local_rank


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def sync_all() -> None:
    torch.cuda.synchronize()
    dist.barrier()


class A2ASync(torch.autograd.Function):
    """Synchronous all-to-all in forward and backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = tuple(x.shape)
        x = x.contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, async_op=False)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor]:
        grad_out = grad_out.contiguous()
        grad_x = torch.empty(ctx.input_shape, device=grad_out.device, dtype=grad_out.dtype)
        dist.all_to_all_single(grad_x, grad_out, async_op=False)
        return (grad_x,)


def a2a_sync(x: torch.Tensor) -> torch.Tensor:
    return A2ASync.apply(x)


class AttachSyncBackward(torch.autograd.Function):
    """Forward output was produced elsewhere; backward still runs sync all-to-all."""

    @staticmethod
    def forward(ctx: Any, x_for_grad: torch.Tensor, precomputed_out: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = tuple(x_for_grad.shape)
        return precomputed_out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_out = grad_out.contiguous()
        grad_x = torch.empty(ctx.input_shape, device=grad_out.device, dtype=grad_out.dtype)
        dist.all_to_all_single(grad_x, grad_out, async_op=False)
        return grad_x, None


def a2a_forward_async_backward_sync(x: torch.Tensor) -> tuple[torch.Tensor, dist.Work]:
    x = x.contiguous()
    out = torch.empty_like(x)
    # 因为这个 a2a 是纯通信算子，没有梯度，不会保存到梯度图里面，所有需要 AttachSyncBackward 来计算反向梯度
    handle = dist.all_to_all_single(out, x, async_op=True)
    y = AttachSyncBackward.apply(x, out)
    return y, handle


class TinyBlock(nn.Module):
    def __init__(self, hidden: int, compute_iters: int):
        super().__init__()
        self.pre = nn.Linear(hidden, hidden, bias=False)
        self.expert = nn.Linear(hidden, hidden, bias=False)
        self.post = nn.Linear(hidden, 1, bias=False)
        self.compute_iters = compute_iters

    def pre_compute(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for _ in range(self.compute_iters):
            y = torch.relu(self.pre(y))
        return y

    def expert_compute(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for _ in range(self.compute_iters):
            y = torch.relu(self.expert(y))
        return y

    def post_compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(x).float().mean()


def build_case(tokens: int, hidden: int, compute_iters: int, seed: int) -> tuple[TinyBlock, list[torch.Tensor]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    block = TinyBlock(hidden, compute_iters).cuda() # 模拟
    inputs = [
        torch.randn(tokens, hidden, device="cuda", requires_grad=True),
        torch.randn(tokens, hidden, device="cuda", requires_grad=True),
    ]
    return block, inputs


def finish(block: TinyBlock, inputs: list[torch.Tensor], loss: torch.Tensor) -> torch.Tensor:
    loss.backward()
    checksum = loss.detach()
    for p in block.parameters():
        if p.grad is not None:
            checksum = checksum + p.grad.float().mean() * 0.0
    for x in inputs:
        if x.grad is not None:
            checksum = checksum + x.grad.float().mean() * 0.0
    return checksum


def demo_1_no_overlap(tokens: int, hidden: int, compute_iters: int, seed: int) -> torch.Tensor:
    block, inputs = build_case(tokens, hidden, compute_iters, seed)
    losses = []

    for x in inputs:
        # pre -> a2a -> expert -> a2a -> post
        h = block.pre_compute(x)
        dispatched = a2a_sync(h)
        expert_out = block.expert_compute(dispatched)
        combined = a2a_sync(expert_out)
        losses.append(block.post_compute(combined))

    return finish(block, inputs, sum(losses))


def demo_2_forward_overlap(tokens: int, hidden: int, compute_iters: int, seed: int) -> torch.Tensor:
    # 只考虑 forward overlap
    block, inputs = build_case(tokens, hidden, compute_iters, seed)

    h0 = block.pre_compute(inputs[0])
    # d0 和 h0 可以 overlap
    d0, d0_handle = a2a_forward_async_backward_sync(h0)

    h1 = block.pre_compute(inputs[1])
    # d1 和 h1 可以 overlap
    d1, d1_handle = a2a_forward_async_backward_sync(h1)

    d0_handle.wait()
    e0 = block.expert_compute(d0)
    c0, c0_handle = a2a_forward_async_backward_sync(e0)

    # This compute can overlap with mb0 combine.
    d1_handle.wait()
    e1 = block.expert_compute(d1)
    c1, c1_handle = a2a_forward_async_backward_sync(e1)

    c0_handle.wait()
    loss0 = block.post_compute(c0)
    c1_handle.wait()
    loss1 = block.post_compute(c1)

    return finish(block, inputs, loss0 + loss1)


def warmup() -> None:
    demo_1_no_overlap(tokens=8, hidden=8, compute_iters=1, seed=999)
    sync_all()


def time_demo(rank: int, name: str, fn: Callable[[], torch.Tensor]) -> None:
    sync_all()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = fn()
    end.record()
    sync_all()
    log(rank, f"{name:<28} loss={loss.item(): .6f} elapsed={start.elapsed_time(end):8.3f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["1", "2", "all"], default="all")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--compute-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank = setup_dist()
    log(rank, f"rank={rank}, local_rank={local_rank}, device=cuda:{local_rank}")
    log(rank, f"tokens={args.tokens}, hidden={args.hidden}, compute_iters={args.compute_iters}")
    warmup()

    demos = [
        ("demo 1: no overlap", demo_1_no_overlap),
        ("demo 2: fwd overlap", demo_2_forward_overlap),
    ]
    selected = demos if args.demo == "all" else [demos[int(args.demo) - 1]]

    try:
        for name, fn in selected:
            time_demo(rank, name, lambda fn=fn: fn(args.tokens, args.hidden, args.compute_iters, args.seed))
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
