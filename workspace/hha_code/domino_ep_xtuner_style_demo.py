"""
Minimal 2-GPU demo for the current XTuner-style overlap implementation.

Run:
    torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_xtuner_style_demo.py

Compare this with domino_ep_noop_demo.py:

No-op style:
    Insert an explicit NoOp autograd Function in forward.  Its backward waits
    for an async communication handle.

XTuner-style:
    Do not insert a NoOp Function.  Register a backward pre-hook on an existing
    activation's grad_fn.  The hook waits for an async communication token.

The communication itself is still represented by a custom autograd Function:
forward returns the precomputed async all-to-all output; backward launches the
reverse all-to-all asynchronously and stores the handle in a token object.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
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


@dataclass
class AsyncComm:
    work: dist.Work
    keep_alive: tuple[Any, ...]

    def wait(self) -> None:
        self.work.wait()
        self.keep_alive = ()


@dataclass
class CommToken:
    name: str
    comm: AsyncComm | None = None
    # Optional removable handle returned by register_prehook.
    # The hook is already active after registration; saving this handle is only
    # useful if we later want to call hook_handle.remove().  This demo never
    # removes hooks, so this field is not part of the overlap logic.
    hook_handle: Any | None = None

    def set(self, comm: AsyncComm) -> None:
        self.comm = comm

    def wait(self) -> None:
        if self.comm is None:
            raise RuntimeError(f"Token {self.name!r} was waited before its backward comm was launched.")
        self.comm.wait()
        self.comm = None


def a2a_forward_async(x: torch.Tensor) -> tuple[torch.Tensor, AsyncComm]:
    x = x.contiguous()
    out = torch.empty_like(x)
    work = dist.all_to_all_single(out, x, async_op=True)
    return out, AsyncComm(work=work, keep_alive=(x, out))


class AsyncA2AWithToken(torch.autograd.Function):
    """Forward returns precomputed_out. Backward launches reverse all-to-all async."""

    @staticmethod
    def forward(ctx: Any, x_for_grad: torch.Tensor, token: CommToken, precomputed_out: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = tuple(x_for_grad.shape)
        ctx.token = token
        return precomputed_out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        grad_out = grad_out.contiguous()
        grad_x = torch.empty(ctx.input_shape, device=grad_out.device, dtype=grad_out.dtype)
        work = dist.all_to_all_single(grad_x, grad_out, async_op=True)
        ctx.token.set(AsyncComm(work=work, keep_alive=(grad_out, grad_x)))
        return grad_x, None, None


def attach_async_a2a_backward(x_for_grad: torch.Tensor, token: CommToken, precomputed_out: torch.Tensor) -> torch.Tensor:
    return AsyncA2AWithToken.apply(x_for_grad, token, precomputed_out)


def register_wait_prehook(x: torch.Tensor, token: CommToken) -> torch.Tensor:
    """Attach a wait point to an existing autograd node.

    During backward, PyTorch calls this pre-hook immediately before x.grad_fn's
    own backward.  That is the role NoOp.backward played in the older version,
    but without adding another autograd Function node.
    """

    if x.grad_fn is None:
        raise RuntimeError("register_wait_prehook expects a non-leaf activation.")

    def wait_before_this_node(_grad_outputs: tuple[torch.Tensor, ...]) -> None:
        token.wait()
        return None

    # Saving the returned handle is optional.  register_prehook has already
    # attached the hook to x.grad_fn; the handle only enables manual removal.
    token.hook_handle = x.grad_fn.register_prehook(wait_before_this_node)
    return x


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
    block = TinyBlock(hidden, compute_iters).cuda()
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


def demo_xtuner_style(tokens: int, hidden: int, compute_iters: int, seed: int) -> torch.Tensor:
    block, inputs = build_case(tokens, hidden, compute_iters, seed)

    h0 = block.pre_compute(inputs[0])
    token_d0 = CommToken("dispatch_mb0") # 类似 no op, 只有 wait 功能
    h0 = register_wait_prehook(h0, token_d0)
    d0_raw, d0_comm = a2a_forward_async(h0)

    # This forward compute overlaps with mb0 dispatch.
    h1 = block.pre_compute(inputs[1])

    # In backward, this node launches reverse dispatch for mb0 and stores the handle in token_d0.
    d0 = attach_async_a2a_backward(h0, token_d0, d0_raw)

    token_d1 = CommToken("dispatch_mb1")
    h1 = register_wait_prehook(h1, token_d1)
    d1_raw, d1_comm = a2a_forward_async(h1)

    d0_comm.wait()
    e0 = block.expert_compute(d0)

    # In backward, this node launches reverse dispatch for mb1.
    d1 = attach_async_a2a_backward(h1, token_d1, d1_raw)

    token_c0 = CommToken("combine_mb0")
    e0 = register_wait_prehook(e0, token_c0)
    c0_raw, c0_comm = a2a_forward_async(e0)

    d1_comm.wait()
    e1 = block.expert_compute(d1)

    # In backward, this node launches reverse combine for mb0.
    c0 = attach_async_a2a_backward(e0, token_c0, c0_raw)

    token_c1 = CommToken("combine_mb1")
    e1 = register_wait_prehook(e1, token_c1)
    c1_raw, c1_comm = a2a_forward_async(e1)

    # In backward, this node launches reverse combine for mb1.
    c1 = attach_async_a2a_backward(e1, token_c1, c1_raw)

    c0_comm.wait()
    loss0 = block.post_compute(c0)
    c1_comm.wait()
    loss1 = block.post_compute(c1)

    return finish(block, inputs, loss0 + loss1)


def warmup() -> None:
    demo_xtuner_style(tokens=8, hidden=8, compute_iters=1, seed=999)
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

    try:
        time_demo(
            rank,
            "demo: xtuner style",
            lambda: demo_xtuner_style(args.tokens, args.hidden, args.compute_iters, args.seed),
        )
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
