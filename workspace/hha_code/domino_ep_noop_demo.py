"""
Minimal 2-GPU demo for the Domino / old-XTuner no-op backward trick.

Run:
    torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_noop_demo.py

This file intentionally contains only the no-op idea:

1. Forward all-to-all is launched async.
2. A custom autograd node launches the reverse all-to-all in backward.
3. A NoOp node does nothing in forward, but waits for that backward handle later.

The forward code decides where the backward launch/wait nodes sit in the graph.
Autograd then executes them in reverse order, so backward communication can be
overlapped with unrelated backward compute.
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
    handle: dist.Work
    keep_alive: tuple[Any, ...]

    def wait(self) -> None:
        self.handle.wait()
        self.keep_alive = ()


@dataclass
class StepResult:
    loss: torch.Tensor
    grads: torch.Tensor


def a2a_forward_async(x: torch.Tensor) -> tuple[torch.Tensor, AsyncComm]:
    x = x.contiguous()
    out = torch.empty_like(x)
    handle = dist.all_to_all_single(out, x, async_op=True)
    return out, AsyncComm(handle=handle, keep_alive=(x, out))


class A2ASync(torch.autograd.Function):
    """Synchronous all-to-all in forward and backward. Used as the numerical reference."""

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


class LaunchBackwardA2A(torch.autograd.Function):
    """Forward returns extra_input. Backward launches reverse all-to-all async.

    This mirrors old XTuner's _AllToAll_BWD:
      forward(inputs, ..., extra_input) -> extra_input
      backward(grad_extra_input) -> async reverse all-to-all grad for inputs
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_for_grad: torch.Tensor,
        handle_table: dict[str, AsyncComm],
        key: str,
        extra_input: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_shape = tuple(x_for_grad.shape)
        ctx.handle_table = handle_table
        ctx.key = key
        return extra_input

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        grad_out = grad_out.contiguous()
        grad_x = torch.empty(ctx.input_shape, device=grad_out.device, dtype=grad_out.dtype)
        handle = dist.all_to_all_single(grad_x, grad_out, async_op=True)
        ctx.handle_table[ctx.key] = AsyncComm(handle=handle, keep_alive=(grad_out, grad_x))
        return grad_x, None, None, None


def launch_backward_a2a(
    x_for_grad: torch.Tensor,
    handle_table: dict[str, AsyncComm],
    key: str,
    extra_input: torch.Tensor,
) -> torch.Tensor:
    return LaunchBackwardA2A.apply(x_for_grad, handle_table, key, extra_input)


def a2a_fwd_bwd_overlap(
    x: torch.Tensor,
    handle_table: dict[str, AsyncComm],
    key: str,
    *,
    is_forward: bool,
    extra_input: torch.Tensor | None = None,
) -> tuple[AsyncComm | None, torch.Tensor]:
    """Old-XTuner-shaped helper.

    is_forward=True:
        launch real forward all-to-all and return (handle, comm_output).

    is_forward=False:
        do not launch forward communication.  Return extra_input wrapped by a
        backward-launch autograd node.  During backward, that node launches the
        reverse all-to-all and stores its handle in handle_table[key].
    """

    if is_forward:
        out, handle = a2a_forward_async(x)
        return handle, out

    if extra_input is None:
        raise RuntimeError("extra_input is required when is_forward=False.")
    out = launch_backward_a2a(x, handle_table, key, extra_input)
    return None, out


# 前向时候存全局 handle dict 对象和对应 key
# 反向时候根据 key 找到 handle 对象，并等待
class NoOpWait(torch.autograd.Function):
    """Forward is identity. Backward waits for an async handle launched elsewhere."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, handle_table: dict[str, AsyncComm], key: str) -> torch.Tensor:
        ctx.handle_table = handle_table
        ctx.key = key
        return x

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        handle = ctx.handle_table.pop(ctx.key)
        handle.wait()
        return grad_out, None, None


def no_op_wait(x: torch.Tensor, handle_table: dict[str, AsyncComm], key: str) -> torch.Tensor:
    return NoOpWait.apply(x, handle_table, key)


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


def collect_grad_vector(block: TinyBlock, inputs: list[torch.Tensor]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for parameter in block.parameters():
        assert parameter.grad is not None
        pieces.append(parameter.grad.detach().float().flatten())
    for x in inputs:
        assert x.grad is not None
        pieces.append(x.grad.detach().float().flatten())
    return torch.cat(pieces)


def finish(block: TinyBlock, inputs: list[torch.Tensor], loss: torch.Tensor) -> StepResult:
    loss.backward()
    return StepResult(loss=loss.detach(), grads=collect_grad_vector(block, inputs))


def demo_reference_sync(tokens: int, hidden: int, compute_iters: int, seed: int) -> StepResult:
    block, inputs = build_case(tokens, hidden, compute_iters, seed)
    losses = []

    for x in inputs:
        h = block.pre_compute(x)
        d = a2a_sync(h)
        e = block.expert_compute(d)
        c = a2a_sync(e)
        losses.append(block.post_compute(c))

    return finish(block, inputs, sum(losses))


# forward: pre0 -> dispatch0 -> pre1 -> dispatch1 -> expert0 -> combine0 -> expert1 -> combine1 -> post0 -> post1
# demo_noop_c1_e1_c0: combine1_bwd -> expert1_bwd -> combine0_bwd
# demo_noop_c1_c0_e1: combine1_bwd -> combine0_bwd -> expert1_bwd
def _prepare_until_combine0(
    tokens: int,
    hidden: int,
    compute_iters: int,
    seed: int,
) -> tuple[TinyBlock, list[torch.Tensor], dict[str, AsyncComm], AsyncComm | None, torch.Tensor, AsyncComm | None, torch.Tensor, torch.Tensor]:
    block, inputs = build_case(tokens, hidden, compute_iters, seed)
    handles: dict[str, AsyncComm] = {}

    h0 = block.pre_compute(inputs[0])

    # 注意 "dispatch_mb0" 这个 key 很关键
    # 前向时候只是保存全局 handle dict 对象和对应 key
    # 然后会在 launch_backward_a2a(h0, handles, "dispatch_mb0", d0_raw) 中执行 handles[key]=需要同步的 handle 句柄
    # 从而实现在执行 backward 时候，no op 这行会自动找到对应 key 的 handle 句柄，并等待，实现 backward overlap
    h0 = no_op_wait(h0, handles, "dispatch_mb0") # only for backward async await
    

    # 这里 d0 和 h0 可以 overlap
    d0_comm, d0 = a2a_fwd_bwd_overlap( # launch forward all-to-all
        h0,
        handles,
        "dispatch_mb0",
        is_forward=True,
    )

    # This forward compute overlaps with mb0 dispatch.
    h1 = block.pre_compute(inputs[1])


    # Same shape as old XTuner:
    #   _, global_input_tokens = moe_all_to_all_pre(..., is_forward=False, extra_input=global_input_tokens)
    # Forward value is still d0; backward launches reverse dispatch for mb0.
    _, d0 = a2a_fwd_bwd_overlap( # 添加 a2a backward 计算过程 + 插入 dispatch_mb0 的 backward 句柄
        h0,
        handles,
        "dispatch_mb0",
        is_forward=False,
        extra_input=d0,
    )

    h1 = no_op_wait(h1, handles, "dispatch_mb1") # only for backward async await


    # 这里 d1 和 h1 可以 overlap
    d1_comm, d1 = a2a_fwd_bwd_overlap( # launch forward all-to-all
        h1,
        handles,
        "dispatch_mb1",
        is_forward=True,
    )

    assert d0_comm is not None
    d0_comm.wait()
    e0 = block.expert_compute(d0)


    # Forward value is still d1; backward launches reverse dispatch for mb1.
    _, d1 = a2a_fwd_bwd_overlap(
        h1,
        handles,
        "dispatch_mb1",
        is_forward=False,
        extra_input=d1,
    )

    e0 = no_op_wait(e0, handles, "combine_mb0")
    c0_comm, c0 = a2a_fwd_bwd_overlap(
        e0,
        handles,
        "combine_mb0",
        is_forward=True,
    )
    return block, inputs, handles, c0_comm, c0, d1_comm, d1, e0


# noop 版本 A：把 BwdLaunch_C0 插在 expert1 前面。
# 直观看 backward: combine1_bwd -> expert1_bwd -> combine0_bwd

## forward 相同位置表示两个 micro-batch 可以 overlap
#  MB0: Pre0 ── Dispatch0 ── Expert0 ── Combine0 ── Post0
#                    │           │         │           │ 
#  MB1:          Pre1 ── Dispatch1 ─────── Expert1 ── Combine1 ── Post1

## backward 相同位置表示两个 micro-batch 可以 overlap
#   MB0:        Post0 ─────── Combine0 ─────── Expert0 ── Dispatch0 ── Pre0
#                   │            │             │             │
#   MB1: Post1 ── Combine1 ── Expert1 ── Dispatch1 ────── Pre1
def demo_noop_c1_e1_c0(tokens: int, hidden: int, compute_iters: int, seed: int) -> StepResult:
    block, inputs, handles, c0_comm, c0, d1_comm, d1, e0 = _prepare_until_combine0(
        tokens, hidden, compute_iters, seed
    )

    # Forward value is still c0; backward launches reverse combine for mb0.
    _, c0 = a2a_fwd_bwd_overlap(
        e0,
        handles,
        "combine_mb0",
        is_forward=False,
        extra_input=c0,
    )

    assert d1_comm is not None
    d1_comm.wait()
    e1 = block.expert_compute(d1)


    e1 = no_op_wait(e1, handles, "combine_mb1")
    c1_comm, c1 = a2a_fwd_bwd_overlap(
        e1,
        handles,
        "combine_mb1",
        is_forward=True,
    )

    # Forward value is still c1; backward launches reverse combine for mb1.
    # 后面没有通信算子了，所以插入到这里
    _, c1 = a2a_fwd_bwd_overlap(
        e1,
        handles,
        "combine_mb1",
        is_forward=False,
        extra_input=c1,
    )
    # bwd 时候 block.post_compute(c0) 和 combine_mb1 a2a_fwd_bwd_overlap 是可以 overlap 的，
    # 因为在 cpu launch 都发起情况下，计算流和通信流可以同时跑
    assert c0_comm is not None
    c0_comm.wait()
    loss0 = block.post_compute(c0)
    assert c1_comm is not None
    c1_comm.wait()
    loss1 = block.post_compute(c1)

    return finish(block, inputs, loss0 + loss1)


# noop 版本 B：把 BwdLaunch_C0 插在 expert1 后面。
# 直观看 backward: combine1_bwd -> combine0_bwd -> expert1_bwd


# 和上面的区别主要在于后半段，

## forward 相同位置表示两个 micro-batch 可以 overlap

#  MB0: Pre0 ── Dispatch0 ── Expert0 ── Combine0 ─── Post0
#                    │           │         │           │      
#  MB1:          Pre1 ── Dispatch1 ── Expert1 ── Combine1 ── Post1

## backward 相同位置表示两个 micro-batch 可以 overlap
#   MB0:        Post0 ── Combine0 ───────── Expert0 ── Dispatch0 ── Pre0
#                   │         │                │             │
#   MB1: Post1 ── Combine1 ── Expert1 ── Dispatch1 ────── Pre1

def demo_noop_c1_c0_e1(tokens: int, hidden: int, compute_iters: int, seed: int) -> StepResult:
    block, inputs, handles, c0_comm, c0, d1_comm, d1, e0 = _prepare_until_combine0(
        tokens, hidden, compute_iters, seed
    )

    assert d1_comm is not None
    d1_comm.wait()
    e1 = block.expert_compute(d1)

    # Forward value is still c0; backward launches reverse combine for mb0.
    _, c0 = a2a_fwd_bwd_overlap(
        e0,
        handles,
        "combine_mb0",
        is_forward=False,
        extra_input=c0,
    )

    e1 = no_op_wait(e1, handles, "combine_mb1")
    c1_comm, c1 = a2a_fwd_bwd_overlap(
        e1,
        handles,
        "combine_mb1",
        is_forward=True,
    )

    # Forward value is still c1; backward launches reverse combine for mb1.
    _, c1 = a2a_fwd_bwd_overlap(
        e1,
        handles,
        "combine_mb1",
        is_forward=False,
        extra_input=c1,
    )

    assert c0_comm is not None
    c0_comm.wait()
    loss0 = block.post_compute(c0)
    assert c1_comm is not None
    c1_comm.wait()
    loss1 = block.post_compute(c1)

    return finish(block, inputs, loss0 + loss1)


def warmup() -> None:
    demo_noop_c1_e1_c0(tokens=8, hidden=8, compute_iters=1, seed=999)
    sync_all()


def compare_with_reference(rank: int, name: str, ref: StepResult, result: StepResult) -> None:
    loss_diff = (result.loss - ref.loss).abs().item()
    grad_max_diff = (result.grads - ref.grads).abs().max().item()
    grad_l2_diff = torch.linalg.vector_norm(result.grads - ref.grads).item()
    ok = loss_diff == 0.0 and grad_max_diff == 0.0
    log(
        rank,
        f"validate {name:<16} ok={ok} loss_diff={loss_diff:.3e} "
        f"grad_max_diff={grad_max_diff:.3e} grad_l2_diff={grad_l2_diff:.3e}",
    )
    if not ok:
        raise RuntimeError(f"{name} does not match sync reference.")


def time_demo(rank: int, name: str, fn: Callable[[], StepResult]) -> StepResult:
    sync_all()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    sync_all()
    grad_norm = torch.linalg.vector_norm(result.grads).item()
    log(rank, f"{name:<28} loss={result.loss.item(): .6f} grad_norm={grad_norm:.6e} elapsed={start.elapsed_time(end):8.3f} ms")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--compute-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--demo",
        choices=["c1_e1_c0", "c1_c0_e1", "all"],
        default="all",
        help="Which combine backward-launch insertion point to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank = setup_dist()
    log(rank, f"rank={rank}, local_rank={local_rank}, device=cuda:{local_rank}")
    log(rank, f"tokens={args.tokens}, hidden={args.hidden}, compute_iters={args.compute_iters}")
    warmup()

    try:
        demos = [
            ("c1_e1_c0", demo_noop_c1_e1_c0),
            ("c1_c0_e1", demo_noop_c1_c0_e1),
        ]
        selected = demos if args.demo == "all" else [item for item in demos if item[0] == args.demo]
        ref = time_demo(
            rank,
            "sync reference",
            lambda: demo_reference_sync(args.tokens, args.hidden, args.compute_iters, args.seed),
        )
        for name, fn in selected:
            result = time_demo(
                rank,
                f"noop {name}",
                lambda fn=fn: fn(args.tokens, args.hidden, args.compute_iters, args.seed),
            )
            compare_with_reference(rank, name, ref, result)
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
