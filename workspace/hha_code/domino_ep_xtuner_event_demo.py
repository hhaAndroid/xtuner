"""
Minimal 2-GPU demo for the current XTuner-style event/comm-stream overlap.

Run:
    torchrun --nproc_per_node=2 workspace/hha_code/domino_ep_xtuner_event_demo.py

This version is closer to xtuner/v1/module/dispatcher/torch_all2all.py:

1. Forward all-to-all is enqueued on a dedicated communication stream.
2. Forward compute stream waits on a forward_finished_event before using the
   communication output.
3. Backward hooks record when grad_output is ready.
4. The async all-to-all backward waits on that event, runs on the communication
   stream, and records a backward_finished_event.
5. A prehook on the upstream grad_fn waits on backward_finished_event before
   upstream backward compute consumes the gradient.
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
class A2AEvents:
    name: str
    forward_previous_event: torch.cuda.Event
    forward_finished_event: torch.cuda.Event
    backward_previous_event: torch.cuda.Event
    backward_finished_event: torch.cuda.Event


@dataclass
class StepResult:
    loss: torch.Tensor
    grads: torch.Tensor


class A2ASync(torch.autograd.Function):
    """Synchronous all-to-all in forward and backward. Used as reference."""

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


class AsyncA2AWithEvents(torch.autograd.Function):
    """Forward/backward all-to-all are both enqueued on the comm stream."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        events: A2AEvents,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        
        # async_op=False 容易误导。它只表示这个 collective 对当前 stream 的语义是同步/顺序的，但因为当前 stream 是 comm_stream，
        # CPU 退出 with 后并不等 comm_stream 真的跑完。也就是说，通信任务已经排进 comm_stream，但可能还在 GPU 上执行。
        # PyTorch caching allocator 后续如果想复用这块 storage，会确保 comm_stream 上在 record_stream 之前已经排队的 work 都完成
        # record_stream(comm_stream) => allocator，请把这个 tensor 的 storage 生命周期延长到 comm_stream 当前已排队任务完成之后
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(events.forward_previous_event) # 不同流肯定要等上一个依赖完成，否则结果不对
            dist.all_to_all_single(out, x, async_op=False)
            # record_stream 不是同步操作。它只是告诉 PyTorch allocator：
            # x/out 的 storage 后续还会被 comm_stream 上已经排队的通信读写，
            # 在 comm_stream 使用结束前不要提前复用这些显存。
            x.record_stream(comm_stream)
            out.record_stream(comm_stream)
            events.forward_finished_event.record(comm_stream)

        ctx.input_shape = tuple(x.shape)
        ctx.events = events
        ctx.comm_stream = comm_stream
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        grad_out = grad_out.contiguous()
        grad_x = torch.empty(ctx.input_shape, device=grad_out.device, dtype=grad_out.dtype)

        with torch.cuda.stream(ctx.comm_stream):
            ctx.comm_stream.wait_event(ctx.events.backward_previous_event)
            dist.all_to_all_single(grad_x, grad_out, async_op=False)
            # backward 同理：grad_out 是通信输入，grad_x 是通信输出。
            # 它们的 storage 需要保持到 comm_stream 上的 reverse all-to-all 完成。
            grad_out.record_stream(ctx.comm_stream)
            grad_x.record_stream(ctx.comm_stream)
            ctx.events.backward_finished_event.record(ctx.comm_stream)

        return grad_x, None, None


def make_a2a_events(name: str, x: torch.Tensor) -> A2AEvents:
    """Create events and attach the upstream backward wait point."""

    forward_previous_event = torch.cuda.Event()
    forward_previous_event.record(torch.cuda.current_stream())

    events = A2AEvents(
        name=name,
        forward_previous_event=forward_previous_event,
        forward_finished_event=torch.cuda.Event(),
        backward_previous_event=torch.cuda.Event(),
        backward_finished_event=torch.cuda.Event(),
    )

    if x.grad_fn is None:
        raise RuntimeError("make_a2a_events expects a non-leaf activation.")

    def wait_before_upstream_backward(_grad_outputs: tuple[torch.Tensor, ...]) -> None:
        torch.cuda.current_stream().wait_event(events.backward_finished_event)
        return None

    x.grad_fn.register_prehook(wait_before_upstream_backward)
    return events


def a2a_event_async(x: torch.Tensor, events: A2AEvents, comm_stream: torch.cuda.Stream) -> torch.Tensor:
    return AsyncA2AWithEvents.apply(x, events, comm_stream)


def wait_forward_comm(events: A2AEvents) -> None:
    torch.cuda.current_stream().wait_event(events.forward_finished_event)


def record_backward_ready(x: torch.Tensor, events: A2AEvents) -> torch.Tensor:
    """Record that x.grad is ready, so reverse all-to-all can start."""

    def record_event(_grad: torch.Tensor) -> None:
        events.backward_previous_event.record(torch.cuda.current_stream())
        return None

    x.register_hook(record_event) # 注册 backward 完成事件，让下游 backward 可以开始
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


def demo_xtuner_event_style(tokens: int, hidden: int, compute_iters: int, seed: int) -> StepResult:
    block, inputs = build_case(tokens, hidden, compute_iters, seed)
    comm_stream = torch.cuda.Stream()

    h0 = block.pre_compute(inputs[0])
    d0_events = make_a2a_events("dispatch_mb0", h0)
    d0 = a2a_event_async(h0, d0_events, comm_stream)
    d0 = record_backward_ready(d0, d0_events)

    # This forward compute overlaps with mb0 dispatch on the comm stream.
    h1 = block.pre_compute(inputs[1])
    d1_events = make_a2a_events("dispatch_mb1", h1)
    d1 = a2a_event_async(h1, d1_events, comm_stream)
    d1 = record_backward_ready(d1, d1_events)

    wait_forward_comm(d0_events)
    e0 = block.expert_compute(d0)
    c0_events = make_a2a_events("combine_mb0", e0)
    c0 = a2a_event_async(e0, c0_events, comm_stream)
    c0 = record_backward_ready(c0, c0_events)

    wait_forward_comm(d1_events)
    e1 = block.expert_compute(d1)
    c1_events = make_a2a_events("combine_mb1", e1)
    c1 = a2a_event_async(e1, c1_events, comm_stream)
    c1 = record_backward_ready(c1, c1_events)

    wait_forward_comm(c0_events)
    loss0 = block.post_compute(c0)
    wait_forward_comm(c1_events)
    loss1 = block.post_compute(c1)

    return finish(block, inputs, loss0 + loss1)


def warmup() -> None:
    demo_xtuner_event_style(tokens=8, hidden=8, compute_iters=1, seed=999)
    sync_all()


def compare_with_reference(rank: int, ref: StepResult, result: StepResult) -> None:
    loss_diff = (result.loss - ref.loss).abs().item()
    grad_max_diff = (result.grads - ref.grads).abs().max().item()
    grad_l2_diff = torch.linalg.vector_norm(result.grads - ref.grads).item()
    ok = loss_diff == 0.0 and grad_max_diff == 0.0
    log(
        rank,
        f"validate xtuner_event ok={ok} loss_diff={loss_diff:.3e} "
        f"grad_max_diff={grad_max_diff:.3e} grad_l2_diff={grad_l2_diff:.3e}",
    )
    if not ok:
        raise RuntimeError("xtuner_event does not match sync reference.")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank = setup_dist()
    log(rank, f"rank={rank}, local_rank={local_rank}, device=cuda:{local_rank}")
    log(rank, f"tokens={args.tokens}, hidden={args.hidden}, compute_iters={args.compute_iters}")
    warmup()

    try:
        ref = time_demo(
            rank,
            "sync reference",
            lambda: demo_reference_sync(args.tokens, args.hidden, args.compute_iters, args.seed),
        )
        result = time_demo(
            rank,
            "xtuner event style",
            lambda: demo_xtuner_event_style(args.tokens, args.hidden, args.compute_iters, args.seed),
        )
        compare_with_reference(rank, ref, result)
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
