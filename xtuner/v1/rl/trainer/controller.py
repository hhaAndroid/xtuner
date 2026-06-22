import heapq
import math
import os
import random
from typing import Literal, TypedDict

import ray
import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .worker import TrainingWorker, WorkerLogItem


TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantage: float
    rollout_logprobs: torch.Tensor | None


class TrainingController:
    def __init__(self, workers: list[TrainingWorker]) -> None:
        self.workers = workers
        self.logger = get_logger()

    # TODO(hha): 这个逻辑不够通用，应该复用 sft 函数，从而支持 expand soft pack
    def _get_balanced_pack_infos(
        self,
        num_tokens: list[int],
        pack_max_length: int,
        num_bins: int,
        dp_size: int,
    ) -> list[dict]:
        """Distribute sequences into ``num_bins`` packs as evenly as possible.

        Sequences are assigned with the Longest-Processing-Time (LPT) heuristic: they are processed from longest
        to shortest and each is placed into the currently least-loaded pack that still has room. ``num_bins`` is the
        number of packs the train workers need (``optimizer_steps * dp_size``); when the data does not fit under the
        ``pack_max_length`` capacity, ``num_bins`` is grown in steps of ``dp_size`` so the result stays divisible by
        the data parallel size.

        Args:
            num_tokens (list[int]): Token count of each sequence.
            pack_max_length (int): Maximum number of tokens allowed in one pack.
            num_bins (int): Initial number of packs to distribute into.
            dp_size (int): Data parallel size; ``num_bins`` is grown by this step when the capacity is exceeded.

        Returns:
            list[dict]: One entry per pack with the assigned sequence ``indices`` and the ``longest`` sequence
            length in the pack. Some packs may be empty when the number of sequences is smaller than ``num_bins``.
        """
        order = sorted(range(len(num_tokens)), key=lambda i: num_tokens[i], reverse=True)
        while True:
            heap = [(0, b) for b in range(num_bins)]  # (load, bin_idx)
            heapq.heapify(heap)
            bins: list[list[int]] = [[] for _ in range(num_bins)]
            feasible = True
            for i in order:
                load, b = heap[0]  # the least-loaded pack has the most remaining room
                if load + num_tokens[i] <= pack_max_length:
                    heapq.heapreplace(heap, (load + num_tokens[i], b))
                    bins[b].append(i)
                else:
                    # Even the emptiest pack cannot fit this sequence: we need more packs.
                    feasible = False
                    break
            if feasible:
                break
            num_bins += dp_size
        return [{"indices": b, "longest": int(max((num_tokens[i] for i in b), default=0))} for b in bins]

    # TODO(hha): 这个逻辑不够通用，和模型绑定了
    def _packing(self, data_batches, pack_max_length, language_cfg, dp_size, optimizer_steps):
        num_tokens = [data["seq_ctx"].input_ids.numel() for data in data_batches]
        # Train workers perform `optimizer_steps` updates per dp rank, so they need at least
        # `optimizer_steps * dp_size` packs globally. Grow the target up to the capacity lower bound
        # (total tokens / pack_max_length) and keep it divisible by `dp_size`.
        min_bins = optimizer_steps * dp_size
        cap_bins = math.ceil(math.ceil(sum(num_tokens) / pack_max_length) / dp_size) * dp_size
        num_bins = max(min_bins, cap_bins)
        if len(num_tokens) < num_bins:
            self.logger.warning(
                f"Number of valid sequences ({len(num_tokens)}) is smaller than the required number of packs "
                f"({num_bins} = optimizer_steps {optimizer_steps} * dp_size {dp_size}). Some packs will contain "
                "only padding, leading to no-op optimizer steps. Consider providing more data or reducing "
                "optimizer_steps."
            )
        pack_infos = self._get_balanced_pack_infos(num_tokens, pack_max_length, num_bins, dp_size)
        packed_data_batches = []

        is_qwen3_vl = False
        if len(data_batches[0]["seq_ctx"].position_ids.shape) == 3:
            is_qwen3_vl = True

        has_rollout_routed_experts = False
        if data_batches[0]["seq_ctx"].rollout_routed_experts is not None:
            assert language_cfg is not None
            has_rollout_routed_experts = True
            n_routed_experts = language_cfg.n_routed_experts

        for pack_info in pack_infos:
            indices = pack_info["indices"]
            total_len = sum([data_batches[i]["seq_ctx"].input_ids.shape[1] for i in indices])
            pad_len = pack_max_length - total_len
            seq_ctx_list = [data_batches[i]["seq_ctx"] for i in indices]
            label_list = [data_batches[i]["shifted_labels"] for i in indices]
            advantage_list = [data_batches[i]["advantage"] for i in indices]

            rollout_logprobs_list = None
            if "rollout_logprobs" in data_batches[0] and data_batches[0]["rollout_logprobs"] is not None:
                rollout_logprobs_list = [data_batches[i]["rollout_logprobs"] for i in indices]

            if pad_len > 0:
                # Reduce the attn calculation time by using multiple short sequence packs
                pad_tokens = tuple(
                    torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
                    for _ in range(pad_len // 1024)
                )
                if pad_len % 1024 > 0:
                    pad_tokens = pad_tokens + (
                        torch.zeros(1, pad_len % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"),
                    )
                pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")
                pad_seq_ctx.num_padding = pad_len
                pad_labels = torch.full(
                    (1, pad_len),
                    -100,
                    dtype=data_batches[0]["shifted_labels"].dtype,
                    device=data_batches[0]["shifted_labels"].device,
                )
                pad_advantages = [-100] * pad_len
                if is_qwen3_vl:
                    _position_ids_list = []
                    for pad_token in pad_tokens:
                        _position_ids = torch.arange(pad_token.size(-1)).view(1, 1, -1).expand(3, 1, -1)
                        _position_ids_list.append(_position_ids)
                    pad_seq_ctx.position_ids = torch.cat(_position_ids_list, dim=-1)

                if has_rollout_routed_experts:
                    pad_rand_index = torch.randint(low=0, high=n_routed_experts, size=(pad_len, 1, 1))
                    pad_seq_ctx.rollout_routed_experts = pad_rand_index

                seq_ctx_list.append(pad_seq_ctx)
                label_list.append(pad_labels)
                advantage_list.append(pad_advantages)
                if rollout_logprobs_list is not None:
                    pad_rollout_logprobs = torch.zeros(
                        1,
                        pad_len,
                        dtype=data_batches[0]["rollout_logprobs"].dtype,
                        device=data_batches[0]["shifted_labels"].device,
                    )
                    rollout_logprobs_list.append(pad_rollout_logprobs)

            seq_ctx = SequenceContext.cat(seq_ctx_list)
            shifted_labels = torch.cat(label_list, dim=1)  # (1, max_len)
            advantage_flat = [item for sublist in advantage_list for item in sublist]
            advantages = torch.tensor(advantage_flat, dtype=torch.float32).unsqueeze(0)

            rollout_logprobs = None
            if rollout_logprobs_list is not None:
                rollout_logprobs = torch.cat(rollout_logprobs_list, dim=1)  # (1, max_len)

            packed_data_batches.append(
                {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": shifted_labels,
                    "advantages": advantages,
                    "rollout_logprobs": rollout_logprobs,
                }
            )

        # `_get_balanced_pack_infos` sorts sequences by length, so the resulting pack order is correlated with
        # sequence length. Since the worker slices packs into optimizer-step groups by position, that ordering
        # would make each optimizer step a length-stratified (hence biased) mini-batch. Shuffle the balanced packs
        # so every step is a random sample; the packs are load-balanced, so shuffling preserves the balance.
        if not XTUNER_DETERMINISTIC:
            random.shuffle(packed_data_batches)
        return packed_data_batches

    def _grouped_by_max_length(self, packed_data_batches):
        # sort 过后可能第一个 batch 会有很多 pad tokens，因为最后一个 pack 可能只有少量真实数据。
        # 比如组成了 16 个 pack，第 16 个 pack 可能只有几条真实数据，剩下的都是 pad tokens。
        # 排序后这条 pack 会被放在最前面，导致 rank0 的第一个 step 消耗的有效 token 数往往少于其他 rank，是正常现象。
        return sorted(packed_data_batches, key=lambda x: x["seq_ctx"].max_length_q, reverse=True)

    def fit(
        self, data_batches: list[ColateItem], pack_max_length: int, rollout_idx: int, optimizer_steps: int
    ) -> list[WorkerLogItem]:
        language_cfg = None
        if data_batches[0]["seq_ctx"].rollout_routed_experts is not None:
            model_cfg = ray.get(self.workers[0].get_model_cfg.remote())  # type: ignore[attr-defined]
            language_cfg = model_cfg
            if isinstance(model_cfg, BaseComposeConfig):
                language_cfg = model_cfg.text_config

        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]
        dp_size = len(self.workers) // data_replicate_size

        packed_data_batches = self._packing(data_batches, pack_max_length, language_cfg, dp_size, optimizer_steps)
        # packed_data_batches = self._grouped_by_max_length(packed_data_batches)

        # `_packing` guarantees a pack count that is divisible by `dp_size` and provides at least
        # `optimizer_steps` packs per dp rank, so no extra padding / distribution alignment is needed here.
        num_packed_data_batches = len(packed_data_batches)
        assert (
            num_packed_data_batches % dp_size == 0 and num_packed_data_batches >= optimizer_steps * dp_size
        ), (
            f"Unexpected packed batch count {num_packed_data_batches} for dp_size {dp_size} and "
            f"optimizer_steps {optimizer_steps}."
        )

        handles = []
        for worker_idx, worker in enumerate(self.workers):
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=packed_data_batches[(worker_idx // data_replicate_size) :: dp_size],
                    rollout_idx=rollout_idx,
                )
            )
        try:
            log_infos = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        finally:
            # Free pixel_values ObjectRefs put by Sampler.put_to_ray. The
            # workers' fit() already ray.get'd them into local tensors, so
            # plasma copies are no longer needed and otherwise leak across
            # every train step (one ref per packed sample, including padding).
            free_pixel_value_refs: list[ray.ObjectRef] = []
            for data in packed_data_batches:
                if data["seq_ctx"].pixel_values is not None:
                    free_pixel_value_refs.extend(data["seq_ctx"].pixel_values)
            if len(free_pixel_value_refs) > 0:
                free_object_refs(free_pixel_value_refs)
            del packed_data_batches
        return log_infos

    def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
        if target == "model":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "optimizer":
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "all":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        return

    def onload(self, target: Literal["model", "optimizer", "all"] = "all"):
        """Onload the model or optimizer of the training workers."""
        if target == "model":
            ray.get([worker.onload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "optimizer":
            ray.get([worker.onload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "all":
            ray.get([worker.onload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
            ray.get([worker.onload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        return

    def update_rollout_info(self, info_dict):
        ray.get([worker.update_rollout_info.remote(**info_dict) for worker in self.workers])  # type: ignore[attr-defined]

    def set_train_rollout_mode(self, train_rollout_mode: str):
        ray.get([worker.set_train_rollout_mode.remote(train_rollout_mode) for worker in self.workers])

    def update_weights(self):
        """Update the weights of the training workers."""
        handles = [worker.update_weights.remote() for worker in self.workers]
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        handles = [worker.save_hf.remote(hf_dir, save_dtype) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training workers from the checkpoint."""
        handles = [worker.resume.remote(load_checkpoint_cfg) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def save(self, dcp_dir: str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training workers."""
        handles = [worker.save.remote(dcp_dir, no_save_optimizer) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return
