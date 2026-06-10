import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional, cast
from uuid import uuid4

import ray
import torch
from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.utils import XTUNER_DETERMINISTIC
from xtuner.v1.utils.logger import get_logger


logger = get_logger(__name__)


def _json_dumps_stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _otel_run_id() -> str | None:
    return (
        os.environ.get("XTUNER_OTEL_RUN_ID")
        or os.environ.get("RUN_ID")
        or os.environ.get("MODEL_NAME")
        or os.environ.get("WORK_DIR")
    )


def _build_otel_attrs(data: RolloutState, group_index: int, repeat_index: int, prompt_repeat_k: int) -> dict[str, Any]:
    data_source = data.data_source
    if data.message_uid is not None:
        case_payload = {"data_source": data_source, "message_uid": data.message_uid, "repeat_index": repeat_index}
    else:
        case_payload = {"data_source": data_source, "message": data.message, "repeat_index": repeat_index}
    case_hash = hashlib.sha1(_json_dumps_stable(case_payload).encode("utf-8")).hexdigest()[:16]

    attrs: dict[str, Any] = {
        "case.id": case_hash,
        "sample.group_index": group_index,
        "sample.repeat_index": repeat_index,
        "sample.repeat_k": prompt_repeat_k,
    }
    run_id = _otel_run_id()
    if run_id:
        attrs["run.id"] = run_id
    if data.message_uid is not None:
        attrs["sample.message_uid"] = data.message_uid
    if data_source is not None:
        attrs["sample.data_source"] = _json_dumps_stable(data_source) if isinstance(data_source, dict) else str(data_source)
    return attrs


class SamplerConfig(BaseModel):
    """Configuration for sampling prompts into rollout groups.

    ``SamplerConfig`` wraps a dataloader configuration and controls how many
    rollout samples are generated from the same prompt. The sampler first tries
    to reuse eligible replay-buffer samples and falls back to the dataloader
    when no reusable sample is available.

    Args:
        dataloader_cfg (DataloaderConfig): Dataset dataloader configuration
            that yields ``RolloutState`` prompts.
        prompt_repeat_k (int): Number of rollout samples to create for each
            prompt. This is commonly the GRPO group size. Defaults to 1.

    **Examples:**

    Example sampler for an 8-response group::

        config = SamplerConfig(
            dataloader_cfg=dataloader_cfg,
            prompt_repeat_k=8,
        )
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    dataloader_cfg: DataloaderConfig
    prompt_repeat_k: int = 1

    def build(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str, replay_buffer: ReplayBuffer
    ) -> "Sampler":
        if isinstance(tokenizer, str):
            tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            tokenizer_obj = tokenizer
        dataloader = self.dataloader_cfg.build(
            tokenizer=tokenizer_obj, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        return Sampler(dataloader=dataloader, prompt_repeat_k=self.prompt_repeat_k, replay_buffer=replay_buffer)


# TODO: The best solution is to put it in the fake_collator,
# but it will cause a deadlock problem, so it is temporarily placed here.
# The best solution should be to start the dataloader using spawn.
def put_to_ray(data: RolloutState) -> RolloutState:
    if hasattr(data, "mm_info") and data.mm_info is not None:
        pixel_values = data.mm_info.get("pixel_values", None)
        if pixel_values is not None:
            data.mm_info["pixel_values"] = ray.put(pixel_values)
    return data


class _DatasetSampler:
    def __init__(self, dataloader: Dataloader, prompt_repeat_k: int):
        self.dataloader = dataloader
        self.dataloader_iter: Optional[Iterator] = None
        self.cur_epoch = 0
        self.prompt_repeat_k = prompt_repeat_k
        self._consumed_samples: int = 0

    def __len__(self) -> int:
        return len(self.dataloader)

    def sample_from_dataloader(self) -> list[RolloutState]:
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)
        assert self.dataloader_iter is not None
        try:
            data = cast(RolloutState, next(self.dataloader_iter)[0])
            data = put_to_ray(data)

        except StopIteration:
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = cast(RolloutState, next(self.dataloader_iter)[0])
            data = put_to_ray(data)

        if XTUNER_DETERMINISTIC:
            message_uid = self._consumed_samples
            uid_base = self._consumed_samples * self.prompt_repeat_k

        group_data = []
        for item_idx in range(self.prompt_repeat_k):
            new_data = copy.deepcopy(data)
            new_data.extra_fields.setdefault("otel", {}).update(
                _build_otel_attrs(data, self._consumed_samples, item_idx, self.prompt_repeat_k)
            )
            if XTUNER_DETERMINISTIC:
                new_data.message_uid = message_uid
                new_data.uid = uid_base + item_idx
                new_data.session_uid = new_data.uid
            else:
                new_data.uid = uuid4().int
            group_data.append(new_data)
        self._consumed_samples += 1
        return cast(list[RolloutState], group_data)


class Sampler(_DatasetSampler):
    _DATALOADER_FILE = "dataloader"

    def __init__(
        self,
        dataloader: Dataloader,
        prompt_repeat_k: int,
        replay_buffer: ReplayBuffer,
    ):
        super().__init__(dataloader, prompt_repeat_k)
        self.replay_buffer = replay_buffer

    async def sample(self, task_name: str, group_status: list[Status] | None = None) -> list[RolloutState]:
        for status in group_status or []:
            buffer_data = await self.replay_buffer.get(1, task_name=task_name, group_status=status)
            if buffer_data:
                return buffer_data[0]
        return self.sample_from_dataloader()

    def save(self, checkpoint_path: Path | str) -> None:
        """Save the sampler's dataloader state to checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        dataloader_state = self.dataloader.get_state_dict()
        torch.save(dataloader_state, checkpoint_path / self._DATALOADER_FILE)

    def resume(self, checkpoint_path: Path | str) -> None:
        """Resume the sampler's dataloader state from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        dataloader_path = checkpoint_path / self._DATALOADER_FILE
        if not dataloader_path.exists():
            logger.warning(f"Dataloader state {dataloader_path} not found, skipping resume.")
            return
        state = torch.load(dataloader_path, map_location="cpu")
        self.dataloader.load_state_dict(state)
        self.dataloader_iter = iter(self.dataloader)
        self._consumed_samples = state["sampler"]["step"]
        self.cur_epoch = state["sampler"]["epoch"]
