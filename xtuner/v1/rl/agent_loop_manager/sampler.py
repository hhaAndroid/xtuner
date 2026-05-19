import copy
import warnings
from pathlib import Path
from typing import Iterator, Optional, cast
from uuid import uuid4

import ray
import torch
from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.rl.agent_loop_manager.trajectory_scheduler import PromptRequest
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.utils import XTUNER_DETERMINISTIC
from xtuner.v1.utils.logger import get_logger


logger = get_logger(__name__)


class SamplerExhausted(Exception):
    """Raised by :class:`Sampler` when ``single_epoch=True`` and the
    dataloader has yielded its last item.

    Producers running an evaluation pass should catch this and stop
    requesting new prompts so they never sample past the dataset size.
    Async callers must use a custom exception (instead of letting
    ``StopIteration`` escape) because PEP 479 turns a ``StopIteration``
    raised inside a coroutine into ``RuntimeError``.
    """


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
    single_epoch: bool = False
    """When True, the sampler stops at the end of one dataloader pass and
    raises :class:`SamplerExhausted`. Use for evaluation (so the producer
    cannot oversample the val set when ``max_on_fly`` is much larger than
    the dataset). Default ``False`` keeps the legacy training behaviour of
    rolling into the next epoch."""

    def build(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str, replay_buffer: ReplayBuffer
    ) -> "Sampler":
        if self.prompt_repeat_k != 1:
            warnings.warn(
                "SamplerConfig.prompt_repeat_k is deprecated and will be removed in a future "
                "release. The trajectory-level producer expresses repeat policy via "
                "GroupPolicyConfig.min_repeat / max_repeat; keep prompt_repeat_k only while the "
                "legacy group-based sample_from_dataloader path is still in use.",
                DeprecationWarning,
                stacklevel=2,
            )
        if isinstance(tokenizer, str):
            tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            tokenizer_obj = tokenizer
        dataloader = self.dataloader_cfg.build(
            tokenizer=tokenizer_obj, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        return Sampler(
            dataloader=dataloader,
            prompt_repeat_k=self.prompt_repeat_k,
            replay_buffer=replay_buffer,
            single_epoch=self.single_epoch,
        )


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
    def __init__(self, dataloader: Dataloader, prompt_repeat_k: int, single_epoch: bool = False):
        self.dataloader = dataloader
        self.dataloader_iter: Optional[Iterator] = None
        self.cur_epoch = 0
        self.prompt_repeat_k = prompt_repeat_k
        self._consumed_samples: int = 0
        self._single_epoch = single_epoch
        self._exhausted = False

    def __len__(self) -> int:
        return len(self.dataloader)

    def sample_from_dataloader(self) -> list[RolloutState]:
        data, seq_index = self._fetch_from_dataloader()
        group_data: list[RolloutState] = []
        for item_idx in range(self.prompt_repeat_k):
            new_data = copy.deepcopy(data)
            if XTUNER_DETERMINISTIC:
                uid_base = seq_index * self.prompt_repeat_k
                new_data.uid = uid_base + item_idx
                new_data.session_uid = new_data.uid
            else:
                new_data.uid = uuid4().int
            group_data.append(new_data)
        return cast(list[RolloutState], group_data)

    def _fetch_from_dataloader(self) -> tuple[RolloutState, int]:
        # Returns (data, seq_index) where seq_index is the pre-increment
        # _consumed_samples value. The counter is advanced before return so
        # callers that do not use seq_index still see the new position.
        # Single-epoch mode (eval) must never roll into the next epoch:
        # otherwise the producer's ``max_on_fly`` saturation will pull the
        # same val sample multiple times whenever ``max_on_fly`` exceeds the
        # dataset size.
        if self._exhausted:
            raise SamplerExhausted("Sampler is exhausted; single_epoch=True.")
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)
        assert self.dataloader_iter is not None
        try:
            data = cast(RolloutState, next(self.dataloader_iter)[0])
        except StopIteration:
            if self._single_epoch:
                self._exhausted = True
                raise SamplerExhausted("Dataloader exhausted in single_epoch mode.") from None
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = cast(RolloutState, next(self.dataloader_iter)[0])
        data = put_to_ray(data)
        seq_index = self._consumed_samples
        if XTUNER_DETERMINISTIC:
            data.message_uid = seq_index
        self._consumed_samples += 1
        return data, seq_index


class Sampler(_DatasetSampler):
    _DATALOADER_FILE = "dataloader"

    def __init__(
        self,
        dataloader: Dataloader,
        prompt_repeat_k: int,
        replay_buffer: ReplayBuffer,
        single_epoch: bool = False,
    ):
        super().__init__(dataloader, prompt_repeat_k, single_epoch=single_epoch)
        self.replay_buffer = replay_buffer

    @property
    def exhausted(self) -> bool:
        """True once a single-epoch sampler has yielded its last item."""
        return self._exhausted

    def reset(self) -> None:
        """Reset the iterator and the exhausted flag for a new pass.

        Evaluation runs should call this before each pass so the second
        call to :meth:`sample_prompt` does not see a stale exhausted flag.
        """
        self.dataloader_iter = None
        self._exhausted = False
        self._consumed_samples = 0
        self.cur_epoch = 0

    async def sample(self, task_name: str, group_status: list[Status] | None = None) -> list[RolloutState]:
        for status in group_status or []:
            buffer_data = await self.replay_buffer.get(1, task_name=task_name, group_status=status)
            if buffer_data:
                return buffer_data[0]
        return self.sample_from_dataloader()

    async def sample_prompt(self, task_name: str) -> PromptRequest:
        """Fetch a single prompt for trajectory-level production.

        The returned :class:`PromptRequest` owns a deep-copied
        :class:`RolloutState` whose ``message_uid`` is guaranteed non-``None``:

        * In deterministic mode the sampler writes ``_consumed_samples`` into
          ``message_uid`` (same value the legacy path uses for group
          identity).
        * Otherwise ``message_uid`` is left untouched when the dataloader
          already provides one (prompt-hash based id) and is filled with
          ``uuid4().int`` as a last-resort fallback.

        ``RolloutState.uid`` is intentionally left as whatever the dataloader
        produced; per-trajectory uids are assigned at spawn time by the
        scheduler.

        Args:
            task_name (str): Task owning the prompt.

        Returns:
            PromptRequest: The prompt wrapped for submission to the scheduler.
        """
        data, _ = self._fetch_from_dataloader()
        prompt = copy.deepcopy(data)
        prompt_uid = prompt.message_uid if prompt.message_uid is not None else uuid4().int
        prompt.message_uid = prompt_uid
        return PromptRequest(
            prompt_uid=prompt_uid,
            task_name=task_name,
            prompt=prompt,
        )

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
