from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import ray
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import CPUResourcesConfig, create_task
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


AGENT_LOOP_ACTOR_MAX_CONCURRENCY = 1000000
DEFAULT_JUDGER_CANCEL_TIMEOUT_S = 5.0


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    hf_checkpoint: str
    sample_params: SampleParams
    cpu_resources: CPUResourcesConfig | None = None
    bound_worker_url: str | None = None

    def build(self, rollout_controller, judger: Judger | None = None, logger=None) -> "RouterAgentLoop":
        metadata = _get_agent_loop_metadata(rollout_controller)
        worker_entries = _get_active_worker_entries(metadata)
        if not worker_entries:
            raise RuntimeError(f"No active rollout worker URL available for {self.__class__.__name__} actors.")

        placement_group = metadata["placement_group"]
        workers = []
        for entry in worker_entries:
            actor_config = self.model_copy(update={"bound_worker_url": entry["url"]})
            ray_actor_cls = actor_config.get_ray_actor_cls()
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=entry["head_bundle_idx"],
                placement_group_capture_child_tasks=True,
            )
            workers.append(
                ray_actor_cls.options(
                    num_cpus=1,
                    scheduling_strategy=scheduling_strategy,
                ).remote(config=actor_config, rollout_ctl=rollout_controller, judger=judger)
            )

        if logger is not None:
            logger.info(
                f"{self.__class__.__name__} auto actor mode: "
                f"active_workers={len(workers)}, urls={[entry['url'] for entry in worker_entries]}, "
                f"num_cpus_per_actor=1, max_concurrency={AGENT_LOOP_ACTOR_MAX_CONCURRENCY}"
            )
        return RouterAgentLoop(workers=workers, rollout_ctl=rollout_controller)

    def get_ray_actor_cls(self) -> ActorClass:
        return get_ray_agent_loop_cls(self.get_agent_loop_cls())

    @abstractmethod
    def get_agent_loop_cls(self) -> type["AgentLoop"]: ...


def _get_rollout_metadata(rollout_controller) -> dict[str, Any]:
    get_rollout_metadata = rollout_controller.get_rollout_metadata
    if hasattr(get_rollout_metadata, "remote"):
        return ray.get(get_rollout_metadata.remote())  # type: ignore[attr-defined]
    return get_rollout_metadata()


def _get_agent_loop_metadata(rollout_controller) -> dict[str, Any]:
    get_agent_loop_metadata = getattr(rollout_controller, "get_agent_loop_metadata", None)
    if get_agent_loop_metadata is None:
        raise RuntimeError("RolloutController does not expose get_agent_loop_metadata().")
    if hasattr(get_agent_loop_metadata, "remote"):
        return ray.get(get_agent_loop_metadata.remote())  # type: ignore[attr-defined]
    return get_agent_loop_metadata()


def _get_active_worker_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    server_url_dict = metadata["server_url_dict"]
    worker_server_urls_status = metadata.get("worker_server_urls_status") or {}
    worker_url_to_placement = metadata.get("worker_url_to_placement") or {}
    worker_entries: list[dict[str, Any]] = []

    for rank in sorted(server_url_dict, key=lambda value: int(value)):
        urls = server_url_dict[rank]
        urls = [urls] if isinstance(urls, str) else urls
        for url in urls:
            if not worker_server_urls_status.get(url, True):
                continue
            placement = worker_url_to_placement.get(url)
            if placement is None:
                raise RuntimeError(f"Missing rollout placement metadata for worker URL {url}.")
            worker_entries.append(
                {
                    "url": url,
                    "rank": placement["rank"],
                    "head_bundle_idx": placement["head_bundle_idx"],
                    "engine_bundle_idxs": placement["engine_bundle_idxs"],
                }
            )
    return worker_entries


class AgentLoop(ABC):
    def __init__(
        self,
        config: AgentLoopConfig,
        rollout_ctl: RolloutController,
        judger: Judger | None = None,
        logger=None,
    ) -> None:
        self.config = config
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = config.hf_checkpoint
        self.tokenizer = load_tokenizer(config.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(config.hf_checkpoint, trust_remote_code=True)
        self.sample_params = config.sample_params
        self.judger = judger
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    @abstractmethod
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState: ...

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples

    async def pause(self) -> None:
        # Base AgentLoop only pauses rollout generation.
        #
        # We intentionally do not define generic judger pause behavior in the
        # Judger base class. Judger subclasses can implement judge() in very
        # different ways, and one base pause implementation cannot cover all of
        # them. Requiring users to follow a base-class pause protocol would also
        # increase the mental overhead of writing a new judge() implementation.
        #
        # For now, only SingleTurnAgentLoop defines how to pause an in-flight
        # judger call. Other AgentLoop subclasses should override pause() if
        # they need their own judger pause semantics.
        await self.rollout_ctl.pause_generation.remote()  # type: ignore[attr-defined]


class RouterAgentLoop:
    def __init__(self, workers: list[AgentLoopActorProxy], rollout_ctl: RolloutController):
        self.workers = workers
        self.rollout_ctl = rollout_ctl
        self._worker_loads = dict.fromkeys(workers, 0)
        self._rr_index = 0
        self._lock = asyncio.Lock()

    async def _pick_worker(self) -> AgentLoopActorProxy:
        async with self._lock:
            min_load = min(self._worker_loads.values())
            candidates = [worker for worker in self.workers if self._worker_loads[worker] == min_load]
            worker = candidates[self._rr_index % len(candidates)]
            self._rr_index = (self._rr_index + 1) % len(self.workers)
            self._worker_loads[worker] += 1
            return worker

    async def _release_worker(self, worker: AgentLoopActorProxy) -> None:
        async with self._lock:
            self._worker_loads[worker] -= 1

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        worker = await self._pick_worker()
        try:
            return await worker.generate_sample.remote(rollout_state, **kwargs)
        finally:
            await self._release_worker(worker)

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        worker = await self._pick_worker()
        try:
            return await worker.generate_group.remote(rollout_state, **kwargs)
        finally:
            await self._release_worker(worker)

    def get_worker_status(self) -> dict[str, int]:
        return {str(worker): load for worker, load in self._worker_loads.items()}

    async def pause(self) -> None:
        await asyncio.gather(
            *(worker.pause.remote() for worker in self.workers),
        )


async def get_agent_loop_rollout_ctl(agent_loop: AgentLoopSpec) -> RolloutController:
    return agent_loop.rollout_ctl


_RAY_AGENT_LOOP_CLS_CACHE: dict[type[AgentLoop], ActorClass] = {}


def get_ray_agent_loop_cls(agent_loop_cls: type[AgentLoop]) -> ActorClass:
    if agent_loop_cls not in _RAY_AGENT_LOOP_CLS_CACHE:
        _RAY_AGENT_LOOP_CLS_CACHE[agent_loop_cls] = ray.remote(
            max_concurrency=AGENT_LOOP_ACTOR_MAX_CONCURRENCY
        )(agent_loop_cls)
    return _RAY_AGENT_LOOP_CLS_CACHE[agent_loop_cls]


AgentLoopActorProxy: TypeAlias = ActorProxy[Any]
AgentLoopSpec: TypeAlias = RouterAgentLoop
