import asyncio
import importlib
import time
from typing import Any, Awaitable, Callable, cast

from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.rollout import RolloutController

from .agent_loop import AgentLoop, AgentLoopConfig


class HarborResult(BaseModel):
    """Normalized result returned by a Harbor bridge function.

    The bridge function can return dict-like payloads with at least `response` OR
    `response_ids` set.
    """

    model_config = ConfigDict(extra="allow")

    response: str | None = None
    response_ids: list[int] | None = None
    logprobs: list[float] | None = None
    response_mask: list[int] | None = None
    finish_reason: str | None = "stop"
    reward: dict[str, Any] | None = None
    error_msg: str | None = None


HarborGenerateFn = Callable[[RolloutState, dict[str, Any]], HarborResult | dict[str, Any] | Awaitable[HarborResult | dict[str, Any]]]


class HarborAgentLoopConfig(AgentLoopConfig):
    """AgentLoop config that delegates generation to Harbor.

    Args:
        bridge_import_path: Python import path in form `module.submodule:function`.
            The function will be called with `(rollout_state, context_dict)`.
        bridge_kwargs: Extra kwargs passed into context_dict for the bridge.

    Bridge return payload supports keys:
      - response / response_ids
      - logprobs
      - response_mask
      - finish_reason
      - reward
      - error_msg
    """

    bridge_import_path: str
    bridge_kwargs: dict[str, Any] = Field(default_factory=dict)
    prefer_rollout_gateway: bool = True
    rollout_metadata_ttl_sec: int = 10

    def build(self, rollout_controller, judger=None, logger=None) -> "HarborAgentLoop":
        return HarborAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            bridge_import_path=self.bridge_import_path,
            bridge_kwargs=self.bridge_kwargs,
            prefer_rollout_gateway=self.prefer_rollout_gateway,
            rollout_metadata_ttl_sec=self.rollout_metadata_ttl_sec,
            judger=judger,
            logger=logger,
        )


class HarborAgentLoop(AgentLoop):
    """AgentLoop that uses an external Harbor bridge for rollout generation.

    Note:
        `rollout_ctl` is kept to preserve AgentLoop interface compatibility but
        is not used directly by this implementation.
    """

    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        bridge_import_path: str,
        bridge_kwargs: dict[str, Any] | None = None,
        prefer_rollout_gateway: bool = True,
        rollout_metadata_ttl_sec: int = 10,
        judger=None,
        logger=None,
    ):
        super().__init__(rollout_ctl=rollout_ctl, sample_params=sample_params, hf_checkpoint=hf_checkpoint, judger=judger, logger=logger)
        self.bridge_import_path = bridge_import_path
        self.bridge_kwargs = bridge_kwargs or {}
        self.prefer_rollout_gateway = prefer_rollout_gateway
        self.rollout_metadata_ttl_sec = max(1, rollout_metadata_ttl_sec)
        self._bridge_fn: HarborGenerateFn = self._load_bridge(bridge_import_path)
        self._rollout_meta_cache: dict[str, Any] | None = None
        self._rollout_meta_cache_ts: float = 0.0

    @staticmethod
    def _load_bridge(import_path: str) -> HarborGenerateFn:
        if ":" not in import_path:
            raise ValueError(
                f"Invalid bridge_import_path '{import_path}'. Expected format: module.submodule:function"
            )
        module_name, func_name = import_path.split(":", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, func_name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Bridge function '{func_name}' not found/callable in module '{module_name}'")
        return cast(HarborGenerateFn, fn)

    async def _call_bridge(self, rollout_state: RolloutState, rollout_step: int) -> HarborResult:
        resolved_gateway = await self._resolve_rollout_gateway()
        context = {
            "rollout_step": rollout_step,
            "sample_params": rollout_state.sample_params.model_dump() if rollout_state.sample_params else self.sample_params.model_dump(),
            "hf_checkpoint": self.hf_checkpoint,
            **resolved_gateway,
            **self.bridge_kwargs,
        }
        out = self._bridge_fn(rollout_state, context)
        if asyncio.iscoroutine(out):
            out = await cast(Awaitable[HarborResult | dict[str, Any]], out)

        if isinstance(out, HarborResult):
            return out
        if isinstance(out, dict):
            return HarborResult.model_validate(out)
        raise TypeError(f"Bridge function returned unsupported type: {type(out)}")

    async def _get_rollout_metadata(self) -> dict[str, Any]:
        now = time.time()
        if (
            self._rollout_meta_cache is not None
            and (now - self._rollout_meta_cache_ts) < self.rollout_metadata_ttl_sec
        ):
            return self._rollout_meta_cache

        ref = self.rollout_ctl.get_rollout_metadata.remote()  # type: ignore[attr-defined]
        metadata = await ref
        self._rollout_meta_cache = cast(dict[str, Any], metadata)
        self._rollout_meta_cache_ts = now
        return self._rollout_meta_cache

    @staticmethod
    def _select_active_gateway_url(metadata: dict[str, Any]) -> str | None:
        status_map = cast(dict[str, bool], metadata.get("worker_server_urls_status", {}) or {})
        server_url_dict = cast(dict[str, list[str]], metadata.get("server_url_dict", {}) or {})

        # 1) prefer active urls from status map
        for url, is_active in status_map.items():
            if is_active and url:
                return url

        # 2) fallback to first url in server_url_dict
        for urls in server_url_dict.values():
            for url in urls or []:
                if url:
                    return url
        return None

    @staticmethod
    def _normalize_api_base(url: str) -> str:
        # Keep existing /v1; append if missing.
        u = url.rstrip("/")
        if u.endswith("/v1"):
            return u
        return f"{u}/v1"

    async def _resolve_rollout_gateway(self) -> dict[str, Any]:
        if not self.prefer_rollout_gateway:
            return {}
        try:
            metadata = await self._get_rollout_metadata()
        except Exception as e:
            self.logger.warning(f"Failed to query rollout metadata, fallback to bridge kwargs: {e}")
            return {}

        url = self._select_active_gateway_url(metadata)
        if not url:
            return {}

        rollout_cfg = cast(dict[str, Any], metadata.get("rollout_config", {}) or {})
        api_key = rollout_cfg.get("api_key")
        return {
            "api_base": self._normalize_api_base(url),
            "inference_api_key": api_key,
        }

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        rollout_step = kwargs.get("rollout_step", 0)
        # Respect per-state sample params if caller overrides.
        if rollout_state.sample_params is None:
            rollout_state.sample_params = self.sample_params

        try:
            result = await self._call_bridge(rollout_state, rollout_step=rollout_step)
        except Exception as e:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = f"Harbor bridge call failed: {e}"
            return rollout_state

        if result.error_msg:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = result.error_msg
            return rollout_state

        if result.response_ids is None and result.response is None:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = "Harbor bridge returned neither response nor response_ids"
            return rollout_state

        if result.response_ids is None and result.response is not None:
            result.response_ids = self.tokenizer.encode(result.response, add_special_tokens=False)

        assert result.response_ids is not None
        rollout_state.response_ids = result.response_ids
        rollout_state.response = result.response or self.tokenizer.decode(result.response_ids)
        rollout_state.logprobs = result.logprobs
        rollout_state.response_mask = result.response_mask or [1] * len(result.response_ids)
        rollout_state.finish_reason = result.finish_reason or "stop"
        rollout_state.reward = result.reward
        rollout_state.status = Status.COMPLETED

        # Keep compatibility with downstream reward flow.
        rollout_state = await self.judge_sample(rollout_state)
        return rollout_state
