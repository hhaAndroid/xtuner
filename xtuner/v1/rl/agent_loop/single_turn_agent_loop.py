import asyncio
import os
import traceback
import uuid
from typing import Any

from aiohttp import ClientSession, ClientTimeout, TCPConnector

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status, update_status_from_finish_reason
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import create_task

from .agent_loop import AgentLoop, AgentLoopConfig


ROUTED_APIPROXY_BASE_URL = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1"
ROUTED_APIPROXY_API_KEY = "sk-admin"
ROUTED_APIPROXY_TIMEOUT = 3600.0
ROUTED_APIPROXY_MAX_CONNECTIONS = 4096
ROUTED_APIPROXY_MAX_KEEPALIVE_CONNECTIONS = 128


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    """Configuration for the built-in single-turn agent loop.

    ``SingleTurnAgentLoopConfig`` runs one model generation for each input
    ``RolloutState`` and optionally sends the completed output to a judger. It
    is the default choice for math, QA, and other single-response RL tasks.

    Args:
        sample_params (SampleParams): Sampling parameters used by the rollout
            backend, such as temperature and maximum generation length.
        hf_checkpoint (str): Hugging Face checkpoint path used to identify the
            policy checkpoint for the agent loop.
        cpu_resources (CPUResourcesConfig | None): PG-external CPU resources
            used to run this agent loop as Ray actors. ``None`` runs the loop
            in local mode. Defaults to None.
        enable_batch_judge (bool): Whether to judge a generated group in one
            batch in ``generate_group``. Defaults to False.

    **Examples:**

    Example configuration for a single-turn task::

        config = SingleTurnAgentLoopConfig(
            sample_params=SampleParams(max_tokens=1024, temperature=1.0),
            hf_checkpoint="Qwen/Qwen3-8B",
            enable_batch_judge=True,
        )
    """

    enable_batch_judge: bool = False
    api_base_url: str = ROUTED_APIPROXY_BASE_URL
    api_key: str = ROUTED_APIPROXY_API_KEY
    api_timeout: float = ROUTED_APIPROXY_TIMEOUT
    api_max_connections: int = ROUTED_APIPROXY_MAX_CONNECTIONS
    api_max_keepalive_connections: int = ROUTED_APIPROXY_MAX_KEEPALIVE_CONNECTIONS

    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
            enable_batch_judge=self.enable_batch_judge,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            api_timeout=self.api_timeout,
            api_max_connections=self.api_max_connections,
            api_max_keepalive_connections=self.api_max_keepalive_connections,
        )


class SingleTurnAgentLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        enable_batch_judge: bool = False,
        api_base_url: str = ROUTED_APIPROXY_BASE_URL,
        api_key: str = ROUTED_APIPROXY_API_KEY,
        api_timeout: float = ROUTED_APIPROXY_TIMEOUT,
        api_max_connections: int = ROUTED_APIPROXY_MAX_CONNECTIONS,
        api_max_keepalive_connections: int = ROUTED_APIPROXY_MAX_KEEPALIVE_CONNECTIONS,
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.enable_batch_judge = enable_batch_judge
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.api_timeout = api_timeout
        self.api_max_connections = api_max_connections
        self.api_max_keepalive_connections = api_max_keepalive_connections
        self._model_name = os.environ.get("MODEL_NAME")
        self._http_client: ClientSession | None = None
        self._api_base_urls: list[str] | None = None
        self._api_base_url_index = 0
        self._api_base_urls_lock = asyncio.Lock()

    def _get_http_client(self) -> ClientSession:
        if self._http_client is None or self._http_client.closed:
            timeout = ClientTimeout(total=self.api_timeout)
            connector = TCPConnector(
                limit=self.api_max_connections,
                limit_per_host=0,
                keepalive_timeout=30.0,
            )
            self._http_client = ClientSession(timeout=timeout, connector=connector)
        return self._http_client

    async def _get_api_base_urls(self) -> list[str]:
        if self._api_base_urls is not None:
            return self._api_base_urls

        async with self._api_base_urls_lock:
            if self._api_base_urls is not None:
                return self._api_base_urls

            metadata = await self.rollout_ctl.get_rollout_metadata.remote()
            session_url_dict = metadata.get("worker_session_url_dict", {})
            session_urls_status = metadata.get("worker_session_urls_status", {})
            urls = [
                url.rstrip("/")
                for _, url in sorted(session_url_dict.items())
                if url and session_urls_status.get(url, True)
            ]
            if not urls:
                raise RuntimeError("No active SessionServer URLs found in rollout metadata.")
            self._api_base_urls = urls
            self.logger.info(f"[SingleTurnAgentLoop] using direct SessionServer URLs: {urls}")
            return urls

    async def _next_api_base_url(self) -> str:
        urls = await self._get_api_base_urls()
        index = self._api_base_url_index
        self._api_base_url_index = (self._api_base_url_index + 1) % len(urls)
        return urls[index]

    async def generate_sample(
        self,
        rollout_state: RolloutState,
        **kwargs,
    ) -> RolloutState:
        try:
            if rollout_state.uid is None:
                rollout_state.uid = uuid.uuid4().int
            response = await self._chat_completions(rollout_state)
            await self._fill_rollout_state_from_response(rollout_state, response)
        except Exception as exc:
            rollout_state.status = Status.FAILED
            rollout_state.finish_reason = "error"
            rollout_state.error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[SingleTurnAgentLoop] failed: {exc}\n{traceback.format_exc()}")
            return rollout_state

        if rollout_state.status != Status.COMPLETED:
            # 非 COMPLETED 状态（如被截断、放弃等）直接早退，不触发打分
            return rollout_state
        if self.judger is not None and not self.enable_batch_judge:
            # 如果开启了批量打分，则在 generate_group 里统一打分，不在这里逐条打分
            rollout_state = await self.judger.judge(rollout_state)
        return rollout_state

    def _build_http_payload(self, rollout_state: RolloutState, model_name: str) -> dict[str, Any]:
        sample_params = rollout_state.sample_params
        prompt_ids = rollout_state.tokens if rollout_state.tokens is not None else rollout_state.prompt_ids
        payload: dict[str, Any] = {
            "session_id": str(rollout_state.uid),
            "input_ids": prompt_ids,
            "max_tokens": sample_params.max_tokens,
            "temperature": sample_params.temperature,
            "top_p": sample_params.top_p,
            "top_k": sample_params.top_k,
            "repetition_penalty": sample_params.repetition_penalty,
            "skip_special_tokens": sample_params.skip_special_tokens,
            "return_logprob": sample_params.return_logprob,
            "stream": sample_params.stream,
            "include_stop_str_in_output": sample_params.include_stop_str_in_output,
            "spaces_between_special_tokens": sample_params.spaces_between_special_tokens,
            "return_routed_experts": True,
        }
        if sample_params.stops:
            payload["stop"] = sample_params.stops
        if sample_params.stop_token_ids:
            payload["stop_token_ids"] = sample_params.stop_token_ids
        if sample_params.min_tokens > 0:
            payload["min_new_tokens"] = sample_params.min_tokens
        return payload

    async def _chat_completions(self, rollout_state: RolloutState) -> dict[str, Any]:
        model_name = self._model_name
        if model_name is None:
            raise ValueError("MODEL_NAME environment variable is required for direct SessionServer rollout.")
        api_base_url = await self._next_api_base_url()
        url = f"{api_base_url}/generate"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        client = self._get_http_client()
        payload = self._build_http_payload(rollout_state, model_name)
        async with client.post(url, headers=headers, json=payload) as response:
            if response.status >= 400:
                response_text = await response.text()
                raise RuntimeError(f"HTTP rollout failed: status={response.status}. response={response_text}")
            data = await response.json(content_type=None)
        if "meta_info" not in data:
            raise RuntimeError(f"HTTP rollout response missing meta_info: {data}")
        return data

    async def _fill_rollout_state_from_response(
        self,
        rollout_state: RolloutState,
        response: dict[str, Any],
    ) -> None:
        meta_info = response.get("meta_info") or {}
        finish_reason_info = meta_info.get("finish_reason") or {}
        finish_reason = finish_reason_info.get("type")
        status = update_status_from_finish_reason(finish_reason)

        rollout_state.response = response.get("text", "")
        rollout_state.finish_reason = finish_reason
        rollout_state.status = status
        if status != Status.COMPLETED:
            rollout_state.error_msg = f"HTTP rollout finished with status={status.value}, finish_reason={finish_reason}"
            return

        response_ids = response.get("output_ids")
        output_token_logprobs = meta_info.get("output_token_logprobs")
        if output_token_logprobs:
            rollout_state.logprobs = [item[0] for item in output_token_logprobs]
            rollout_state.response_ids = [item[1] for item in output_token_logprobs]
        elif response_ids is not None:
            completion_tokens = meta_info.get("completion_tokens", 0)
            rollout_state.response_ids = response_ids[-completion_tokens:] if completion_tokens > 0 else []
            rollout_state.logprobs = []
        rollout_state.response_mask = [1] * len(rollout_state.response_ids or [])
        rollout_state.routed_experts = meta_info.get("routed_experts")

        # trace_store = get_store()
        # data = await trace_store.export_training_trace.remote(str(rollout_state.uid), text)
        # rollout_state.input_ids = data["input_ids"]
        # rollout_state.labels = data["labels"]
        # rollout_state.response_ids = [
        #     token_id
        #     for token_id, label in zip(data["input_ids"][1:], data["labels"][1:])
        #     if label != -100
        # ]
        # if rollout_state.response is None:
        #     raise ValueError("Response is None")
        # rollout_state.response_mask = [1] * len(rollout_state.response_ids)
        # rollout_state.logprobs = data["logprobs"]
        # rollout_state.routed_experts = data["routed_experts"]

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        if self.judger is not None and self.enable_batch_judge:
            # 批量打分
            group_samples = await self.judger.judge(group_samples)
        return group_samples
