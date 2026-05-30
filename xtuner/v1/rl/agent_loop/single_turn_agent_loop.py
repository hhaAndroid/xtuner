import asyncio
import traceback
from collections import OrderedDict
from itertools import cycle
from typing import Any, overload
from uuid import uuid4

import httpx
import numpy as np
import ray

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    SampleParams,
    Status,
    reset_rollout_response,
    update_status_from_finish_reason,
)
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler
from xtuner.v1.rl.utils import cancel_and_drain, create_task
from xtuner.v1.rl.utils.misc import get_eos_token
from xtuner.v1.utils import XTUNER_DETERMINISTIC
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult

from .agent_loop import DEFAULT_JUDGER_CANCEL_TIMEOUT_S, AgentLoop, AgentLoopConfig


class _WorkerURLRouter:
    def __init__(self, urls: list[str], max_sessions: int = 10000) -> None:
        self._urls = urls
        self._cycler = cycle(urls)
        self._max_sessions = max_sessions
        self._session_to_url: OrderedDict[int, str] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get_url(self, session_id: int) -> str | None:
        if not self._urls:
            return None
        async with self._lock:
            if session_id in self._session_to_url:
                url = self._session_to_url.pop(session_id)
                self._session_to_url[session_id] = url
                return url
            url = next(self._cycler)
            self._session_to_url[session_id] = url
            while len(self._session_to_url) > self._max_sessions:
                self._session_to_url.popitem(last=False)
            return url


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

    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
            enable_batch_judge=self.enable_batch_judge,
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
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.enable_batch_judge = enable_batch_judge
        self._pause_event = asyncio.Event()
        self._generation_abort_counter = 0
        self.rollout_config, worker_urls = self._load_rollout_metadata()
        self.worker_url_router = _WorkerURLRouter(worker_urls)
        self.partial_rollout_handler = PartialRolloutHandler()
        self.lmdeploy_actor = None
        self.enable_return_routed_experts = self.rollout_config.enable_return_routed_experts
        eos_token = get_eos_token(str(self.rollout_config.model_path))
        self.eos_token: list[int] = [eos_token] if isinstance(eos_token, int) else eos_token
        max_batch_size = self.rollout_config.rollout_max_batch_size_per_instance or 1
        worker_count = max(1, len(worker_urls))
        per_worker_http_concurrency = max(1, int(max_batch_size * self.rollout_config.allow_over_concurrency_ratio))
        http_concurrency = worker_count * per_worker_http_concurrency
        self.logger.info(
            "SingleTurnAgentLoop direct rollout HTTP concurrency: "
            f"worker_count={worker_count}, per_worker={per_worker_http_concurrency}, total={http_concurrency}"
        )
        limits = httpx.Limits(max_connections=http_concurrency, max_keepalive_connections=http_concurrency)
        self.client = httpx.AsyncClient(limits=limits, timeout=self.rollout_config.rollout_timeout)

    def _load_rollout_metadata(self) -> tuple[RolloutConfig, list[str]]:
        get_rollout_metadata = self.rollout_ctl.get_rollout_metadata
        if hasattr(get_rollout_metadata, "remote"):
            metadata = ray.get(get_rollout_metadata.remote())  # type: ignore[attr-defined]
        else:
            metadata = get_rollout_metadata()

        server_url_dict = metadata["server_url_dict"]
        worker_server_urls_status = metadata.get("worker_server_urls_status") or {}
        worker_urls: list[str] = []
        for rank in sorted(server_url_dict, key=lambda value: int(value)):
            urls = server_url_dict[rank]
            if isinstance(urls, str):
                worker_urls.append(urls)
            else:
                worker_urls.extend(urls)
        active_worker_urls = [url for url in worker_urls if worker_server_urls_status.get(url, True)]
        self.logger.info(f"SingleTurnAgentLoop direct rollout worker URLs: {active_worker_urls}")
        return metadata["rollout_config"], active_worker_urls

    @overload
    async def run_judger(self, rollout_state: RolloutState) -> RolloutState: ...

    @overload
    async def run_judger(self, rollout_state: list[RolloutState]) -> list[RolloutState]: ...

    async def run_judger(self, rollout_state: RolloutState | list[RolloutState]) -> RolloutState | list[RolloutState]:
        assert self.judger is not None
        judge_task = create_task(self.judger.judge(rollout_state))
        pause_task = create_task(self._pause_event.wait())
        try:
            done, _ = await asyncio.wait({judge_task, pause_task}, return_when=asyncio.FIRST_COMPLETED)
            if judge_task in done:
                return await judge_task
            try:
                return await asyncio.wait_for(
                    asyncio.shield(judge_task),
                    timeout=DEFAULT_JUDGER_CANCEL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                await cancel_and_drain([judge_task])
                for sample in rollout_state if isinstance(rollout_state, list) else [rollout_state]:
                    sample.status = Status.ABORTED
                    sample.finish_reason = "abort"
                    sample.reward = None
                return rollout_state
        except asyncio.CancelledError:
            await cancel_and_drain([judge_task])
            for sample in rollout_state if isinstance(rollout_state, list) else [rollout_state]:
                sample.status = Status.ABORTED
                sample.finish_reason = "abort"
                sample.reward = None
            return rollout_state
        finally:
            await cancel_and_drain([pause_task])

    async def pause(self) -> None:
        self._generation_abort_counter += 1
        self._pause_event.set()
        # TODO: Decide whether Judger needs an explicit pause API for resources not owned by SingleTurnAgentLoop.
        try:
            await super().pause()
        finally:
            self._pause_event.clear()

    async def _wait_pause_request(self) -> None:
        await self._pause_event.wait()

    async def _safe_post_request(self, url: str, headers: dict[str, str], payload: dict) -> HttpRequestResult:
        send_task = None
        pause_task = None
        try:
            if self._pause_event.is_set():
                return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
            req = self.client.build_request(
                "POST",
                url,
                headers=headers,
                json=payload,
            )
            send_task = asyncio.create_task(self.client.send(req))
            pause_task = asyncio.create_task(self._wait_pause_request())
            done, _ = await asyncio.wait(
                {send_task, pause_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if send_task in done:
                response = await send_task
            else:
                try:
                    response = await asyncio.wait_for(asyncio.shield(send_task), timeout=10.0)
                except asyncio.TimeoutError:
                    await cancel_and_drain([send_task])
                    return HttpRequestResult(
                        error_type=HttpRequestErrorType.REQUEST_ABORTED,
                        url=url,
                        payload=payload,
                    )
            response.raise_for_status()
            return HttpRequestResult(response=response)
        except asyncio.CancelledError:
            await cancel_and_drain([send_task, pause_task])
            return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            return HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
        finally:
            await cancel_and_drain([pause_task])

    def _transform_sample_params(self, sample_params: SampleParams) -> dict:
        lmdeploy_sample_params = {
            "temperature": sample_params.temperature,
            "top_p": sample_params.top_p,
            "n": sample_params.n,
            "stream": sample_params.stream,
            "max_tokens": sample_params.max_tokens,
            "repetition_penalty": sample_params.repetition_penalty,
            "top_k": sample_params.top_k,
            "skip_special_tokens": sample_params.skip_special_tokens,
            "spaces_between_special_tokens": sample_params.spaces_between_special_tokens,
            "include_stop_str_in_output": sample_params.include_stop_str_in_output,
            "return_token_ids": sample_params.return_token_ids,
            "return_logprob": sample_params.return_logprob,
            "return_routed_experts": sample_params.return_routed_experts,
        }
        if sample_params.stops:
            lmdeploy_sample_params["stop"] = sample_params.stops
        if sample_params.min_tokens > 0:
            lmdeploy_sample_params["min_new_tokens"] = sample_params.min_tokens
        if sample_params.sampling_seed is not None:
            lmdeploy_sample_params["seed"] = sample_params.sampling_seed
        return lmdeploy_sample_params

    def _get_request_payload(self, rollout_state: RolloutState) -> dict:
        sample_params = rollout_state.sample_params
        input_tokens = rollout_state.tokens
        assert input_tokens is not None, "LMDeploy rollout requires token ids as input."
        assert sample_params.return_token_ids, "LMDeploy rollout requires token ids as output."

        optional_fields: dict[str, object] = {}
        if rollout_state.tools is not None:
            optional_fields["tools"] = rollout_state.tools
        if rollout_state.tool_choice is not None:
            optional_fields["tool_choice"] = rollout_state.tool_choice

        payload: dict[str, Any] = {
            "model": self.rollout_config.model_name,
            "messages": [],
            "input_ids": input_tokens,
            **optional_fields,
        }
        if "image_data" in rollout_state.extra_fields:
            payload["image_data"] = rollout_state.extra_fields["image_data"]

        sample_params = sample_params.model_copy(
            update={
                "return_routed_experts": self.enable_return_routed_experts and sample_params.return_routed_experts
            }
        )
        payload.update(self._transform_sample_params(sample_params))
        return payload

    async def _decode_routed_experts(self, routed_experts: Any) -> Any:
        if isinstance(routed_experts, str):
            if self.lmdeploy_actor is None:
                self.lmdeploy_actor = ray.get_actor("shared_store", namespace="lmdeploy")
            routed_experts_data = await self.lmdeploy_actor.get.remote(routed_experts)
            return ray.put(np.asarray(routed_experts_data))
        return np.asarray(routed_experts)

    async def _safe_handle_response(
        self,
        rollout_state: RolloutState,
        http_response: httpx.Response,
        *,
        enable_partial_rollout: bool,
    ) -> RolloutState:
        uid = rollout_state.message_uid
        sample_params = rollout_state.sample_params
        response = http_response.json()

        response_ids: list[int] = []
        logprobs: list[float] = []
        routed_experts = None
        should_return_routed_experts = self.enable_return_routed_experts and sample_params.return_routed_experts
        try:
            choice = response["choices"][0]
            returned_response = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")
            if finish_reason is None:
                if self._pause_event.is_set():
                    rollout_state.finish_reason = "abort"
                    rollout_state.status = Status.ABORTED
                    self.logger.warning(
                        f"finish_reason is missing when waiting for aborted message {uid}, defaulting to 'abort'. "
                        f"Response: {response}"
                    )
                else:
                    rollout_state.finish_reason = "error"
                    rollout_state.status = Status.FAILED
                    self.logger.warning(
                        f"finish_reason is missing for message {uid}, defaulting to 'error'. Response: {response}"
                    )
                rollout_state.error_msg = "Missing finish_reason in response"
                return rollout_state

            response_ids = choice.get("output_ids") or []
            for logprob, _token_id in choice.get("output_token_logprobs") or []:
                logprobs.append(logprob)

            if should_return_routed_experts:
                assert "routed_experts" in choice, (
                    "enable_return_routed_experts is True, but routed_experts is not in response choice"
                )
                routed_experts = choice["routed_experts"]
                if routed_experts is not None:
                    routed_experts = await self._decode_routed_experts(routed_experts)
                    if not isinstance(routed_experts, ray.ObjectRef):
                        routed_experts = ray.put(routed_experts)

            rollout_status = update_status_from_finish_reason(finish_reason)

            if rollout_status == Status.COMPLETED:
                validation_errors = []
                if not response_ids:
                    validation_errors.append("empty response_ids")
                if sample_params.return_logprob and not logprobs:
                    validation_errors.append("missing logprobs")
                if should_return_routed_experts and routed_experts is None:
                    validation_errors.append("missing routed_experts")
                if validation_errors:
                    error_msg = f"Incomplete rollout data for msg {uid}: {', '.join(validation_errors)}"
                    self.logger.error(error_msg)
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = error_msg
                    return rollout_state
            elif rollout_status == Status.FAILED:
                error_msg = f"Rollout failed for msg {uid} with finish_reason {finish_reason}"
                self.logger.error(error_msg)
                rollout_state.status = Status.FAILED
                rollout_state.error_msg = error_msg
                return rollout_state

            if enable_partial_rollout:
                usage = response.get("usage") or {}
                rollout_state = await self.partial_rollout_handler.postprocess(
                    rollout_state,
                    response=returned_response,
                    response_ids=response_ids,
                    logprobs=logprobs,
                    routed_experts=routed_experts,
                    finish_reason=finish_reason,
                    status=rollout_status,
                    prompt_tokens=usage.get("prompt_tokens", len(rollout_state.tokens or [])),
                    completion_tokens=usage.get("completion_tokens", len(response_ids)),
                )
            else:
                rollout_state.response = returned_response
                rollout_state.response_ids = response_ids
                rollout_state.logprobs = logprobs
                rollout_state.routed_experts = routed_experts
                rollout_state.finish_reason = finish_reason
                rollout_state.status = rollout_status
            return rollout_state
        except KeyError as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            raise RuntimeError(f"Missing expected key {e} in response {response_for_log} for {uid}") from e
        except IndexError as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            raise RuntimeError(f"Index error {e} while processing response {response_for_log} for {uid}") from e
        except AssertionError as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            raise RuntimeError(f"AssertionError: {e} when processing response {response_for_log} for {uid}") from e
        except TypeError as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            raise RuntimeError(f"TypeError: {e} when processing response {response_for_log} for {uid}") from e
        except Exception as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            error_msg = (
                f"Unexpected error: {e} when processing response {response_for_log} for {uid}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(error_msg) from e

    async def _request_worker(
        self,
        rollout_state: RolloutState,
        *,
        enable_partial_rollout: bool,
    ) -> RolloutState:
        if XTUNER_DETERMINISTIC:
            sample_params = rollout_state.sample_params.model_copy(deep=True)
            sample_params.sampling_seed = self.rollout_config.random_seed + (
                (rollout_state.uid or 0) - (rollout_state.message_uid or 0)
            )
            rollout_state.sample_params = sample_params

        session_id = rollout_state.session_uid if rollout_state.session_uid is not None else uuid4().int
        worker_url = await self.worker_url_router.get_url(session_id)
        if worker_url is None:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = "No active rollout worker URL available."
            return rollout_state

        uid = rollout_state.uid
        max_tokens = rollout_state.sample_params.max_tokens
        abort_counter = self._generation_abort_counter

        if enable_partial_rollout:
            rollout_state = self.partial_rollout_handler.preprocess(rollout_state, max_tokens)
        elif rollout_state.status == Status.ABORTED:
            rollout_state = reset_rollout_response(rollout_state)
            rollout_state.sample_params = rollout_state.sample_params.model_copy(update={"max_tokens": max_tokens})
            rollout_state.status = Status.INIT

        payload = self._get_request_payload(rollout_state)
        max_retries = self.rollout_config.max_retry_per_sample

        input_ids = payload.get("input_ids", [])
        request_max_tokens = payload.get("max_tokens")
        last_id = input_ids[-1] if input_ids else "None"
        is_max_tokens_zero = request_max_tokens is not None and request_max_tokens <= 0
        is_eos_reached = len(input_ids) > 0 and input_ids[-1] in self.eos_token
        if rollout_state.status == Status.COMPLETED:
            self.logger.debug(f"Request {uid} is already marked as COMPLETED, skipping generation.")
            return rollout_state
        if is_max_tokens_zero or is_eos_reached:
            self.logger.debug(
                f"No generation needed for request {uid}: max_tokens={request_max_tokens} or "
                f"last input_id={last_id} is in eos_token."
            )
            rollout_state.finish_reason = "stop" if is_eos_reached else "length"
            rollout_state.status = Status.COMPLETED
            return rollout_state

        endpoint_url = f"{worker_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.rollout_config.api_key}",
        }
        for attempt in range(max_retries + 1):
            is_last_attempt = attempt == max_retries
            http_result = await self._safe_post_request(endpoint_url, headers=headers, payload=payload)

            if http_result.response:
                rollout_state = await self._safe_handle_response(
                    rollout_state,
                    http_result.response,
                    enable_partial_rollout=enable_partial_rollout,
                )
                if abort_counter != self._generation_abort_counter and rollout_state.status != Status.ABORTED:
                    rollout_state.finish_reason = "abort"
                    rollout_state.status = Status.ABORTED
                    return rollout_state
                if rollout_state.status in [Status.COMPLETED, Status.ABORTED]:
                    return rollout_state

                if is_last_attempt:
                    self.logger.warning(
                        f"Invalid rollout response for request {uid} after {max_retries} attempts, marking as FAILED."
                    )
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = f"Invalid rollout response after {max_retries} attempts."
                    return rollout_state
                self.logger.warning(
                    f"Invalid rollout response for request {uid}, retrying {attempt + 1}/{max_retries}."
                )
                await asyncio.sleep(0.1)
                continue

            if http_result.error_type == HttpRequestErrorType.REQUEST_ABORTED:
                rollout_state.finish_reason = "abort"
                rollout_state.status = update_status_from_finish_reason("abort")
                return rollout_state

            if http_result.is_client_error:
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} was skipped due to client error "
                    f"{http_result.error_type} with {http_result.error_msg}"
                )
                rollout_state.error_msg = (
                    f"Client error {http_result.error_type} with message: {http_result.error_msg}"
                )
                rollout_state.status = Status.FAILED
                return rollout_state

            if http_result.is_server_error:
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} failed due to server error "
                    f"{http_result.error_type} with {http_result.error_msg}"
                )
                rollout_state.error_msg = (
                    f"Server error {http_result.error_type} with message: {http_result.error_msg}"
                )
                rollout_state.status = Status.FAILED
                return rollout_state

            if http_result.is_retryable:
                if is_last_attempt:
                    self.logger.warning(
                        f"rollout request {uid} to {http_result.url} failed after {max_retries} attempts due to "
                        f"retryable error {http_result.error_type} with {http_result.error_msg}"
                    )
                    rollout_state.error_msg = (
                        f"Request failed after {max_retries} attempts due to retryable error "
                        f"{http_result.error_type} with message: {http_result.error_msg}"
                    )
                    rollout_state.status = Status.FAILED
                    return rollout_state
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} failed due to retryable error "
                    f"{http_result.error_type} with {http_result.error_msg}, retrying {attempt + 1}/{max_retries}."
                )
                await asyncio.sleep(0.1)
                continue

            if http_result.is_unknown_error:
                raise RuntimeError(
                    f"Unexpected error during rollout request {uid} to {http_result.url}: {http_result.exception}"
                )
        return rollout_state

    async def generate_sample(
        self,
        rollout_state: RolloutState,
        **kwargs,
    ) -> RolloutState:
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        rollout_state = await self._request_worker(
            rollout_state,
            enable_partial_rollout=kwargs.get("enable_partial_rollout", False),
        )
        # 非 COMPLETED 状态（如被截断、放弃等）直接早退，不触发打分
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        if self.judger is not None and not self.enable_batch_judge:
            # 如果开启了批量打分，则在 generate_group 里统一打分，不在这里逐条打分
            rollout_state = await self.run_judger(rollout_state)
        return rollout_state

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        if self.judger is not None and self.enable_batch_judge:
            if not any(sample.status == Status.ABORTED for sample in group_samples):
                # 批量打分
                group_samples = await self.run_judger(group_samples)
        return group_samples
