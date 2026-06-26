import asyncio
import traceback
from typing import Any, overload

import httpx
import numpy as np
import ray

from xtuner.v1.rl.rollout.router import RolloutEndpointType
from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    SampleParams,
    Status,
    reset_rollout_response,
    update_status_from_finish_reason,
)
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template
from xtuner.v1.rl.rollout.parser.factory import build_reasoning_parser, build_tool_call_parser
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler
from xtuner.v1.rl.rollout.trace_store import get_store
from xtuner.v1.rl.utils import cancel_and_drain, create_task
from xtuner.v1.rl.utils.misc import get_eos_token
from xtuner.v1.utils import XTUNER_DETERMINISTIC
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult

from .agent_loop import AgentLoop, AgentLoopConfig


DEFAULT_JUDGER_CANCEL_TIMEOUT_S = 5.0


def _get_rollout_metadata(rollout_controller) -> dict[str, Any]:
    get_rollout_metadata = rollout_controller.get_rollout_metadata
    if hasattr(get_rollout_metadata, "remote"):
        return ray.get(get_rollout_metadata.remote())  # type: ignore[attr-defined]
    return get_rollout_metadata()


def _chat_completions_url(rollout_url: str) -> str:
    base_url = rollout_url.rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    """Configuration for the built-in single-turn agent loop.

    ``SingleTurnAgentLoopConfig`` runs one model generation for each input
    ``RolloutState`` and optionally sends the completed output to a judger. It
    is the default choice for math, QA, and other single-response RL tasks.
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

        metadata = _get_rollout_metadata(self.rollout_ctl)
        self.rollout_config = metadata["rollout_config"]
        self.partial_rollout_handler = PartialRolloutHandler()
        self.lmdeploy_actor = None
        self.enable_return_routed_experts = self.rollout_config.enable_return_routed_experts
        self._tool_call_parser = build_tool_call_parser(self.rollout_config.tool_call_parser)
        self._reasoning_parser = build_reasoning_parser(self.rollout_config.reasoning_parser, self.tokenizer)

        eos_token = get_eos_token(str(self.rollout_config.model_path))
        self.eos_token: list[int] = [eos_token] if isinstance(eos_token, int) else eos_token

        max_batch_size = self.rollout_config.rollout_max_batch_size_per_instance or 1
        metadata_worker_count = len(metadata.get("server_url_dict") or {}) or 1
        per_worker_http_concurrency = max(1, int(max_batch_size * self.rollout_config.allow_over_concurrency_ratio))
        http_concurrency = metadata_worker_count * per_worker_http_concurrency
        self.request_semaphore = asyncio.Semaphore(http_concurrency)
        limits = httpx.Limits(max_connections=http_concurrency, max_keepalive_connections=http_concurrency)
        self.client = httpx.AsyncClient(limits=limits, timeout=self.rollout_config.rollout_timeout)

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
                return await asyncio.wait_for(asyncio.shield(judge_task), timeout=DEFAULT_JUDGER_CANCEL_TIMEOUT_S)
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
        try:
            await super().pause()
        finally:
            self._pause_event.clear()

    async def _wait_pause_request(self) -> None:
        await self._pause_event.wait()

    async def _acquire_request_slot(self) -> bool:
        if self._pause_event.is_set():
            return False

        acquire_task = asyncio.create_task(self.request_semaphore.acquire())
        pause_task = asyncio.create_task(self._wait_pause_request())
        try:
            done, _ = await asyncio.wait({acquire_task, pause_task}, return_when=asyncio.FIRST_COMPLETED)
            if acquire_task in done:
                await acquire_task
                if self._pause_event.is_set():
                    self.request_semaphore.release()
                    return False
                return True
            await cancel_and_drain([acquire_task])
            return False
        except asyncio.CancelledError:
            await cancel_and_drain([acquire_task, pause_task])
            raise
        finally:
            await cancel_and_drain([pause_task])

    async def _safe_post_request(self, url: str, headers: dict[str, str], payload: dict) -> HttpRequestResult:
        send_task = None
        pause_task = None
        try:
            if self._pause_event.is_set():
                return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
            req = self.client.build_request("POST", url, headers=headers, json=payload)
            send_task = asyncio.create_task(self.client.send(req))
            pause_task = asyncio.create_task(self._wait_pause_request())
            done, _ = await asyncio.wait({send_task, pause_task}, return_when=asyncio.FIRST_COMPLETED)
            if send_task in done:
                response = await send_task
            else:
                try:
                    response = await asyncio.wait_for(asyncio.shield(send_task), timeout=10.0)
                except asyncio.TimeoutError:
                    await cancel_and_drain([send_task])
                    return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
            response.raise_for_status()
            return HttpRequestResult(response=response)
        except asyncio.CancelledError:
            await cancel_and_drain([send_task, pause_task])
            return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
        except Exception as e:
            return HttpRequestResult(
                error_type=HttpRequestErrorType.from_exception(e), exception=e, url=url, payload=payload
            )
        finally:
            await cancel_and_drain([pause_task])

    def _transform_sample_params(self, sample_params: SampleParams) -> dict[str, Any]:
        request_params = {
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
            request_params["stop"] = sample_params.stops
        if sample_params.min_tokens > 0:
            request_params["min_new_tokens"] = sample_params.min_tokens
        if sample_params.sampling_seed is not None:
            request_params["seed"] = sample_params.sampling_seed
        return request_params

    def _get_session_id(self, rollout_state: RolloutState) -> str:
        session_id = rollout_state.session_uid or rollout_state.uid or rollout_state.message_uid
        if session_id is None:
            raise RuntimeError("SessionServer rollout requires session_uid, uid, or message_uid as session_id.")
        return str(session_id)

    def _get_request_payload(
        self,
        rollout_state: RolloutState,
        *,
        endpoint_type: RolloutEndpointType,
    ) -> dict[str, Any]:
        sample_params = rollout_state.sample_params
        assert sample_params.return_token_ids, "SingleTurnAgentLoop rollout requires token ids as output."

        optional_fields: dict[str, Any] = {}
        if rollout_state.tools is not None:
            optional_fields["tools"] = rollout_state.tools
        if rollout_state.tool_choice is not None:
            optional_fields["tool_choice"] = rollout_state.tool_choice

        if endpoint_type == "session_server":
            if not rollout_state.message:
                raise RuntimeError("SessionServer rollout requires standard chat messages, got empty message.")
            payload: dict[str, Any] = {
                "model": self.rollout_config.model_name,
                "session_id": self._get_session_id(rollout_state),
                "messages": rollout_state.message,
                **optional_fields,
            }
        else:
            input_tokens = rollout_state.tokens
            assert input_tokens is not None, "SingleTurnAgentLoop worker rollout requires token ids as input."
            payload = {
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

    def _apply_output_parsers(self, rollout_state: RolloutState) -> None:
        if self._tool_call_parser is not None:
            parsed = self._tool_call_parser.parse(rollout_state)
            rollout_state.tool_calls = parsed.tool_calls
            rollout_state.response = parsed.remaining_text or None
        if self._reasoning_parser is not None:
            parsed_reasoning = self._reasoning_parser.parse(rollout_state)
            rollout_state.response = parsed_reasoning.remaining_text
            if parsed_reasoning.reasoning_text:
                rollout_state.extra_fields["reasoning_text"] = parsed_reasoning.reasoning_text
            else:
                rollout_state.extra_fields.pop("reasoning_text", None)

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

        logprobs: list[float] = []
        routed_experts = None
        should_return_routed_experts = self.enable_return_routed_experts and sample_params.return_routed_experts
        try:
            choice = response["choices"][0]
            returned_response = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")
            if finish_reason is None:
                rollout_state.finish_reason = "abort" if self._pause_event.is_set() else "error"
                rollout_state.status = Status.ABORTED if self._pause_event.is_set() else Status.FAILED
                rollout_state.error_msg = "Missing finish_reason in response"
                self.logger.warning(f"finish_reason is missing for message {uid}. Response: {response}")
                return rollout_state

            response_ids = choice.get("output_ids") or []
            for logprob, _token_id in choice.get("output_token_logprobs") or []:
                logprobs.append(logprob)

            if should_return_routed_experts:
                if "routed_experts" not in choice:
                    raise AssertionError(
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
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = f"Incomplete rollout data for msg {uid}: {', '.join(validation_errors)}"
                    return rollout_state
            elif rollout_status == Status.FAILED:
                rollout_state.status = Status.FAILED
                rollout_state.error_msg = f"Rollout failed for msg {uid} with finish_reason {finish_reason}"
                return rollout_state

            if enable_partial_rollout:
                usage = response.get("usage") or {}
                routed_experts_expect_len = (
                    usage.get("prompt_tokens", len(rollout_state.tokens or []))
                    + usage.get("completion_tokens", len(response_ids))
                    - 1
                )
                rollout_state = await self.partial_rollout_handler.postprocess(
                    rollout_state,
                    response=returned_response,
                    response_ids=response_ids,
                    logprobs=logprobs,
                    routed_experts=routed_experts,
                    finish_reason=finish_reason,
                    status=rollout_status,
                    routed_experts_expect_len=routed_experts_expect_len,
                )
            else:
                rollout_state.response = returned_response
                rollout_state.response_ids = response_ids
                rollout_state.logprobs = logprobs
                rollout_state.routed_experts = routed_experts
                rollout_state.finish_reason = finish_reason
                rollout_state.status = rollout_status
            self._apply_output_parsers(rollout_state)
            return rollout_state
        except (KeyError, IndexError, AssertionError, TypeError) as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            raise RuntimeError(f"Error {e} while processing response {response_for_log} for {uid}") from e
        except Exception as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            error_msg = (
                f"Unexpected error: {e} when processing response {response_for_log} for {uid}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(error_msg) from e

    def _build_session_server_prompt_text(self, rollout_state: RolloutState, response_message: dict[str, Any]) -> str:
        messages = [*rollout_state.message, response_message]
        return (
            self.tokenizer.apply_chat_template(
                canonicalize_messages_for_chat_template(messages),
                tools=rollout_state.tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        ).rstrip()

    async def _safe_handle_session_server_response(
        self,
        rollout_state: RolloutState,
        http_response: httpx.Response,
    ) -> RolloutState:
        uid = rollout_state.message_uid
        response = http_response.json()
        try:
            choice = response["choices"][0]
            response_message = choice["message"]
            returned_response = response_message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            if finish_reason is None:
                rollout_state.finish_reason = "abort" if self._pause_event.is_set() else "error"
                rollout_state.status = Status.ABORTED if self._pause_event.is_set() else Status.FAILED
                rollout_state.error_msg = "Missing finish_reason in SessionServer response"
                self.logger.warning(f"finish_reason is missing for message {uid}. Response: {response}")
                return rollout_state

            rollout_status = update_status_from_finish_reason(finish_reason)
            if rollout_status == Status.FAILED:
                rollout_state.status = Status.FAILED
                rollout_state.finish_reason = finish_reason
                rollout_state.error_msg = f"SessionServer rollout failed for msg {uid} with finish_reason {finish_reason}"
                return rollout_state

            if rollout_status == Status.ABORTED:
                rollout_state.status = Status.ABORTED
                rollout_state.finish_reason = finish_reason
                return rollout_state

            prompt_text = self._build_session_server_prompt_text(rollout_state, response_message)
            session_id = self._get_session_id(rollout_state)
            data = await get_store().export_training_trace.remote(session_id, prompt_text)

            input_ids = data["input_ids"]
            labels = data["labels"]
            response_ids = [token_id for token_id, label in zip(input_ids[1:], labels[1:]) if label != -100]
            logprobs = data["logprobs"]
            routed_experts = data["routed_experts"]

            validation_errors = []
            if not response_ids:
                validation_errors.append("empty response_ids")
            if rollout_state.sample_params.return_logprob and not logprobs:
                validation_errors.append("missing logprobs")
            if self.enable_return_routed_experts and rollout_state.sample_params.return_routed_experts:
                if not routed_experts or all(item is None for item in routed_experts):
                    validation_errors.append("missing routed_experts")
            if validation_errors:
                rollout_state.status = Status.FAILED
                rollout_state.finish_reason = finish_reason
                rollout_state.error_msg = (
                    f"Incomplete SessionServer trace data for msg {uid}: {', '.join(validation_errors)}"
                )
                return rollout_state

            rollout_state.input_ids = input_ids
            rollout_state.labels = labels
            rollout_state.response_ids = response_ids
            rollout_state.logprobs = logprobs
            rollout_state.routed_experts = routed_experts
            rollout_state.response = returned_response
            rollout_state.finish_reason = finish_reason
            rollout_state.status = rollout_status
            rollout_state.extra_fields["raw_prompt"] = prompt_text
            self._apply_output_parsers(rollout_state)
            return rollout_state
        except (KeyError, IndexError, AssertionError, TypeError) as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            raise RuntimeError(f"Error {e} while processing SessionServer response {response_for_log} for {uid}") from e
        except Exception as e:
            response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
            error_msg = (
                f"Unexpected error: {e} when processing SessionServer response {response_for_log} for {uid}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(error_msg) from e

    async def _request_rollout(
        self,
        rollout_state: RolloutState,
        *,
        rollout_url: str,
        endpoint_type: RolloutEndpointType,
        enable_partial_rollout: bool,
    ) -> RolloutState:
        if XTUNER_DETERMINISTIC:
            sample_params = rollout_state.sample_params.model_copy(deep=True)
            sample_params.sampling_seed = self.rollout_config.random_seed + (
                (rollout_state.uid or 0) - (rollout_state.message_uid or 0)
            )
            rollout_state.sample_params = sample_params

        uid = rollout_state.uid
        max_tokens = rollout_state.sample_params.max_tokens
        abort_counter = self._generation_abort_counter

        if endpoint_type == "session_server" and enable_partial_rollout:
            rollout_state.status = Status.FAILED
            rollout_state.finish_reason = "error"
            rollout_state.error_msg = "SingleTurnAgentLoop does not support partial rollout with session_server endpoint."
            return rollout_state

        if enable_partial_rollout:
            rollout_state = self.partial_rollout_handler.preprocess(rollout_state, max_tokens)
        elif rollout_state.status == Status.ABORTED:
            rollout_state = reset_rollout_response(rollout_state)
            rollout_state.sample_params = rollout_state.sample_params.model_copy(update={"max_tokens": max_tokens})
            rollout_state.status = Status.INIT

        payload = self._get_request_payload(rollout_state, endpoint_type=endpoint_type)
        input_ids = payload.get("input_ids", [])
        request_max_tokens = payload.get("max_tokens")
        last_id = input_ids[-1] if input_ids else "None"
        is_max_tokens_zero = request_max_tokens is not None and request_max_tokens <= 0
        is_eos_reached = endpoint_type == "worker" and len(input_ids) > 0 and input_ids[-1] in self.eos_token
        if rollout_state.status == Status.COMPLETED:
            return rollout_state
        if is_max_tokens_zero or is_eos_reached:
            self.logger.debug(
                f"No generation needed for request {uid}: max_tokens={request_max_tokens} or "
                f"last input_id={last_id} is in eos_token."
            )
            rollout_state.finish_reason = "stop" if is_eos_reached else "length"
            rollout_state.status = Status.COMPLETED
            return rollout_state

        endpoint_url = _chat_completions_url(rollout_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.rollout_config.api_key}",
        }
        max_retries = self.rollout_config.max_retry_per_sample
        for attempt in range(max_retries + 1):
            is_last_attempt = attempt == max_retries
            has_request_slot = await self._acquire_request_slot()
            if not has_request_slot:
                rollout_state.finish_reason = "abort"
                rollout_state.status = update_status_from_finish_reason("abort")
                return rollout_state

            try:
                http_result = await self._safe_post_request(endpoint_url, headers=headers, payload=payload)
            finally:
                self.request_semaphore.release()

            if http_result.response:
                if endpoint_type == "session_server":
                    rollout_state = await self._safe_handle_session_server_response(rollout_state, http_result.response)
                else:
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
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = f"Invalid rollout response after {max_retries} attempts."
                    return rollout_state
                self.logger.warning(f"Invalid rollout response for request {uid}, retrying {attempt + 1}/{max_retries}.")
                await asyncio.sleep(0.1)
                continue

            if http_result.error_type == HttpRequestErrorType.REQUEST_ABORTED:
                rollout_state.finish_reason = "abort"
                rollout_state.status = update_status_from_finish_reason("abort")
                return rollout_state

            if http_result.is_client_error:
                rollout_state.error_msg = f"Client error {http_result.error_type} with message: {http_result.error_msg}"
                rollout_state.status = Status.FAILED
                return rollout_state

            if http_result.is_server_error:
                rollout_state.error_msg = f"Server error {http_result.error_type} with message: {http_result.error_msg}"
                rollout_state.status = Status.FAILED
                return rollout_state

            if http_result.is_retryable:
                if is_last_attempt:
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
        rollout_url = kwargs.get("rollout_url")
        endpoint_type = kwargs.get("rollout_endpoint_type", "worker")
        if rollout_url is None:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = "SingleTurnAgentLoop requires rollout_url."
            return rollout_state

        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        rollout_state = await self._request_rollout(
            rollout_state,
            rollout_url=rollout_url,
            endpoint_type=endpoint_type,
            enable_partial_rollout=kwargs.get("enable_partial_rollout", False),
        )
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        if self.judger is not None and not self.enable_batch_judge:
            rollout_state = await self.run_judger(rollout_state)
        return rollout_state

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        rollout_urls = kwargs.pop("rollout_urls", None)
        rollout_endpoint_types = kwargs.pop("rollout_endpoint_types", None)
        pending_tasks = []
        for idx, state in enumerate(rollout_state):
            state.sample_params = self.sample_params
            sample_kwargs = dict(kwargs)
            if rollout_urls is not None:
                sample_kwargs["rollout_url"] = rollout_urls[idx]
            if rollout_endpoint_types is not None:
                sample_kwargs["rollout_endpoint_type"] = rollout_endpoint_types[idx]
            task = create_task(self.generate_sample(state, **sample_kwargs))
            pending_tasks.append(task)
        group_samples = await asyncio.gather(*pending_tasks)
        if self.judger is not None and self.enable_batch_judge:
            if not any(sample.status == Status.ABORTED for sample in group_samples):
                group_samples = await self.run_judger(group_samples)
        return group_samples
