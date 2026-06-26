import json
import base64
import asyncio
import os
import time
import uuid
from functools import reduce
from operator import add
from typing import Any, Optional

import numpy as np
import ray
from aiohttp import ClientConnectionResetError, ClientSession, ClientTimeout, web

from transformers import AutoConfig, AutoTokenizer
from xtuner.v1.utils import get_logger

from .chat_template import canonicalize_messages_for_chat_template
from .trace_store import TokenizedSegment, get_store


def _is_error_payload(payload: dict) -> bool:
    return payload.get("error") is not None or payload.get("type") == "error" or payload.get("object") == "error"


def _lmdeploy_error_payload(message: str, status: int = 500, error_type: str = "internal_server_error") -> dict:
    return {
        "message": message,
        "type": error_type,
        "code": status,
        "object": "error",
    }


def _short_repr(value: Any, max_chars: int = 512) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def _payload_summary(payload: Optional[dict]) -> dict:
    if not isinstance(payload, dict):
        return {}

    messages = payload.get("messages")
    choices = payload.get("choices")
    return {
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "session_id": payload.get("session_id"),
        "input_ids_len": len(payload.get("input_ids") or []),
        "messages_len": len(messages) if isinstance(messages, list) else None,
        "tools_len": len(payload.get("tools") or []),
        "max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens")),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        "return_logprob": payload.get("return_logprob"),
        "return_token_ids": payload.get("return_token_ids"),
        "return_routed_experts": payload.get("return_routed_experts"),
        "choices_len": len(choices) if isinstance(choices, list) else None,
    }


def _stream_event_summary(event: Any) -> dict:
    if not isinstance(event, dict):
        return {"event": _short_repr(event, 256)}

    summary: dict[str, Any] = {
        "id": event.get("id"),
        "object": event.get("object"),
        "model": event.get("model"),
        "type": event.get("type"),
    }
    if event.get("error") is not None:
        summary["error"] = event.get("error")
    choices = event.get("choices")
    if isinstance(choices, list):
        summary["choices_len"] = len(choices)
        finish_reasons = []
        output_ids_len = 0
        output_token_logprobs_len = 0
        content_len = 0
        reasoning_content_len = 0
        tool_calls_len = 0
        tool_call_arguments_len = 0
        routed_experts_len = 0
        routed_experts_type = None
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reasons.append(choice.get("finish_reason"))
            output_ids_len += len(choice.get("output_ids") or [])
            output_token_logprobs_len += len(choice.get("output_token_logprobs") or [])
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_len += len(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_content_len += len(delta["reasoning_content"])
            for tool_call in delta.get("tool_calls") or []:
                tool_calls_len += 1
                fn = tool_call.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    tool_call_arguments_len += len(fn["arguments"])
            routed_experts = choice.get("routed_experts")
            if routed_experts is not None:
                routed_experts_type = type(routed_experts).__name__
                if isinstance(routed_experts, str):
                    routed_experts_len += len(routed_experts)
        summary["finish_reasons"] = finish_reasons
        summary["output_ids_len"] = output_ids_len
        summary["output_token_logprobs_len"] = output_token_logprobs_len
        summary["content_len"] = content_len
        summary["reasoning_content_len"] = reasoning_content_len
        summary["tool_calls_len"] = tool_calls_len
        summary["tool_call_arguments_len"] = tool_call_arguments_len
        summary["routed_experts_type"] = routed_experts_type
        summary["routed_experts_len"] = routed_experts_len
    if event.get("usage") is not None:
        summary["usage"] = event.get("usage")
    return summary


def _stream_has_traceable_choices(raw: bytes) -> bool:
    text = raw.decode("utf-8", errors="replace")
    has_choices = False
    saw_done = False
    saw_terminal_finish = False
    for line in text.split("\n"):
        line = line.strip()
        if line == "data: [DONE]":
            saw_done = True
            continue
        if line.startswith("data: "):
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                return False
            if _is_error_payload(event):
                return False
            if event.get("choices"):
                if any(choice.get("finish_reason") == "error" for choice in event.get("choices", [])):
                    return False
                if any(choice.get("finish_reason") for choice in event.get("choices", [])):
                    saw_terminal_finish = True
                has_choices = True
    return has_choices and saw_done and saw_terminal_finish


def _extract_output_logprobs(choice: dict, output_token_ids: list[int]) -> list[float]:
    if not output_token_ids:
        return []

    output_token_logprobs = choice.get("output_token_logprobs")
    if output_token_logprobs is None:
        raise RuntimeError(
            "SessionServer response choice has no output_token_logprobs; "
            "the return_logprob protocol is required for training traces."
        )

    logprob_token_ids = [item[1] for item in output_token_logprobs]
    if logprob_token_ids != output_token_ids:
        raise RuntimeError(
            "SessionServer response choice has mismatched output_token_logprobs: "
            f"output_ids_len={len(output_token_ids)}, logprob_ids_len={len(logprob_token_ids)}"
        )
    return [item[0] for item in output_token_logprobs]


_SESSION_SERVER_ONLY_KEYS = {"session_id"}
_DEFAULT_MAX_MODEL_LEN_SAFETY_MARGIN = 16


def _bool_request_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _request_uses_trace_store(req_body: dict) -> bool:
    return _bool_request_value(req_body.get("return_token_ids"), True)


class SessionServer:
    """SessionServer intercepts and records requests sent to a remote LLM API
    worker.

    It acts as a reverse-proxy (or interceptor) in front of an already running
    worker (like lmdeploy, sglang, or vllm). It binds to a specific (host, port)
    and relays any received traffic to the actual worker URL.

    You can optionally provide before_request and after_response hooks to
    perform extra logging, trace state mutations, or message cleanup before/after
    routing the request to the worker backend.

    Args:
        worker_base_url (str): The base URL of the real worker (e.g. "http://127.0.0.1:8000")
        tokenizer_path (str): The path to the tokenizer model.
        host (str): Host for this session server to listen on.
        port (int): Port for this session server to listen on.
        request_timeout (float): Total timeout in seconds for forwarding requests to the worker.
        read_bufsize (int): Buffer limit for line reader in ClientSession. Default is 64MB (2**26).
    """

    def __init__(
        self,
        worker_base_url: str,
        tokenizer_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        request_timeout: float = 1200.0,
        read_bufsize: int = 2**26,
        max_model_len: Optional[int] = None,
        max_model_len_reserved_tokens: int = 0,
        normalize_sglang_sampling_params: bool = False,
        drain_upstream_on_client_disconnect: bool = True,
        enable_return_routed_experts: bool = True,
    ):
        self.worker_base_url = worker_base_url.rstrip("/")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model_config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
        text_config = getattr(self.model_config, "text_config", self.model_config)
        self.routed_experts_num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        self.routed_experts_num_experts_per_tok = getattr(text_config, "num_experts_per_tok", None)
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.read_bufsize = read_bufsize
        self.max_model_len = max_model_len
        self.max_model_len_safety_margin = max(
            _DEFAULT_MAX_MODEL_LEN_SAFETY_MARGIN,
            max_model_len_reserved_tokens + 1,
        )
        self.normalize_sglang_sampling_params = normalize_sglang_sampling_params
        self.drain_upstream_on_client_disconnect = drain_upstream_on_client_disconnect
        self.enable_return_routed_experts = enable_return_routed_experts
        self.store = get_store()
        self.stop_word = self.tokenizer.eos_token or ""

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._lmdeploy_actor: Optional[ray.actor.ActorHandle] = None

    async def on_request(self, req_body: dict, *, trace_enabled: bool = True) -> dict:
        """Hook for processing/modifying the request before forwarding."""

        if not trace_enabled:
            worker_req = {k: v for k, v in req_body.items() if k not in _SESSION_SERVER_ONLY_KEYS}
            if "logprobs" in worker_req:
                worker_req.setdefault("return_logprob", worker_req.pop("logprobs"))
            if not _bool_request_value(worker_req.get("return_logprob"), False):
                worker_req.pop("top_logprobs", None)
                worker_req["return_logprob"] = False
            worker_req["return_token_ids"] = False
            worker_req.setdefault("return_routed_experts", self.enable_return_routed_experts)
            return worker_req

        session_id = req_body["session_id"]
        # 1. chat_template render 出完整 prompt string，不 tokenize 全量
        prompt_text = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(req_body["messages"]),
            tools=req_body.get("tools", None),
            add_generation_prompt=True,
            tokenize=False,
        )

        # 2. Store 做 string prefix match。
        prefix, nodes = await self.store.search.remote(session_id, prompt_text, filter_none=True)
        if prefix:
            get_logger().debug(f"Hit prefix cache for session {session_id}")
        delta, delta_ids = prompt_text[len(prefix) :], []
        if delta:
            delta_ids = self.tokenizer.encode(delta, add_special_tokens=False)
            await self.store.insert.remote(session_id, prompt_text, TokenizedSegment(text=delta, token_ids=delta_ids))
        input_ids = reduce(add, [node.value.token_ids for node in nodes] + [delta_ids])

        # 3. 组装 OpenAI chat completions 请求。
        worker_req = {
            **{
                k: v
                for k, v in req_body.items()
                if k not in _SESSION_SERVER_ONLY_KEYS | {"messages", "logprobs", "top_logprobs"}
            },
            "messages": [],
            "input_ids": input_ids,
            "return_token_ids": True,
            "return_routed_experts": self.enable_return_routed_experts
            and _bool_request_value(req_body.get("return_routed_experts"), True),
            "return_logprob": True,
            "include_stop_str_in_output": True,
        }
        self._normalize_sampling_params(worker_req)
        self._cap_completion_budget(worker_req, input_ids_len=len(input_ids))
        return worker_req

    def _normalize_sampling_params(self, worker_req: dict) -> None:
        if not self.normalize_sglang_sampling_params:
            return

        # Keep SessionServer behavior aligned with SGLangWorker._transform_sample_params.
        # XTuner uses top_k=0 for "disabled", while SGLang's OpenAI server expects -1.
        if worker_req.get("top_p", 0) > 0 and worker_req.get("top_k") != -1:
            worker_req["top_k"] = -1

    def _cap_completion_budget(self, worker_req: dict, *, input_ids_len: int) -> None:
        if self.max_model_len is None:
            return

        remaining_tokens = self.max_model_len - input_ids_len - self.max_model_len_safety_margin
        if remaining_tokens <= 0:
            return

        for key in ("max_tokens", "max_completion_tokens"):
            value = worker_req.get(key)
            if not isinstance(value, int) or value <= remaining_tokens:
                continue
            worker_req[key] = remaining_tokens

    async def on_response(self, worker_resp: dict, *, trace_enabled: bool = True) -> dict:
        """Hook for processing the parsed response received from the worker."""

        if not trace_enabled:
            return {k: v for k, v in worker_resp.items() if k not in {"messages", "tools"}}

        session_id = worker_resp["session_id"]
        messages = worker_resp["messages"]
        tools = worker_resp["tools"]
        choice = worker_resp["choices"][0]

        output_token_ids = choice.get("output_ids")  # len = N_out
        if output_token_ids is None:
            raise RuntimeError(
                "SessionServer response choice has no output_ids; "
                "cannot export a training trace for this assistant turn."
            )
        output_logprobs = _extract_output_logprobs(choice, output_token_ids)
        raw_routed_expert = choice.get("routed_experts")  # 本次 call 的 raw routed_expert，可为 None

        # 2. Store 把 input_delta / assistant_output 两个节点补齐字段。
        old_prompt = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(messages), tools=tools, add_generation_prompt=True, tokenize=False
        )
        messages = [*messages, choice["message"]]
        new_prompt = (
            self.tokenizer.apply_chat_template(
                canonicalize_messages_for_chat_template(messages),
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
        ).rstrip()
        assert new_prompt.startswith(old_prompt) and new_prompt.endswith(self.stop_word)

        if raw_routed_expert is not None:
            raw_routed_expert = await self._decode_routed_experts(raw_routed_expert)
            if len(raw_routed_expert) > 0:
                num_layers = raw_routed_expert.shape[1]
                topk_experts = raw_routed_expert.shape[2]
                dummy_expert = np.full((1, num_layers, topk_experts), 0, dtype=raw_routed_expert.dtype)
                raw_routed_expert = np.concatenate([dummy_expert, raw_routed_expert], axis=0)

            _, nodes = await self.store.search.remote(session_id, old_prompt, filter_none=True)

            # last node in nodes corresponds to the delta inserted in on_request (if any)
            if nodes:
                delta_node_val: TokenizedSegment = nodes[-1].value
                delta_len = len(delta_node_val.token_ids)
                prefix_len = sum(len(n.value.token_ids) for n in nodes[:-1])
                assert prefix_len + delta_len + len(output_token_ids) == len(raw_routed_expert)

                # split raw_routed_expert
                # raw_routed_expert target shape mapping: [prefix_len + delta_len + response_len, ...]
                delta_expert = raw_routed_expert[prefix_len : prefix_len + delta_len]
                response_expert = raw_routed_expert[prefix_len + delta_len :]

                if delta_len > 0:
                    delta_node_val.expert_key = ray.put(delta_expert)
                    # update delta node in store
                    await self.store.insert.remote(session_id, old_prompt, delta_node_val)

                raw_routed_expert = ray.put(response_expert)
            else:
                raw_routed_expert = ray.put(raw_routed_expert)

        await self.store.insert.remote(
            session_id,
            key=new_prompt,
            value=TokenizedSegment(
                text=new_prompt[len(old_prompt) :],
                token_ids=output_token_ids,
                logprobs=output_logprobs,
                labels=output_token_ids,
                expert_key=raw_routed_expert,
                length=len(output_token_ids),
            ),
        )

        # 3. 返回标准 OpenAI response，session_id 由 SessionClient 层再剥
        resp = {k: v for k, v in worker_resp.items() if k != "messages"}
        return resp

    @property
    def url(self) -> str:
        """The bound URL for the SessionServer."""
        return f"http://{self.host}:{self.port}"

    async def start(self):
        """Start the SessionServer proxy application."""
        if self._site is not None:
            return

        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        get_logger().info(f"SessionServer listening on {self.url} (Forwarding to {self.worker_base_url})")

    async def stop(self):
        """Cleanly stop the SessionServer application."""
        if self._runner:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None
        get_logger().info("SessionServer stopped.")

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Proxy handler for the worker API."""

        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started_at = time.monotonic()
        logger = get_logger()

        # Read the request body
        request_body = await request.read()
        request_data = session_id = messages = None
        trace_enabled = False
        orig_return_logprob = orig_return_token_ids = orig_return_routed_experts = False
        if request_body:
            try:
                request_data = json.loads(request_body)

                trace_enabled = _request_uses_trace_store(request_data)
                orig_return_logprob = _bool_request_value(
                    request_data.get("return_logprob", request_data.get("logprobs")), False
                )
                orig_return_token_ids = _bool_request_value(request_data.get("return_token_ids"), False)
                orig_return_routed_experts = _bool_request_value(request_data.get("return_routed_experts"), True)

                session_id = request_data.get("session_id")
                messages = request_data.get("messages")
                tools = request_data.get("tools", None)

                # Apply purely abstract on_request processing
                request_data = await self.on_request(request_data, trace_enabled=trace_enabled)
                # Re-serialize the modified payload back to bytes
                request_body = json.dumps(request_data).encode("utf-8")
            except json.JSONDecodeError:
                pass
            except Exception as exc:
                message = (
                    f"SessionServer request hook failed: {type(exc).__name__}: {exc}; "
                    f"request_id={request_id}; session_id={session_id}; path={request.path}; "
                    f"payload_summary={_short_repr(_payload_summary(request_data), 1024)}"
                )
                logger.error(message)
                return web.json_response(_lmdeploy_error_payload(message), status=500)

        # Build forwarding headers, dropping original Host
        forward_headers = dict(request.headers)
        forward_headers.pop("Host", None)
        forward_headers.pop("host", None)
        forward_headers.pop("Content-Length", None)
        forward_headers.pop("content-length", None)

        # Re-build Path
        req_path = request.match_info["path"]
        target_url = f"{self.worker_base_url}/{req_path.lstrip('/')}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        is_stream = request_data.get("stream", False) if request_data else False
        logger.info(
            "[SessionServer:req_start] "
            f"request_id={request_id} session_id={session_id} method={request.method} path={request.path} "
            f"target_url={target_url} stream={is_stream} trace_enabled={trace_enabled} "
            f"payload_summary={_short_repr(_payload_summary(request_data), 1024)}"
        )

        def _clean_data(data: dict) -> bool:
            modified = False
            for key, drop in [
                ("output_token_logprobs", not orig_return_logprob),
                ("output_ids", not orig_return_token_ids),
                ("routed_experts", not orig_return_routed_experts),
            ]:
                if drop and key in data:
                    data.pop(key)
                    modified = True
                if drop:
                    for c in data.get("choices", []):
                        if key in c:
                            c.pop(key)
                            modified = True

            for c in data.get("choices", []):
                if "logprobs" in c:
                    c.pop("logprobs")
                    modified = True

            for c in data.get("choices", []):
                if c.get("message") and isinstance(c["message"].get("content"), str):
                    if self.stop_word in c["message"]["content"]:
                        c["message"]["content"] = c["message"]["content"].replace(self.stop_word, "")
                        modified = True
                if c.get("delta") and isinstance(c["delta"].get("content"), str):
                    if self.stop_word in c["delta"]["content"]:
                        c["delta"]["content"] = c["delta"]["content"].replace(self.stop_word, "")
                        modified = True

            return modified

        # Forward the request to the upstream worker
        # read_bufsize controls StreamReader's line buffer limit; SSE events with large
        # tool_calls/reasoning_content payloads can exceed the 64KB default and trigger
        # "Chunk too big" from readuntil(b"\n").
        timeout = ClientTimeout(total=self.request_timeout, sock_connect=30)
        response = None
        raw_response = b""
        stream_chunk_count = 0
        stream_byte_count = 0
        stream_nonfinal_byte_count = 0
        stream_final_byte_count = 0
        stream_max_event_bytes = 0
        stream_max_event_summary = None
        stream_routed_experts_event_count = 0
        stream_total_output_ids = 0
        stream_total_output_logprobs = 0
        stream_saw_done = False
        stream_terminal_finish_seen = False
        stream_last_event_summary = None
        stream_last_chunk_at = None
        stream_max_inter_chunk_gap = 0.0
        stream_last_inter_chunk_gap = None
        downstream_write_max_s = 0.0
        downstream_write_slow_count = 0
        upstream_status = None
        stream_finished = False
        stall_log_interval_s = float(os.environ.get("XTUNER_SESSION_SERVER_STALL_LOG_S", "60"))
        slow_write_log_s = float(os.environ.get("XTUNER_SESSION_SERVER_SLOW_WRITE_LOG_S", "5"))
        try:
            async with ClientSession(read_bufsize=self.read_bufsize, timeout=timeout) as client:
                async with client.request(
                    method=request.method, url=target_url, headers=forward_headers, data=request_body
                ) as resp:
                    upstream_status = resp.status
                    logger.info(
                        "[SessionServer:upstream_open] "
                        f"request_id={request_id} session_id={session_id} status={resp.status} "
                        f"elapsed={time.monotonic() - started_at:.2f}s target_url={target_url}"
                    )
                    # Setup proper stream vs sync response objects
                    if is_stream:
                        response_chunks = []
                        response = web.StreamResponse(
                            status=resp.status,
                            headers={
                                k: v
                                for k, v in resp.headers.items()
                                if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
                            },
                        )
                        await response.prepare(request)
                        # If the downstream client closes the socket mid-stream,
                        # stop writing. For backends where abandoned generations keep
                        # consuming GPU, close the upstream stream instead of draining
                        # it only for trace completeness.
                        client_alive = True
                        next_progress_at = started_at + 300.0
                        async def _stall_watchdog() -> None:
                            next_log_at = time.monotonic() + stall_log_interval_s
                            while not stream_finished:
                                await asyncio.sleep(max(1.0, stall_log_interval_s / 2))
                                now = time.monotonic()
                                if now < next_log_at:
                                    continue
                                last_age = now - stream_last_chunk_at if stream_last_chunk_at is not None else None
                                if stream_last_chunk_at is None or last_age >= stall_log_interval_s:
                                    logger.warning(
                                        "[SessionServer:stream_stall] "
                                        f"request_id={request_id} session_id={session_id} "
                                        f"elapsed={now - started_at:.2f}s last_chunk_age="
                                        f"{None if last_age is None else round(last_age, 3)} "
                                        f"chunks={stream_chunk_count} bytes={stream_byte_count} "
                                        f"nonfinal_bytes={stream_nonfinal_byte_count} final_bytes={stream_final_byte_count} "
                                        f"saw_done={stream_saw_done} terminal_finish={stream_terminal_finish_seen} "
                                        f"client_alive={client_alive} upstream_status={upstream_status} "
                                        f"max_gap={round(stream_max_inter_chunk_gap, 3)} "
                                        f"last_event={_short_repr(stream_last_event_summary, 1024)}"
                                    )
                                    next_log_at = now + stall_log_interval_s

                        watchdog_task = asyncio.create_task(_stall_watchdog())
                        try:
                            async for line in resp.content:
                                now = time.monotonic()
                                if stream_last_chunk_at is not None:
                                    stream_last_inter_chunk_gap = now - stream_last_chunk_at
                                    stream_max_inter_chunk_gap = max(
                                        stream_max_inter_chunk_gap, stream_last_inter_chunk_gap
                                    )
                                stream_chunk_count += 1
                                stream_byte_count += len(line)
                                stream_last_chunk_at = now

                                stripped_line = line.strip()
                                line_is_final_event = False
                                if stripped_line == b"data: [DONE]":
                                    stream_saw_done = True
                                elif line.startswith(b"data: "):
                                    try:
                                        event = json.loads(line.decode("utf-8")[6:])
                                        stream_last_event_summary = _stream_event_summary(event)
                                        if len(line) > stream_max_event_bytes:
                                            stream_max_event_bytes = len(line)
                                            stream_max_event_summary = stream_last_event_summary
                                        choices = event.get("choices") if isinstance(event, dict) else None
                                        if isinstance(choices, list):
                                            if any(choice.get("finish_reason") for choice in choices if isinstance(choice, dict)):
                                                line_is_final_event = True
                                                stream_terminal_finish_seen = True
                                            for choice in choices:
                                                if not isinstance(choice, dict):
                                                    continue
                                                stream_total_output_ids += len(choice.get("output_ids") or [])
                                                stream_total_output_logprobs += len(
                                                    choice.get("output_token_logprobs") or []
                                                )
                                                if choice.get("routed_experts") is not None:
                                                    stream_routed_experts_event_count += 1
                                        if _is_error_payload(event):
                                            logger.error(
                                                "[SessionServer:upstream_error_event] "
                                                f"request_id={request_id} session_id={session_id} "
                                                f"elapsed={now - started_at:.2f}s chunks={stream_chunk_count} "
                                                f"bytes={stream_byte_count} upstream_status={upstream_status} "
                                                f"event={_short_repr(stream_last_event_summary, 2048)} "
                                                f"target_url={target_url}"
                                            )
                                    except Exception:
                                        stream_last_event_summary = {
                                            "parse_error": line[:256].decode("utf-8", errors="replace")
                                        }

                                if line_is_final_event:
                                    stream_final_byte_count += len(line)
                                elif stripped_line != b"data: [DONE]":
                                    stream_nonfinal_byte_count += len(line)

                                if now >= next_progress_at:
                                    logger.info(
                                        "[SessionServer:stream_progress] "
                                        f"request_id={request_id} session_id={session_id} "
                                        f"elapsed={now - started_at:.2f}s chunks={stream_chunk_count} "
                                        f"bytes={stream_byte_count} nonfinal_bytes={stream_nonfinal_byte_count} "
                                        f"final_bytes={stream_final_byte_count} saw_done={stream_saw_done} "
                                        f"terminal_finish={stream_terminal_finish_seen} client_alive={client_alive} "
                                        f"upstream_status={upstream_status} "
                                        f"last_gap={None if stream_last_inter_chunk_gap is None else round(stream_last_inter_chunk_gap, 3)} "
                                        f"max_gap={round(stream_max_inter_chunk_gap, 3)} "
                                        f"output_ids={stream_total_output_ids} logprobs={stream_total_output_logprobs} "
                                        f"routed_events={stream_routed_experts_event_count} "
                                        f"last_event={_short_repr(stream_last_event_summary, 1024)}"
                                    )
                                    next_progress_at = now + 300.0

                                # Keep unmodified line for trace store parsing
                                if trace_enabled:
                                    response_chunks.append(line)

                                # Dynamically prune added fields before writing to client
                                if request_data is not None and line.startswith(b"data: ") and stripped_line != b"data: [DONE]":
                                    try:
                                        text = line.decode("utf-8")
                                        data = json.loads(text[6:])
                                        modified = False
                                        if _is_error_payload(data):
                                            data.setdefault(
                                                "session_server",
                                                {
                                                    "request_id": request_id,
                                                    "session_id": session_id,
                                                    "target_url": target_url,
                                                    "upstream_status": upstream_status,
                                                    "elapsed": round(time.monotonic() - started_at, 3),
                                                    "stream_chunks": stream_chunk_count,
                                                    "stream_bytes": stream_byte_count,
                                                    "stream_nonfinal_bytes": stream_nonfinal_byte_count,
                                                    "stream_final_bytes": stream_final_byte_count,
                                                    "stream_saw_done": stream_saw_done,
                                                    "terminal_finish": stream_terminal_finish_seen,
                                                    "last_event": stream_last_event_summary,
                                                },
                                            )
                                            modified = True
                                        if _clean_data(data):
                                            modified = True
                                        if modified:
                                            line = ("data: " + json.dumps(data) + "\n").encode("utf-8")
                                    except Exception:
                                        pass

                                # Delay [DONE] only while a training trace still needs to be exported.
                                if client_alive and (not trace_enabled or stripped_line != b"data: [DONE]"):
                                    try:
                                        write_started_at = time.monotonic()
                                        await response.write(line)
                                        write_elapsed = time.monotonic() - write_started_at
                                        downstream_write_max_s = max(downstream_write_max_s, write_elapsed)
                                        if write_elapsed >= slow_write_log_s:
                                            downstream_write_slow_count += 1
                                            logger.warning(
                                                "[SessionServer:slow_downstream_write] "
                                                f"request_id={request_id} session_id={session_id} "
                                                f"elapsed={now - started_at:.2f}s write_elapsed={write_elapsed:.3f}s "
                                                f"line_bytes={len(line)} chunks={stream_chunk_count} "
                                                f"client_alive={client_alive}"
                                            )
                                    except (ConnectionError, ClientConnectionResetError):
                                        client_alive = False
                                        logger.warning(
                                            "[SessionServer:client_disconnect] "
                                            f"request_id={request_id} session_id={session_id} "
                                            f"elapsed={now - started_at:.2f}s chunks={stream_chunk_count} "
                                            f"bytes={stream_byte_count} nonfinal_bytes={stream_nonfinal_byte_count} "
                                            f"final_bytes={stream_final_byte_count} saw_done={stream_saw_done} "
                                            f"terminal_finish={stream_terminal_finish_seen} "
                                            f"last_event={_short_repr(stream_last_event_summary, 1024)} "
                                            f"drain_upstream={self.drain_upstream_on_client_disconnect}"
                                        )
                                        if not self.drain_upstream_on_client_disconnect:
                                            break
                        finally:
                            stream_finished = True
                            watchdog_task.cancel()
                            try:
                                await watchdog_task
                            except asyncio.CancelledError:
                                pass

                        raw_response = b"".join(response_chunks) if trace_enabled else b""
                    else:
                        raw_response = await resp.read()
                        final_raw_response = raw_response

                        if request_data is not None:
                            try:
                                clean_data = json.loads(raw_response)
                                if _clean_data(clean_data):
                                    final_raw_response = json.dumps(clean_data).encode("utf-8")
                            except Exception:
                                pass

                        response = web.Response(
                            status=resp.status,
                            headers={
                                k: v
                                for k, v in resp.headers.items()
                                if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
                            },
                            body=final_raw_response,  # Modified raw response without our injected trace params
                        )
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            status = 504 if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else 500
            context = {
                "request_id": request_id,
                "session_id": session_id,
                "method": request.method,
                "path": request.path,
                "target_url": target_url,
                "is_stream": is_stream,
                "trace_enabled": trace_enabled,
                "elapsed": round(elapsed, 3),
                "upstream_status": upstream_status,
                "stream_chunks": stream_chunk_count,
                "stream_bytes": stream_byte_count,
                "stream_nonfinal_bytes": stream_nonfinal_byte_count,
                "stream_final_bytes": stream_final_byte_count,
                "stream_max_event_bytes": stream_max_event_bytes,
                "stream_max_event": stream_max_event_summary,
                "stream_routed_experts_events": stream_routed_experts_event_count,
                "stream_output_ids": stream_total_output_ids,
                "stream_output_token_logprobs": stream_total_output_logprobs,
                "stream_saw_done": stream_saw_done,
                "stream_terminal_finish_seen": stream_terminal_finish_seen,
                "last_chunk_age": round(elapsed - (stream_last_chunk_at - started_at), 3)
                if stream_last_chunk_at is not None
                else None,
                "last_inter_chunk_gap": round(stream_last_inter_chunk_gap, 3)
                if stream_last_inter_chunk_gap is not None
                else None,
                "max_inter_chunk_gap": round(stream_max_inter_chunk_gap, 3),
                "downstream_write_max_s": round(downstream_write_max_s, 3),
                "downstream_write_slow_count": downstream_write_slow_count,
                "last_event": stream_last_event_summary,
                "payload_summary": _payload_summary(request_data),
            }
            message = f"SessionServer forwarding failed: {type(exc).__name__}: {exc}; context={_short_repr(context, 4096)}"
            logger.exception(message)
            error_payload = _lmdeploy_error_payload(message, status=status)
            if is_stream and response is not None and getattr(response, "prepared", False):
                try:
                    await response.write(("data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    await response.write_eof()
                    return response
                except (ConnectionError, ClientConnectionResetError):
                    pass
            return web.json_response(error_payload, status=status)

        logger.info(
            "[SessionServer:req_done] "
            f"request_id={request_id} session_id={session_id} status={upstream_status} "
            f"stream={is_stream} elapsed={time.monotonic() - started_at:.2f}s "
            f"chunks={stream_chunk_count} bytes={stream_byte_count} "
            f"nonfinal_bytes={stream_nonfinal_byte_count} final_bytes={stream_final_byte_count} "
            f"max_event_bytes={stream_max_event_bytes} routed_events={stream_routed_experts_event_count} "
            f"output_ids={stream_total_output_ids} logprobs={stream_total_output_logprobs} "
            f"saw_done={stream_saw_done} terminal_finish={stream_terminal_finish_seen} "
            f"max_gap={round(stream_max_inter_chunk_gap, 3)} "
            f"downstream_write_max_s={round(downstream_write_max_s, 3)} "
            f"downstream_write_slow_count={downstream_write_slow_count} "
            f"max_event={_short_repr(stream_max_event_summary, 1024)} "
            f"last_event={_short_repr(stream_last_event_summary, 1024)}"
        )

        # Apply abstract on_response processing
        response_data = None
        skip_done = bool(is_stream and not trace_enabled)
        session_error_msg = None
        if request_data and trace_enabled:
            if is_stream:
                skip_done = not _stream_has_traceable_choices(raw_response)
                if not skip_done:
                    try:
                        response_data = self._parse_stream_response(raw_response)
                    except Exception as exc:
                        session_error_msg = f"SessionServer stream trace failed: {type(exc).__name__}: {exc}"
            else:
                try:
                    response_data = json.loads(raw_response)
                except json.JSONDecodeError:
                    pass
                if isinstance(response_data, dict) and _is_error_payload(response_data):
                    response_data = None

            if response_data is not None:
                try:
                    for c in response_data.get("choices", []):
                        if c.get("message") and isinstance(c["message"].get("content"), str):
                            c["message"]["content"] = c["message"]["content"].replace(self.stop_word, "")

                    response_data["session_id"] = session_id
                    response_data["messages"] = messages
                    response_data["tools"] = tools
                    await self.on_response(response_data, trace_enabled=trace_enabled)
                except Exception as exc:
                    session_error_msg = f"SessionServer response hook failed: {type(exc).__name__}: {exc}"

        if session_error_msg:
            get_logger().error(session_error_msg)

        if is_stream:
            try:
                if session_error_msg:
                    error_payload = _lmdeploy_error_payload(session_error_msg)
                    await response.write(
                        ("data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n").encode("utf-8")
                    )
                    skip_done = True
                if not skip_done:
                    await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            except (ConnectionError, ClientConnectionResetError):
                # Client already gone; trace was still recorded above.
                pass
        elif session_error_msg:
            return web.json_response(_lmdeploy_error_payload(session_error_msg), status=500)

        return response

    async def _decode_routed_experts(self, routed_experts: Any) -> np.ndarray:
        if isinstance(routed_experts, str):
            # SGLang returns routed experts as a base64-encoded int32 tensor.
            # LMDeploy returns a Ray shared-store key string. Try the SGLang
            # format first and fall back to LMDeploy shared_store lookup.
            try:
                if self.routed_experts_num_hidden_layers and self.routed_experts_num_experts_per_tok:
                    routed_experts_flat = np.frombuffer(base64.b64decode(routed_experts, validate=True), dtype=np.int32)
                    routed_experts_array = routed_experts_flat.reshape(
                        -1,
                        self.routed_experts_num_hidden_layers,
                        self.routed_experts_num_experts_per_tok,
                    )
                    return routed_experts_array.copy()
            except Exception:
                pass

            if self._lmdeploy_actor is None:
                self._lmdeploy_actor = ray.get_actor("shared_store", namespace="lmdeploy")
            assert self._lmdeploy_actor is not None, "LMDeploy actor should be available in the shared store."
            routed_experts_data = await self._lmdeploy_actor.get.remote(routed_experts)
            return np.asarray(routed_experts_data)
        return np.asarray(routed_experts)

    @staticmethod
    def _parse_stream_response(raw: bytes) -> Optional[dict]:
        """Parse SSE stream to reconstruct the complete final message state."""
        text = raw.decode("utf-8", errors="replace")
        events = []
        saw_done = False
        for line in text.split("\n"):
            line = line.strip()
            if line == "data: [DONE]":
                saw_done = True
                continue
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if _is_error_payload(event):
                    raise RuntimeError(f"Upstream SSE stream returned error: {json.dumps(event, ensure_ascii=False)}")
                events.append(event)

        if not events:
            return None
        if not any(event.get("choices") for event in events):
            raise RuntimeError(f"Upstream SSE stream ended without choices: {json.dumps(events, ensure_ascii=False)}")

        # Reconstruct standard stream output (Assuming OpenAI format here)
        message: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}

        for event in events:
            if event.get("id") and "id" not in message:
                message["id"] = event["id"]
            if event.get("model"):
                message["model"] = event["model"]

            choices = event.get("choices", [])
            for choice in choices:
                if choice.get("finish_reason") == "error":
                    raise RuntimeError(
                        f"Upstream SSE choice finished with error: {json.dumps(event, ensure_ascii=False)}"
                    )
                delta = choice.get("delta", {})

                # Check text content
                if delta.get("content"):
                    content_parts.append(delta["content"])

                # Check output ids
                if choice.get("output_ids") is not None:
                    assistant_choice = message["choices"][0]
                    if "output_ids" not in assistant_choice:
                        assistant_choice["output_ids"] = []
                    assistant_choice["output_ids"].extend(choice["output_ids"])

                # Check routed experts. LMDeploy only emits this in the final
                # chunk, often as a Ray shared-store key string.
                if choice.get("routed_experts") is not None:
                    assistant_choice = message["choices"][0]
                    assistant_choice["routed_experts"] = choice["routed_experts"]

                # Check raw output logprobs from LMDeploy return_logprob protocol.
                if choice.get("output_token_logprobs") is not None:
                    assistant_choice = message["choices"][0]
                    if "output_token_logprobs" not in assistant_choice:
                        assistant_choice["output_token_logprobs"] = []
                    assistant_choice["output_token_logprobs"].extend(choice["output_token_logprobs"])

                # Check reasoning content
                if delta.get("reasoning_content"):
                    assistant_msg = message["choices"][0]["message"]
                    assistant_msg["reasoning_content"] = (
                        assistant_msg.get("reasoning_content", "") + delta["reasoning_content"]
                    )

                # Check tool calls
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": tc_delta.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_map[idx]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]

                if choice.get("finish_reason"):
                    message["choices"][0]["finish_reason"] = choice["finish_reason"]

            if event.get("usage") is not None:
                usage = event["usage"]

        msg = message["choices"][0]["message"]
        msg["content"] = "".join(content_parts)
        if tool_calls_map:
            msg["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
        if usage:
            message["usage"] = usage

        assistant_choice = message["choices"][0]
        if not saw_done:
            raise RuntimeError("Upstream SSE stream ended without [DONE].")
        if not assistant_choice.get("finish_reason"):
            raise RuntimeError("Upstream SSE stream ended without terminal finish_reason.")
        if assistant_choice.get("output_ids") is None:
            raise RuntimeError("Upstream SSE stream ended without output_ids.")

        return message


class SessionServerActor:
    """Ray actor wrapper that owns one SessionServer instance."""

    def __init__(
        self,
        worker_base_url: str,
        tokenizer_path: str,
        host: str,
        port: int,
        request_timeout: float,
        max_model_len: Optional[int] = None,
        max_model_len_reserved_tokens: int = 0,
        normalize_sglang_sampling_params: bool = False,
        drain_upstream_on_client_disconnect: bool = True,
        enable_return_routed_experts: bool = True,
    ):
        self.worker_base_url = worker_base_url
        self.tokenizer_path = tokenizer_path
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.max_model_len = max_model_len
        self.max_model_len_reserved_tokens = max_model_len_reserved_tokens
        self.normalize_sglang_sampling_params = normalize_sglang_sampling_params
        self.drain_upstream_on_client_disconnect = drain_upstream_on_client_disconnect
        self.enable_return_routed_experts = enable_return_routed_experts
        self.server: SessionServer | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> str:
        if self.server is not None:
            return self.url

        self.server = SessionServer(
            worker_base_url=self.worker_base_url,
            tokenizer_path=self.tokenizer_path,
            host=self.host,
            port=self.port,
            request_timeout=self.request_timeout,
            max_model_len=self.max_model_len,
            max_model_len_reserved_tokens=self.max_model_len_reserved_tokens,
            normalize_sglang_sampling_params=self.normalize_sglang_sampling_params,
            drain_upstream_on_client_disconnect=self.drain_upstream_on_client_disconnect,
            enable_return_routed_experts=self.enable_return_routed_experts,
        )
        await self.server.start()
        return self.server.url

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.stop()
            self.server = None
