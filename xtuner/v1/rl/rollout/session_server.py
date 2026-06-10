import json
import time
from functools import reduce
from operator import add
from typing import Any, Optional

import numpy as np
import ray
from aiohttp import ClientConnectionResetError, ClientSession, ClientTimeout, web

from transformers import AutoTokenizer
from xtuner.v1.utils import get_logger

from .chat_template import canonicalize_messages_for_chat_template
from .otel import begin_span, end_span, extract_context, inject_context, set_attrs, start_span, use_context
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


_SESSION_SERVER_ONLY_KEYS = {"session_id", "_otel_trace_context"}


def _bool_request_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _request_uses_trace_store(req_body: dict) -> bool:
    return _bool_request_value(req_body.get("return_token_ids"), True)


def _list_len(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def _choices_output_ids_len(data: dict) -> int:
    total = 0
    for choice in data.get("choices") or []:
        output_ids = choice.get("output_ids")
        if isinstance(output_ids, list):
            total += len(output_ids)
    return total


def _response_output_ids_len(data: dict) -> int | None:
    output_ids = data.get("output_ids")
    if isinstance(output_ids, list):
        return len(output_ids)
    total = _choices_output_ids_len(data)
    return total if total > 0 else None


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
    ):
        self.worker_base_url = worker_base_url.rstrip("/")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.read_bufsize = read_bufsize
        self.store = get_store()
        self.stop_word = self.tokenizer.eos_token or ""

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._lmdeploy_actor: Optional[ray.actor.ActorHandle] = None

    async def on_request(self, req_body: dict, *, trace_enabled: bool = True) -> dict:
        """Hook for processing/modifying the request before forwarding."""

        with start_span(
            "xtuner.session_server.on_request",
            session_id=req_body.get("session_id"),
            trace_store_enabled=trace_enabled,
            messages=len(req_body.get("messages") or []) if isinstance(req_body.get("messages"), list) else None,
            tools=len(req_body.get("tools") or []) if isinstance(req_body.get("tools"), list) else None,
        ) as span:
            return await self._on_request_impl(req_body, trace_enabled=trace_enabled, span=span)

    async def _on_request_impl(self, req_body: dict, *, trace_enabled: bool, span: Any = None) -> dict:
        if not trace_enabled:
            worker_req = {k: v for k, v in req_body.items() if k not in _SESSION_SERVER_ONLY_KEYS}
            if "logprobs" in worker_req:
                worker_req.setdefault("return_logprob", worker_req.pop("logprobs"))
            if not _bool_request_value(worker_req.get("return_logprob"), False):
                worker_req.pop("top_logprobs", None)
                worker_req["return_logprob"] = False
            worker_req["return_token_ids"] = False
            worker_req.setdefault("return_routed_experts", True)
            return worker_req

        session_id = req_body["session_id"]
        # 1. chat_template render 出完整 prompt string，不 tokenize 全量
        with start_span("xtuner.session_server.apply_chat_template", session_id=session_id):
            prompt_text = self.tokenizer.apply_chat_template(
                canonicalize_messages_for_chat_template(req_body["messages"]),
                tools=req_body.get("tools", None),
                add_generation_prompt=True,
                tokenize=False,
            )
        set_attrs(span, prompt_chars=len(prompt_text))

        # 2. Store 做 string prefix match。
        with start_span("xtuner.session_server.trace_store.search_prompt", session_id=session_id):
            prefix, nodes = await self.store.search.remote(session_id, prompt_text, filter_none=True)
        if prefix:
            get_logger().debug(f"Hit prefix cache for session {session_id}")
        delta, delta_ids = prompt_text[len(prefix) :], []
        if delta:
            with start_span("xtuner.session_server.tokenize_delta", session_id=session_id, delta_chars=len(delta)):
                delta_ids = self.tokenizer.encode(delta, add_special_tokens=False)
            with start_span(
                "xtuner.session_server.trace_store.insert_prompt_delta",
                session_id=session_id,
                delta_tokens=len(delta_ids),
            ):
                await self.store.insert.remote(
                    session_id, prompt_text, TokenizedSegment(text=delta, token_ids=delta_ids)
                )
        input_ids = reduce(add, [node.value.token_ids for node in nodes] + [delta_ids])
        set_attrs(span, prefix_chars=len(prefix), delta_chars=len(delta), input_tokens=len(input_ids))

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
            "return_routed_experts": True,
            "return_logprob": True,
            "include_stop_str_in_output": True,
        }
        return worker_req

    async def on_response(self, worker_resp: dict, *, trace_enabled: bool = True) -> dict:
        """Hook for processing the parsed response received from the worker."""

        with start_span(
            "xtuner.session_server.on_response",
            session_id=worker_resp.get("session_id"),
            trace_store_enabled=trace_enabled,
        ) as span:
            return await self._on_response_impl(worker_resp, trace_enabled=trace_enabled, span=span)

    async def _on_response_impl(self, worker_resp: dict, *, trace_enabled: bool, span: Any = None) -> dict:
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
        message = choice.get("message") or {}
        set_attrs(
            span,
            output_tokens=len(output_token_ids),
            response_chars=len(message.get("content") or ""),
            finish_reason=choice.get("finish_reason"),
        )
        output_logprobs = _extract_output_logprobs(choice, output_token_ids)
        raw_routed_expert = choice.get("routed_experts")  # 本次 call 的 raw routed_expert，可为 None

        # 2. Store 把 input_delta / assistant_output 两个节点补齐字段。
        with start_span("xtuner.session_server.apply_old_prompt_template", session_id=session_id):
            old_prompt = self.tokenizer.apply_chat_template(
                canonicalize_messages_for_chat_template(messages),
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        messages = [*messages, choice["message"]]
        with start_span("xtuner.session_server.apply_new_prompt_template", session_id=session_id):
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
            with start_span("xtuner.session_server.decode_routed_experts", session_id=session_id):
                raw_routed_expert = await self._decode_routed_experts(raw_routed_expert)
            if len(raw_routed_expert) > 0:
                num_layers = raw_routed_expert.shape[1]
                topk_experts = raw_routed_expert.shape[2]
                dummy_expert = np.full((1, num_layers, topk_experts), 0, dtype=raw_routed_expert.dtype)
                raw_routed_expert = np.concatenate([dummy_expert, raw_routed_expert], axis=0)

            with start_span("xtuner.session_server.trace_store.search_old_prompt", session_id=session_id):
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
                    with start_span(
                        "xtuner.session_server.trace_store.update_prompt_delta",
                        session_id=session_id,
                        delta_tokens=delta_len,
                    ):
                        await self.store.insert.remote(session_id, old_prompt, delta_node_val)

                raw_routed_expert = ray.put(response_expert)
            else:
                raw_routed_expert = ray.put(raw_routed_expert)

        with start_span(
            "xtuner.session_server.trace_store.insert_response",
            session_id=session_id,
            output_tokens=len(output_token_ids),
        ):
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

        request_body = await request.read()
        body_trace_context = None
        if request_body:
            try:
                body_data = json.loads(request_body)
                body_trace_context = body_data.get("_otel_trace_context") if isinstance(body_data, dict) else None
            except json.JSONDecodeError:
                body_trace_context = None

        traceparent_header = request.headers.get("traceparent")
        traceparent_body = None
        if isinstance(body_trace_context, dict):
            traceparent_body = body_trace_context.get("traceparent")
        parent_context_source = "header" if traceparent_header else "none"
        parent_context = extract_context(request.headers)
        if traceparent_body:
            parent_context = extract_context(body_trace_context)
            parent_context_source = "body"

        with use_context(parent_context):
            return await self._handle_request_impl(
                request,
                request_body=request_body,
                traceparent_header_present=bool(traceparent_header),
                traceparent_body_present=bool(traceparent_body),
                traceparent_context_source=parent_context_source,
            )

    async def _handle_request_impl(
        self,
        request: web.Request,
        request_body: bytes | None = None,
        request_span: Any = None,
        traceparent_header_present: bool = False,
        traceparent_body_present: bool = False,
        traceparent_context_source: str = "none",
    ) -> web.Response:
        """Proxy handler for the worker API."""

        # Read the request body
        if request_body is None:
            with start_span("xtuner.session_server.read_request_body"):
                request_body = await request.read()
        else:
            with start_span("xtuner.session_server.read_request_body", cached=True):
                pass
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
                body_trace_context = request_data.pop("_otel_trace_context", None)
                set_attrs(
                    request_span,
                    session_id=session_id,
                    trace_store_enabled=trace_enabled,
                    request_bytes=len(request_body),
                    messages=len(messages) if isinstance(messages, list) else None,
                    tools=len(tools) if isinstance(tools, list) else None,
                    body_trace_context_present=isinstance(body_trace_context, dict),
                )

                # Apply purely abstract on_request processing
                request_data = await self.on_request(request_data, trace_enabled=trace_enabled)
                input_ids = request_data.get("input_ids") if isinstance(request_data, dict) else None
                set_attrs(
                    request_span,
                    input_tokens=_list_len(input_ids),
                    max_tokens=request_data.get("max_tokens") if isinstance(request_data, dict) else None,
                )
                # Re-serialize the modified payload back to bytes
                request_body = json.dumps(request_data).encode("utf-8")
            except json.JSONDecodeError:
                pass
            except Exception as exc:
                message = f"SessionServer request hook failed: {type(exc).__name__}: {exc}"
                get_logger().error(message)
                return web.json_response(_lmdeploy_error_payload(message), status=500)

        # Build forwarding headers, dropping original Host
        forward_headers = dict(request.headers)
        forward_headers.pop("Host", None)
        forward_headers.pop("host", None)
        forward_headers.pop("Content-Length", None)
        forward_headers.pop("content-length", None)
        inject_context(forward_headers)

        # Re-build Path
        req_path = request.match_info["path"]
        target_url = f"{self.worker_base_url}/{req_path.lstrip('/')}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        is_stream = request_data.get("stream", False) if request_data else False
        input_tokens = _list_len(request_data.get("input_ids")) if request_data else None
        max_tokens = request_data.get("max_tokens") if request_data else None
        set_attrs(
            request_span,
            target_url=target_url,
            stream=is_stream,
            input_tokens=input_tokens,
            max_tokens=max_tokens,
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
        forward_span = begin_span(
            "xtuner.session_server.forward_worker",
            target_url=target_url,
            stream=is_stream,
            request_bytes=len(request_body) if request_body else 0,
            timeout_s=self.request_timeout,
            input_tokens=input_tokens,
            max_tokens=max_tokens,
            model=request_data.get("model") if request_data else None,
            http_method=request.method,
            http_path=request.path,
            worker_base_url=self.worker_base_url,
            traceparent_header_present=traceparent_header_present,
            traceparent_body_present=traceparent_body_present,
            traceparent_context_source=traceparent_context_source,
        )
        try:
            async with ClientSession(read_bufsize=self.read_bufsize, timeout=timeout) as client:
                async with client.request(
                    method=request.method, url=target_url, headers=forward_headers, data=request_body
                ) as resp:
                    set_attrs(forward_span, http_status=resp.status)
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
                        # If the downstream client closes the socket mid-stream
                        # (e.g. AsyncAPIClient bails out on a finish_reason=='error'
                        # chunk after the prompt overflowed the session window),
                        # keep draining the upstream so the trace is still recorded
                        # in full but stop attempting to write to the closed socket.
                        client_alive = True
                        stream_span = begin_span(
                            "xtuner.session_server.stream_read",
                            target_url=target_url,
                            input_tokens=input_tokens,
                            max_tokens=max_tokens,
                        )
                        stream_start = time.perf_counter()
                        first_chunk_ms = None
                        first_output_token_ms = None
                        first_content_ms = None
                        chunk_count = 0
                        raw_response_bytes = 0
                        output_tokens = 0
                        usage_prompt_tokens = None
                        usage_completion_tokens = None
                        usage_total_tokens = None
                        finish_reason = None
                        try:
                            async for line in resp.content:
                                chunk_count += 1
                                raw_response_bytes += len(line)
                                if first_chunk_ms is None:
                                    first_chunk_ms = (time.perf_counter() - stream_start) * 1000
                                # Keep unmodified line for trace store parsing
                                if trace_enabled:
                                    response_chunks.append(line)

                                # Dynamically prune added fields before writing to client
                                if (
                                    request_data is not None
                                    and line.startswith(b"data: ")
                                    and line.strip() != b"data: [DONE]"
                                ):
                                    try:
                                        text = line.decode("utf-8")
                                        data = json.loads(text[6:])
                                        event_output_tokens = _choices_output_ids_len(data)
                                        if event_output_tokens > 0 and first_output_token_ms is None:
                                            first_output_token_ms = (time.perf_counter() - stream_start) * 1000
                                        output_tokens += event_output_tokens
                                        usage = data.get("usage")
                                        if isinstance(usage, dict):
                                            usage_prompt_tokens = usage.get("prompt_tokens", usage_prompt_tokens)
                                            usage_completion_tokens = usage.get(
                                                "completion_tokens", usage_completion_tokens
                                            )
                                            usage_total_tokens = usage.get("total_tokens", usage_total_tokens)
                                        for choice in data.get("choices") or []:
                                            delta = choice.get("delta") or {}
                                            if delta.get("content") and first_content_ms is None:
                                                first_content_ms = (time.perf_counter() - stream_start) * 1000
                                            if choice.get("finish_reason"):
                                                finish_reason = choice.get("finish_reason")
                                        if _clean_data(data):
                                            line = ("data: " + json.dumps(data) + "\n").encode("utf-8")
                                    except Exception:
                                        pass

                                # Delay [DONE] only while a training trace still needs to be exported.
                                if client_alive and (not trace_enabled or line.strip() != b"data: [DONE]"):
                                    try:
                                        await response.write(line)
                                    except (ConnectionError, ClientConnectionResetError):
                                        client_alive = False
                        finally:
                            end_span(
                                stream_span,
                                first_chunk_ms=first_chunk_ms,
                                first_output_token_ms=first_output_token_ms,
                                first_content_ms=first_content_ms,
                                chunks=chunk_count,
                                raw_response_bytes=raw_response_bytes,
                                output_tokens=output_tokens if output_tokens > 0 else None,
                                prompt_tokens=usage_prompt_tokens,
                                completion_tokens=usage_completion_tokens,
                                total_tokens=usage_total_tokens,
                                finish_reason=finish_reason,
                                client_alive=client_alive,
                            )
                            set_attrs(
                                forward_span,
                                first_chunk_ms=first_chunk_ms,
                                first_output_token_ms=first_output_token_ms,
                                first_content_ms=first_content_ms,
                                output_tokens=output_tokens if output_tokens > 0 else None,
                                prompt_tokens=usage_prompt_tokens,
                                completion_tokens=usage_completion_tokens,
                                total_tokens=usage_total_tokens,
                                finish_reason=finish_reason,
                            )

                        raw_response = b"".join(response_chunks) if trace_enabled else b""
                    else:
                        with start_span("xtuner.session_server.read_response", target_url=target_url):
                            raw_response = await resp.read()
                        final_raw_response = raw_response
                        set_attrs(forward_span, response_bytes=len(raw_response))

                        if request_data is not None:
                            try:
                                clean_data = json.loads(raw_response)
                                usage = clean_data.get("usage") if isinstance(clean_data, dict) else None
                                set_attrs(
                                    forward_span,
                                    output_tokens=_response_output_ids_len(clean_data)
                                    if isinstance(clean_data, dict)
                                    else None,
                                    prompt_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                                    completion_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
                                    total_tokens=usage.get("total_tokens") if isinstance(usage, dict) else None,
                                )
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
            end_span(forward_span, exc=exc)
            raise
        else:
            end_span(forward_span, response_bytes=len(raw_response) if raw_response is not None else None)

        # Apply abstract on_response processing
        response_data = None
        skip_done = bool(is_stream and not trace_enabled)
        session_error_msg = None
        if request_data and trace_enabled:
            if is_stream:
                skip_done = not _stream_has_traceable_choices(raw_response)
                if not skip_done:
                    try:
                        with start_span(
                            "xtuner.session_server.parse_stream_response",
                            raw_response_bytes=len(raw_response),
                        ):
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

    def __init__(self, worker_base_url: str, tokenizer_path: str, host: str, port: int, request_timeout: float):
        self.worker_base_url = worker_base_url
        self.tokenizer_path = tokenizer_path
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
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
        )
        await self.server.start()
        return self.server.url

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.stop()
            self.server = None
