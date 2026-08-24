#!/usr/bin/env python3
"""Small Anthropic Messages compatibility proxy for Claude Code + SGLang.

The proxy keeps SGLang untouched.  It accepts Anthropic /v1/messages requests,
forwards them to SGLang as non-streaming requests, normalizes the returned
content blocks, and synthesizes Anthropic SSE when the client requested stream.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


THINK_RE = re.compile(r"<think>.*?</think>|</?think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def _normalize_content_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
    if tool_blocks:
        normalized_tools = []
        for block in tool_blocks:
            normalized_tools.append(
                {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                }
            )
        return normalized_tools

    normalized = []
    for block in blocks:
        block_type = block.get("type")
        if block_type in {"text", "thinking"}:
            text = str(block.get("text") or block.get("thinking") or "")
            text = _strip_thinking(text)
            if text:
                normalized.append({"type": "text", "text": text})
        else:
            normalized.append(block)
    return normalized or [{"type": "text", "text": ""}]


def _normalize_message_response(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    content = payload.get("content")
    if isinstance(content, list):
        payload["content"] = _normalize_content_blocks(
            [block for block in content if isinstance(block, dict)]
        )
    return payload


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_anthropic_events(payload: dict[str, Any]):
    message_id = payload.get("id") or f"msg_proxy_{int(time.time() * 1000)}"
    model = payload.get("model") or "sglang"
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    content = payload.get("content") if isinstance(payload.get("content"), list) else []

    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": int(usage.get("input_tokens") or 0),
                    "output_tokens": 0,
                },
            },
        },
    )

    for index, block in enumerate(content):
        block_type = block.get("type")
        if block_type == "tool_use":
            tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": {},
                    },
                },
            )
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(tool_input, ensure_ascii=False),
                    },
                },
            )
        else:
            text = str(block.get("text") or "")
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})

    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": payload.get("stop_reason"),
                "stop_sequence": payload.get("stop_sequence"),
            },
            "usage": {"output_tokens": int(usage.get("output_tokens") or 0)},
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


def create_app(upstream_base_url: str, timeout_s: float) -> FastAPI:
    app = FastAPI()
    upstream = upstream_base_url.rstrip("/")

    def upstream_url(path: str) -> str:
        if upstream.endswith("/v1"):
            return f"{upstream}{path}"
        return f"{upstream}/v1{path}"

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "upstream": upstream}

    @app.get("/v1/models")
    async def models(request: Request):
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(upstream_url("/models"), headers=dict(request.headers))
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        body = await request.json()
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(upstream_url("/messages/count_tokens"), headers=dict(request.headers), json=body)
        if resp.status_code == 404:
            return JSONResponse({"input_tokens": 0})
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        client_wants_stream = bool(body.get("stream"))
        upstream_body = dict(body)
        upstream_body["stream"] = False

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(upstream_url("/messages"), headers=dict(request.headers), json=upstream_body)

        if resp.status_code >= 400:
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

        payload = _normalize_message_response(resp.json())
        if client_wants_stream:
            return StreamingResponse(_stream_anthropic_events(payload), media_type="text/event-stream")
        return JSONResponse(payload)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-base-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    uvicorn.run(create_app(args.upstream_base_url, args.timeout_s), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
