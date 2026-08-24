#!/usr/bin/env python3
"""Send token-id LMDeploy requests and print ids/metadata."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

import requests
from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/"
    "models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://10.102.250.69:23333")
    parser.add_argument("--model-name", default="hha_xtuner_qwen35_35b")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--case", choices=["non-stream", "stream", "both"], default="both")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--return-logprob", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-full-json", action="store_true")
    return parser.parse_args()


def headers(args: argparse.Namespace) -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    if args.api_key:
        result["Authorization"] = f"Bearer {args.api_key}"
    return result


def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Get status, item list, and return eligibility for a customer order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order id, for example #W1234567.",
                        }
                    },
                    "required": ["order_id"],
                },
            },
        }
    ]


def source_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a customer support assistant. Use tools whenever tool data is needed.",
        },
        {
            "role": "user",
            "content": "Please check whether order #W1234567 can be returned. Use the available tool.",
        },
    ]


def build_input_ids(args: argparse.Namespace) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        source_messages(),
        tools=tool_schema(),
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def build_payload(args: argparse.Namespace, input_ids: list[int], stream: bool) -> dict[str, Any]:
    # This mirrors xtuner/v1/rl/rollout/session_server.py: messages is empty,
    # input_ids carries the already-rendered tool prompt, and token/logprob
    # metadata is explicitly requested from LMDeploy.
    return {
        "model": args.model_name,
        "session_id": args.session_id if args.session_id is not None else uuid.uuid4().int % 2147483647,
        "messages": [],
        "input_ids": input_ids,
        "tools": tool_schema(),
        "tool_choice": {"type": "function", "function": {"name": "get_order_details"}},
        "parallel_tool_calls": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_completion_tokens": args.max_tokens,
        "stream": stream,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": False,
        "return_token_ids": True,
        "return_routed_experts": True,
        "return_logprob": args.return_logprob,
        "include_stop_str_in_output": True,
    }


def raise_for_status_with_body(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    print(f"[error] response_text={response.text}", file=sys.stderr)
    try:
        print(f"[error] response_json={json.dumps(response.json(), ensure_ascii=False, indent=2)}", file=sys.stderr)
    except Exception:
        pass
    response.raise_for_status()


def shape(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return f"object_ref:{value}"
    dims = []
    cur = value
    while isinstance(cur, list):
        dims.append(len(cur))
        cur = cur[0] if cur else None
    return "x".join(map(str, dims)) if dims else type(value).__name__


def short_list(values: list[Any] | None, limit: int = 64) -> str:
    if values is None:
        return "None"
    if len(values) <= limit:
        return json.dumps(values, ensure_ascii=False)
    head = json.dumps(values[:limit], ensure_ascii=False)
    return f"{head[:-1]}, ...] len={len(values)}"


def print_request(prefix: str, payload: dict[str, Any]) -> None:
    input_ids = payload["input_ids"]
    print(f"\n[{prefix}] session_id={payload['session_id']}")
    print(f"[{prefix}] input_ids_len={len(input_ids)}")
    print(f"[{prefix}] input_ids={short_list(input_ids)}")


def print_choice(prefix: str, choice: dict[str, Any], *, show_full_json: bool) -> None:
    message = choice.get("message") or choice.get("delta") or {}
    output_ids = choice.get("output_ids")
    output_token_logprobs = choice.get("output_token_logprobs")
    routed_experts = choice.get("routed_experts")

    print(f"\n[{prefix}] finish_reason={choice.get('finish_reason')}")
    if message.get("reasoning_content"):
        print(f"[{prefix}] reasoning_content={message['reasoning_content']}")
    if message.get("content"):
        print(f"[{prefix}] content={message['content']}")
    if message.get("tool_calls"):
        print(f"[{prefix}] tool_calls={json.dumps(message['tool_calls'], ensure_ascii=False)}")
    print(f"[{prefix}] output_ids_len={0 if output_ids is None else len(output_ids)}")
    print(f"[{prefix}] output_ids={short_list(output_ids)}")
    print(f"[{prefix}] output_token_logprobs_len={0 if output_token_logprobs is None else len(output_token_logprobs)}")
    print(f"[{prefix}] output_token_logprobs={short_list(output_token_logprobs, limit=16)}")
    print(f"[{prefix}] routed_experts_shape={shape(routed_experts)}")
    if show_full_json:
        print(json.dumps(choice, ensure_ascii=False, indent=2))


def print_merged_stream(
    *,
    content: str,
    reasoning_content: str,
    tool_calls: list[dict[str, Any]],
    output_ids: list[int],
    output_token_logprobs: list[Any],
    final_choice: dict[str, Any] | None,
    show_full_json: bool,
) -> None:
    final_delta = (final_choice or {}).get("delta") or {}
    routed_experts = (final_choice or {}).get("routed_experts")
    finish_reason = (final_choice or {}).get("finish_reason")
    merged_choice = {
        "finish_reason": finish_reason,
        "message": {
            "role": "assistant",
            "content": content,
        },
        "output_ids": output_ids,
        "output_token_logprobs": output_token_logprobs,
        "routed_experts": routed_experts,
    }
    if reasoning_content:
        merged_choice["message"]["reasoning_content"] = reasoning_content
    if tool_calls:
        merged_choice["message"]["tool_calls"] = tool_calls
    # If the final chunk carried visible text such as a stop token, it has
    # already been included in content above. Keep this for debugging parser
    # behavior without changing the merged non-stream-like view.
    if final_delta and show_full_json:
        merged_choice["final_delta"] = final_delta
    print_choice("stream:merged", merged_choice, show_full_json=show_full_json)


def run_non_stream(args: argparse.Namespace, input_ids: list[int]) -> None:
    print("=" * 88)
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    payload = build_payload(args, input_ids, stream=False)
    print_request("non-stream:req", payload)
    response = requests.post(url, headers=headers(args), json=payload, timeout=args.timeout)
    print(f"\n[non-stream] http_status={response.status_code}")
    raise_for_status_with_body(response)
    body = response.json()
    if args.show_full_json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
    for idx, choice in enumerate(body.get("choices", [])):
        print_choice(f"non-stream:{idx}", choice, show_full_json=False)
    print(f"[non-stream] usage={json.dumps(body.get('usage'), ensure_ascii=False)}")
    print("=" * 88)


def _merge_tool_delta(dst: dict[int, dict[str, Any]], tool_delta: dict[str, Any]) -> None:
    index = int(tool_delta.get("index", 0))
    item = dst.setdefault(index, {"type": "function", "function": {"name": "", "arguments": ""}})
    if tool_delta.get("id"):
        item["id"] = tool_delta["id"]
    if tool_delta.get("type"):
        item["type"] = tool_delta["type"]
    fn_delta = tool_delta.get("function") or {}
    fn = item.setdefault("function", {})
    if fn_delta.get("name"):
        fn["name"] = fn.get("name", "") + fn_delta["name"]
    if fn_delta.get("arguments"):
        fn["arguments"] = fn.get("arguments", "") + fn_delta["arguments"]


def run_stream(args: argparse.Namespace, input_ids: list[int]) -> None:
    print("=" * 88)
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    payload = build_payload(args, input_ids, stream=True)
    print_request("stream:req", payload)
    response = requests.post(
        url,
        headers=headers(args),
        json=payload,
        timeout=args.timeout,
        stream=True,
    )
    print(f"\n[stream] http_status={response.status_code}")
    raise_for_status_with_body(response)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    output_ids: list[int] = []
    output_token_logprobs: list[Any] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    final_choice: dict[str, Any] | None = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data: "):
            print(f"[stream] raw={raw_line}")
            continue
        data = raw_line.removeprefix("data: ").strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if args.show_full_json:
            print(json.dumps(chunk, ensure_ascii=False, indent=2))
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("content"):
                content_parts.append(delta["content"])
            output_ids.extend(choice.get("output_ids") or [])
            output_token_logprobs.extend(choice.get("output_token_logprobs") or [])
            for tool_delta in delta.get("tool_calls") or []:
                _merge_tool_delta(tool_calls, tool_delta)
            if choice.get("finish_reason") is not None:
                final_choice = choice

    print(f"[stream] reasoning_content={''.join(reasoning_parts)}")
    print(f"[stream] content={''.join(content_parts)}")
    print(f"[stream] tool_calls={json.dumps([tool_calls[i] for i in sorted(tool_calls)], ensure_ascii=False)}")
    print(f"[stream] output_ids_len={len(output_ids)}")
    print(f"[stream] output_ids={short_list(output_ids)}")
    print(f"[stream] output_token_logprobs_len={len(output_token_logprobs)}")
    print(f"[stream] output_token_logprobs={short_list(output_token_logprobs, limit=16)}")
    merged_tool_calls = [tool_calls[i] for i in sorted(tool_calls)]
    print_merged_stream(
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=merged_tool_calls,
        output_ids=output_ids,
        output_token_logprobs=output_token_logprobs,
        final_choice=final_choice,
        show_full_json=args.show_full_json,
    )
    if final_choice is not None:
        print_choice("stream:final", final_choice, show_full_json=False)
    else:
        print("[stream] no final choice received", file=sys.stderr)
    print("=" * 88)


def main() -> None:
    args = parse_args()
    input_ids = build_input_ids(args)
    if args.case in {"non-stream", "both"}:
        run_non_stream(args, input_ids)
    if args.case in {"stream", "both"}:
        run_stream(args, input_ids)


if __name__ == "__main__":
    main()
