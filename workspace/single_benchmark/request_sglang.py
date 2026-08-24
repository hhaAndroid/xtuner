#!/usr/bin/env python3
"""Send non-streaming token-id SGLang chat requests and print ids/metadata."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import uuid
from typing import Any

import numpy as np
import requests
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/"
    "models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://10.102.249.71:23333")
    parser.add_argument("--model-name", default="hha_xtuner_qwen35_35b")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--case", choices=["non-stream", "stream", "both"], default="non-stream")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--prompt-mode", choices=["tool", "long"], default="tool")
    parser.add_argument("--return-token-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-logprob", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-routed-experts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routed-experts-num-layers", type=int, default=None)
    parser.add_argument("--routed-experts-top-k", type=int, default=None)
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


def source_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt_mode == "long":
        return [
            {
                "role": "user",
                "content": (
                    "Write a long, detailed technical note about debugging streamed "
                    "LLM inference services. Continue until you naturally reach the "
                    "token limit. Do not call tools."
                ),
            }
        ]

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
    tools = tool_schema() if args.prompt_mode == "tool" else None
    prompt = tokenizer.apply_chat_template(
        source_messages(args),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def load_routed_experts_shape(args: argparse.Namespace) -> tuple[int | None, int | None]:
    if args.routed_experts_num_layers and args.routed_experts_top_k:
        return args.routed_experts_num_layers, args.routed_experts_top_k

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    num_layers = args.routed_experts_num_layers or getattr(
        text_config, "num_hidden_layers", None
    )
    top_k = args.routed_experts_top_k or getattr(
        text_config, "num_experts_per_tok", None
    )
    return num_layers, top_k


def build_payload(args: argparse.Namespace, input_ids: list[int], stream: bool) -> dict[str, Any]:
    # SGLang's OpenAI chat TITO path still requires non-empty messages.
    # input_ids carries the already-rendered prompt and bypasses template
    # tokenization; metadata is returned through choices[*].meta_info.
    payload = {
        "model": args.model_name,
        "session_id": args.session_id if args.session_id is not None else str(uuid.uuid4().int % 2147483647),
        "messages": source_messages(args),
        "input_ids": input_ids,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_completion_tokens": args.max_tokens,
        "stream": stream,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": False,
        "return_prompt_token_ids": args.return_token_ids,
        "return_meta_info": True,
        "return_routed_experts": args.return_routed_experts,
        "routed_experts_start_len": len(input_ids) - 1,
        "logprobs": args.return_logprob,
        "include_stop_str_in_output": True,
    }
    if args.prompt_mode == "tool":
        payload.update(
            {
                "tools": tool_schema(),
                "tool_choice": {"type": "function", "function": {"name": "get_order_details"}},
                "parallel_tool_calls": False,
            }
        )
    return payload


def chat_choice_to_choice(choice: dict[str, Any]) -> dict[str, Any]:
    meta_info = choice.get("meta_info") or {}
    output_token_logprobs = meta_info.get("output_token_logprobs") or []
    output_ids = [
        item[1]
        for item in output_token_logprobs
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    return {
        "finish_reason": choice.get("finish_reason"),
        "message": choice.get("message") or {},
        "delta": choice.get("delta") or {},
        "output_ids": output_ids,
        "output_token_logprobs": output_token_logprobs,
        "routed_experts": meta_info.get("routed_experts"),
    }


def build_generate_payload(
    args: argparse.Namespace, input_ids: list[int], stream: bool
) -> dict[str, Any]:
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_tokens,
    }
    return {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "stream": stream,
        "return_logprob": args.return_logprob,
        "return_routed_experts": args.return_routed_experts,
        "logprob_start_len": len(input_ids) - 1,
        "session_id": args.session_id
        if args.session_id is not None
        else str(uuid.uuid4().int % 2147483647),
    }


def generate_body_to_choice(body: dict[str, Any]) -> dict[str, Any]:
    meta_info = body.get("meta_info") or {}
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    return {
        "finish_reason": finish_reason,
        "message": {
            "role": "assistant",
            "content": body.get("text") or "",
        },
        "delta": {
            "content": body.get("text") or "",
        },
        "output_ids": body.get("output_ids") or [],
        "output_token_logprobs": meta_info.get("output_token_logprobs") or [],
        "routed_experts": meta_info.get("routed_experts"),
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


def _list_shape(value: Any) -> tuple[int, ...]:
    dims = []
    cur = value
    while isinstance(cur, list):
        dims.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(dims)


def decode_routed_experts(
    routed_experts: Any,
    *,
    input_ids_len: int,
    output_ids_len: int,
    num_layers: int | None,
    top_k: int | None,
) -> tuple[Any, str, str]:
    if routed_experts is None:
        return None, "None", "None"

    if isinstance(routed_experts, list):
        list_shape = "x".join(map(str, _list_shape(routed_experts))) or "list"
        return routed_experts, "list", list_shape

    if not isinstance(routed_experts, str):
        value_shape = shape(routed_experts)
        return routed_experts, type(routed_experts).__name__, value_shape

    try:
        flat = np.frombuffer(base64.b64decode(routed_experts.encode("utf-8")), dtype=np.int32)
    except Exception as exc:
        value_shape = f"object_ref:{routed_experts}"
        return routed_experts, f"str_decode_error:{exc}", value_shape

    if flat.size == 0:
        return [], "base64/int32", "0"

    if num_layers and top_k:
        per_token = num_layers * top_k
        if flat.size % per_token == 0:
            raw = flat.reshape(flat.size // per_token, num_layers, top_k)
            return (
                raw.tolist(),
                "base64/int32",
                "x".join(map(str, raw.shape)),
            )

    candidate_token_lens = []
    if input_ids_len > 0 and output_ids_len >= 0:
        candidate_token_lens.append(input_ids_len + output_ids_len - 1)
    candidate_token_lens.extend([input_ids_len + output_ids_len, input_ids_len - 1])
    if output_ids_len > 0:
        candidate_token_lens.append(output_ids_len)

    seen = set()
    for token_len in candidate_token_lens:
        if token_len <= 0 or token_len in seen:
            continue
        seen.add(token_len)
        if flat.size % token_len != 0:
            continue
        layer_topk = flat.size // token_len
        for topk in (8, 6, 5, 4, 3, 2, 1):
            if layer_topk % topk == 0:
                arr = flat.reshape(token_len, layer_topk // topk, topk)
                return (
                    arr.tolist(),
                    "base64/int32",
                    "x".join(map(str, arr.shape)),
                )

    return flat.tolist(), "base64/int32_flat", f"{flat.size}"


def short_list(values: list[Any] | None, limit: int = 64) -> str:
    if values is None:
        return "None"
    if len(values) <= limit:
        return json.dumps(values, ensure_ascii=False)
    head = json.dumps(values[:limit], ensure_ascii=False)
    return f"{head[:-1]}, ...] len={len(values)}"


def short_lengths(values: list[int], limit: int = 20) -> str:
    if not values:
        return "[]"
    if len(values) <= limit:
        return json.dumps(values)
    half = max(1, limit // 2)
    return f"{json.dumps(values[:half])[:-1]}, ..., {json.dumps(values[-half:])[1:]} len={len(values)}"


def classify_stream_lengths(lengths: list[int]) -> str:
    nonzero = [value for value in lengths if value > 0]
    if not nonzero:
        return "none"
    if len(nonzero) == 1:
        return "single"

    max_len = max(nonzero)
    nondecreasing = sum(
        1 for prev, cur in zip(nonzero, nonzero[1:]) if cur >= prev
    ) / (len(nonzero) - 1)
    increasing = sum(
        1 for prev, cur in zip(nonzero, nonzero[1:]) if cur > prev
    ) / (len(nonzero) - 1)
    small_chunks = sum(1 for value in nonzero if value <= 8) / len(nonzero)

    if max_len >= 32 and nondecreasing >= 0.8 and increasing >= 0.3:
        return "likely_cumulative"
    if small_chunks >= 0.8:
        return "likely_delta"
    return "mixed_or_batched_delta"


def merge_stream_values(merged: list[Any], values: list[Any] | None) -> tuple[str, int]:
    if not values:
        return "empty", 0
    if merged and len(values) > len(merged) and values[: len(merged)] == merged:
        suffix = values[len(merged) :]
        merged.extend(suffix)
        return "cumulative", len(suffix)
    merged.extend(values)
    return "delta", len(values)


def print_stream_field_diag(prefix: str, lengths: list[int], modes: dict[str, int]) -> None:
    nonzero = [value for value in lengths if value > 0]
    total_len = sum(lengths)
    max_len = max(nonzero) if nonzero else 0
    avg_len = total_len / len(nonzero) if nonzero else 0.0
    print(
        f"[stream:diag] {prefix} classification={classify_stream_lengths(lengths)} "
        f"events={len(lengths)} nonzero_events={len(nonzero)} max_chunk_len={max_len} "
        f"sum_chunk_lens={total_len} avg_nonzero_len={avg_len:.2f} merge_modes={modes}"
    )
    print(f"[stream:diag] {prefix} chunk_lens={short_lengths(lengths)}")


def choice_payload_summary(choice: dict[str, Any]) -> dict[str, Any]:
    delta = choice.get("delta") or {}
    content = delta.get("content")
    reasoning_content = delta.get("reasoning_content")
    tool_calls = delta.get("tool_calls") or []
    output_ids = choice.get("output_ids") or []
    output_logprobs = choice.get("output_token_logprobs") or []
    routed_experts = choice.get("routed_experts")
    return {
        "finish_reason": choice.get("finish_reason"),
        "content_len": len(content) if isinstance(content, str) else 0,
        "reasoning_content_len": len(reasoning_content) if isinstance(reasoning_content, str) else 0,
        "tool_calls_len": len(tool_calls),
        "output_ids_len": len(output_ids),
        "output_token_logprobs_len": len(output_logprobs),
        "routed_experts_type": type(routed_experts).__name__ if routed_experts is not None else None,
        "routed_experts_len": len(routed_experts) if isinstance(routed_experts, str) else 0,
    }


def print_request(prefix: str, payload: dict[str, Any]) -> None:
    input_ids = payload["input_ids"]
    print(f"\n[{prefix}] session_id={payload['session_id']}")
    print(f"[{prefix}] input_ids_len={len(input_ids)}")
    print(f"[{prefix}] input_ids={short_list(input_ids)}")


def print_choice(
    prefix: str,
    choice: dict[str, Any],
    *,
    input_ids_len: int,
    num_layers: int | None,
    top_k: int | None,
    show_full_json: bool,
) -> None:
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
    (
        decoded_routed_experts,
        routed_experts_format,
        routed_experts_shape,
    ) = decode_routed_experts(
        routed_experts,
        input_ids_len=input_ids_len,
        output_ids_len=0 if output_ids is None else len(output_ids),
        num_layers=num_layers,
        top_k=top_k,
    )
    print(f"[{prefix}] routed_experts_format={routed_experts_format}")
    print(f"[{prefix}] routed_experts_shape={routed_experts_shape}")
    if show_full_json and decoded_routed_experts is not routed_experts:
        print(f"[{prefix}] decoded_routed_experts={short_list(decoded_routed_experts, limit=2)}")
    if show_full_json:
        print(json.dumps(choice, ensure_ascii=False, indent=2))


def print_merged_stream(
    *,
    content: str,
    reasoning_content: str,
    tool_calls: list[dict[str, Any]],
    output_ids: list[int],
    output_token_logprobs: list[Any],
    input_ids_len: int,
    num_layers: int | None,
    top_k: int | None,
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
    print_choice(
        "stream:merged",
        merged_choice,
        input_ids_len=input_ids_len,
        num_layers=num_layers,
        top_k=top_k,
        show_full_json=show_full_json,
    )


def run_non_stream(
    args: argparse.Namespace,
    input_ids: list[int],
    num_layers: int | None,
    top_k: int | None,
) -> None:
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
    choices = body.get("choices") or []
    for idx, choice in enumerate(choices):
        print_choice(
            f"non-stream:{idx}",
            chat_choice_to_choice(choice),
            input_ids_len=len(input_ids),
            num_layers=num_layers,
            top_k=top_k,
            show_full_json=args.show_full_json,
        )
    meta_info = dict((choices[0].get("meta_info") if choices else None) or {})
    if isinstance(meta_info.get("routed_experts"), str):
        meta_info["routed_experts"] = f"<base64 len={len(meta_info['routed_experts'])}>"
    print(f"[non-stream] meta_info={json.dumps(meta_info, ensure_ascii=False)}")
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


def run_stream(
    args: argparse.Namespace,
    input_ids: list[int],
    num_layers: int | None,
    top_k: int | None,
) -> None:
    print("[stream] skipped: /v1/chat/completions TITO metadata is only tested in non-stream mode.")
    return
    print("=" * 88)
    url = f"{args.base_url.rstrip('/')}/generate"
    payload = build_generate_payload(args, input_ids, stream=True)
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
    data_event_count = 0
    stream_payload_bytes = 0
    output_ids_chunk_lens: list[int] = []
    output_logprobs_chunk_lens: list[int] = []
    output_ids_merge_modes = {"empty": 0, "delta": 0, "cumulative": 0}
    output_logprobs_merge_modes = {"empty": 0, "delta": 0, "cumulative": 0}
    final_payload_bytes = 0
    nonfinal_payload_bytes = 0
    routed_experts_event_count = 0
    max_event_bytes = 0
    max_event_index = 0
    max_event_summary: list[dict[str, Any]] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data: "):
            print(f"[stream] raw={raw_line}")
            continue
        data = raw_line.removeprefix("data: ").strip()
        if data == "[DONE]":
            break
        data_event_count += 1
        event_bytes = len(data.encode("utf-8"))
        stream_payload_bytes += event_bytes
        chunk = json.loads(data)
        event_is_final = False
        event_has_routed_experts = False
        event_summary = []
        if args.show_full_json:
            print(json.dumps(chunk, ensure_ascii=False, indent=2))
        choice = generate_body_to_choice(chunk)
        event_summary.append(choice_payload_summary(choice))
        if choice.get("finish_reason") is not None:
            event_is_final = True
            final_choice = choice
        if choice.get("routed_experts") is not None:
            event_has_routed_experts = True
        content_parts[:] = [chunk.get("text") or ""]
        choice_output_ids = choice.get("output_ids") or []
        choice_output_logprobs = choice.get("output_token_logprobs") or []
        output_ids_chunk_lens.append(len(choice_output_ids))
        output_logprobs_chunk_lens.append(len(choice_output_logprobs))
        ids_mode, _ = merge_stream_values(output_ids, choice_output_ids)
        logprobs_mode, _ = merge_stream_values(
            output_token_logprobs,
            choice_output_logprobs,
        )
        output_ids_merge_modes[ids_mode] += 1
        output_logprobs_merge_modes[logprobs_mode] += 1
        if event_is_final:
            final_payload_bytes += event_bytes
        else:
            nonfinal_payload_bytes += event_bytes
        if event_has_routed_experts:
            routed_experts_event_count += 1
        if event_bytes > max_event_bytes:
            max_event_bytes = event_bytes
            max_event_index = data_event_count
            max_event_summary = event_summary

    print(f"[stream:diag] data_events={data_event_count} payload_bytes={stream_payload_bytes}")
    print(
        f"[stream:diag] payload_bytes_nonfinal={nonfinal_payload_bytes} "
        f"payload_bytes_final={final_payload_bytes} routed_experts_events={routed_experts_event_count} "
        f"max_event_index={max_event_index} max_event_bytes={max_event_bytes} "
        f"max_event_summary={json.dumps(max_event_summary, ensure_ascii=False)}"
    )
    print_stream_field_diag("output_ids", output_ids_chunk_lens, output_ids_merge_modes)
    print_stream_field_diag(
        "output_token_logprobs",
        output_logprobs_chunk_lens,
        output_logprobs_merge_modes,
    )
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
        input_ids_len=len(input_ids),
        num_layers=num_layers,
        top_k=top_k,
        final_choice=final_choice,
        show_full_json=args.show_full_json,
    )
    if final_choice is not None:
        print_choice(
            "stream:final",
            final_choice,
            input_ids_len=len(input_ids),
            num_layers=num_layers,
            top_k=top_k,
            show_full_json=args.show_full_json,
        )
    else:
        print("[stream] no final choice received", file=sys.stderr)
    print("=" * 88)


def main() -> None:
    args = parse_args()
    input_ids = build_input_ids(args)
    num_layers, top_k = load_routed_experts_shape(args)
    print(f"[config] routed_experts_num_layers={num_layers} routed_experts_top_k={top_k}")
    if args.case in {"non-stream", "both"}:
        run_non_stream(args, input_ids, num_layers, top_k)
    if args.case in {"stream", "both"}:
        run_stream(args, input_ids, num_layers, top_k)


if __name__ == "__main__":
    main()
