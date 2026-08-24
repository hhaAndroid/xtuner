#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def http_get(url, headers=None, timeout=10):
    req = urllib.request.Request(url, method="GET")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "ignore")


def http_post(url, payload, headers=None, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "ignore")


def pretty_print(title, status_code, body):
    print(f"\n=== {title} ===")
    print(f"status: {status_code}")
    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception:
        print(body)


def build_openai_headers(api_key):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_anthropic_headers(api_key, anthropic_version):
    headers = {"anthropic-version": anthropic_version}
    if api_key:
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def main():
    parser = argparse.ArgumentParser(description="Verify XTuner rollout gateway with OpenAI and Anthropic HTTP APIs.")
    parser.add_argument("--base-url", default=os.environ.get("GATEWAY_BASE_URL", "http://127.0.0.1:28000"))
    parser.add_argument("--model", default=os.environ.get("GATEWAY_MODEL"))
    parser.add_argument("--api-key", default=os.environ.get("GATEWAY_API_KEY"))
    parser.add_argument("--prompt", default="请只回复: gateway-ok")
    parser.add_argument("--protocol", choices=["all", "openai", "anthropic"], default="all")
    parser.add_argument("--anthropic-version", default=os.environ.get("ANTHROPIC_VERSION", "2023-06-01"))
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    metadata_status, metadata_body = http_get(f"{base_url}/metadata")
    health_status, health_body = http_get(f"{base_url}/healthz")

    pretty_print("GET /healthz", health_status, health_body)
    pretty_print("GET /metadata", metadata_status, metadata_body)

    model = args.model
    if not model and metadata_status == 200:
        try:
            metadata = json.loads(metadata_body)
            model = metadata["rollout_config"]["model_name"]
        except Exception:
            model = None

    if not model:
        print("\nerror: no model provided and failed to infer one from /metadata", file=sys.stderr)
        sys.exit(2)

    failures = 0

    if args.protocol in ("all", "openai"):
        openai_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": args.prompt},
            ],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        }
        status, body = http_post(
            f"{base_url}/v1/chat/completions",
            openai_payload,
            headers=build_openai_headers(args.api_key),
        )
        pretty_print("POST /v1/chat/completions", status, body)
        if status != 200:
            failures += 1

    if args.protocol in ("all", "anthropic"):
        anthropic_payload = {
            "model": model,
            "system": "You are a concise assistant.",
            "messages": [
                {"role": "user", "content": args.prompt},
            ],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        }
        status, body = http_post(
            f"{base_url}/v1/messages",
            anthropic_payload,
            headers=build_anthropic_headers(args.api_key, args.anthropic_version),
        )
        pretty_print("POST /v1/messages", status, body)
        if status != 200:
            failures += 1

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
