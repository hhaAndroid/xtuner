"""Quick smoke-test for the LLM service defined in internclaw/config.py.

Usage:
    python test_llm_service.py
    RL_LLM_BASE_URL=http://... RL_LLM_MODEL=my-model python test_llm_service.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# BASE_URL = os.environ.get(
#     "RL_LLM_BASE_URL",
#     # "http://0.0.0.0:8000/v1",
#     "http://10.102.249.52:8000/v1",
# ).rstrip("/")

BASE_URL = os.environ.get(
    "RL_LLM_BASE_URL",
    "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
).rstrip("/")

MODEL = os.environ.get(
    "RL_LLM_MODEL",
    "xtuner-qwen35-30b",
)
API_KEY = os.environ.get("RL_LLM_API_KEY", "sk-admin")
TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "60"))

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS, method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def check_models():
    print(f"[1/2] GET {BASE_URL}/models")
    try:
        result = _get(f"{BASE_URL}/models")
        ids = [m.get("id", "") for m in result.get("data", [])]
        print(f"      Available models: {ids}")
        if MODEL in ids:
            print(f"      ✓ Target model '{MODEL}' is listed.")
        else:
            print(f"      ✗ Target model '{MODEL}' NOT found in model list.")
    except Exception as e:
        print(f"      ✗ Failed: {e}")


def check_chat():
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "extra_body": {"spaces_between_special_tokens": False},
    }
    print(f"[2/2] POST {url}")
    t0 = time.time()
    try:
        result = _post(url, payload)
        elapsed = time.time() - t0
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f"      Response ({elapsed:.2f}s): {content!r}")
        print(f"      Usage: {usage}")
        print("      ✓ Chat completions endpoint OK.")
    except Exception as e:
        print(f"      ✗ Failed ({time.time() - t0:.2f}s): {e}")


def main():
    print("=" * 60)
    print(f"  BASE_URL : {BASE_URL}")
    print(f"  MODEL    : {MODEL}")
    print(f"  API_KEY  : {API_KEY[:8]}***")
    print(f"  TIMEOUT  : {TIMEOUT}s")
    print("=" * 60)

    check_models()
    check_chat()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
