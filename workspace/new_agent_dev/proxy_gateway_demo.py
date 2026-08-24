import os
import time
import os
import httpx
import json
import urllib.request

os.environ['XTUNER_USE_LMDEPLOY'] = '1'
os.environ["LMD_SKIP_WARMUP"] = "1"
os.environ["XTUNER_USE_FA3"] = "1"


def wait_for_gateway_ready(base_url: str, *, timeout_seconds: float = 180.0) -> None:
    """Block until a gateway server responds successfully on ``/livez``."""
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/livez", timeout=5.0)
            if response.status_code == 200:
                return
            last_error = response.text
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(1.0)
    raise AssertionError(f"Gateway did not become ready at {base_url}: {last_error}")


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


def check_chat(base_url: str, model: str):
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        url = f"{normalized_base_url}/chat/completions"
    else:
        url = f"{normalized_base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "session_id": "test-session-001",
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "extra_body": {"spaces_between_special_tokens": False},
    }
    print(f"========================POST {url}================================")
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


def check_claude_messages(base_url: str, model: str):
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        url = f"{normalized_base_url}/messages"
    else:
        url = f"{normalized_base_url}/v1/messages"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Reply with exactly: pong"}],
            }
        ],
        "max_tokens": 64,
    }
    print(f"========================POST {url}================================")
    t0 = time.time()
    try:
        result = _post(url, payload)
        elapsed = time.time() - t0
        content_blocks = result.get("content", [])
        text_content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        usage = result.get("usage", {})
        print(f"      Response ({elapsed:.2f}s): {text_content!r}")
        print(f"      Usage: {usage}")
        print("      ✓ Claude v1/messages endpoint OK.")
    except Exception as e:
        print(f"      ✗ Failed ({time.time() - t0:.2f}s): {e}")


if __name__ == "__main__":

    gateway_url='http://10.102.250.69:35003/v1'
    # check_chat(gateway_url, 'xtuner_qwen3p5_vl_35b')
    check_chat('http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1', 'xtuner_qwen3p5_vl_35b')
    # check_claude_messages(gateway_url, "xtuner_qwen3p5_vl_35b")
    # check_claude_messages(
    #     "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
    #     "xtuner_qwen3p5_vl_35b",
    # )
