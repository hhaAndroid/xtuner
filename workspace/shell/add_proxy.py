import requests
import json
import urllib.error
import urllib.request
import time
import uuid

ROUTED_API_PROXY_ADMIN_URL = "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000"
ROUTED_API_PROXY_URL = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1"


def _normalize_chat_completions_url(base_url: str) -> str:
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        return f"{normalized_base_url}/chat/completions"
    return f"{normalized_base_url}/v1/chat/completions"


def register_to_routedapiproxy(model_name: str, api_server_url: str) -> dict:
    url = f"{ROUTED_API_PROXY_ADMIN_URL}/v1/models/new"
    api_base = api_server_url
    payload = {
        "model_name": model_name,
        "api_key": "sk-admin",
        "api_base": api_base,
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"failed to register to routedapiproxy: HTTP {resp.status_code}: {resp.text[:2000]}")
        raise
    result = resp.json()
    print(f"registered to routedapiproxy: {result}")
    return result


def delete_from_routedapiproxy(model_name: str) -> None:
    url = f"{ROUTED_API_PROXY_ADMIN_URL}/v1/models/delete"
    payload = {
        "model_name": model_name,
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"failed to delete from routedapiproxy: HTTP {resp.status_code}: {resp.text[:2000]}")
        raise
    print(f"deleted from routedapiproxy: {resp.json()}")

 
TIMEOUT = 120
API_KEY = "sk-admin"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    print(f"========================POST {url}================================")
    req = urllib.request.Request(url, data=data, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def check_chat_completions(base_url: str, model: str) -> bool:
    url = _normalize_chat_completions_url(base_url)
    payload = {
        "model": model,
        "session_id": uuid.uuid4().int % 2_147_483_647,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "spaces_between_special_tokens": False,
    }
    # print(f"========================POST {url}================================")
    t0 = time.time()
    try:
        result = _post(url, payload)
        elapsed = time.time() - t0
        choices = result["choices"]
        usage = result.get("usage", {})
        print(f"      Response ({elapsed:.2f}s): {choices!r}")
        print(f"      Usage: {usage}")
        choice = choices[0]
        message = choice.get("message", {})
        content = (message.get("content") or "").strip()
        reasoning_content = (message.get("reasoning_content") or "").strip()
        finish_reason = choice.get("finish_reason")
        if content != "pong" or reasoning_content or finish_reason == "length":
            print(
                "      ✗ Chat completions endpoint returned an invalid sanity-check response: "
                f"content={content!r}, reasoning_content={reasoning_content!r}, "
                f"finish_reason={finish_reason!r}"
            )
            return False
        print("      ✓ Chat completions endpoint OK.")
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"      ✗ Failed ({time.time() - t0:.2f}s): HTTP {e.code} {e.reason}: {body[:2000]}")
        return False
    except Exception as e:
        print(f"      ✗ Failed ({time.time() - t0:.2f}s): {e}")
        return False


def check_chat_completions_with_retry(base_url: str, model_name: str, max_attempts: int = 5, interval: float = 3.0) -> bool:
    for attempt in range(1, max_attempts + 1):
        if check_chat_completions(base_url, model_name):
            return True
        if attempt < max_attempts:
            print(
                f"check chat completions failed for {base_url}, "
                f"retrying {attempt}/{max_attempts - 1} after {interval}s"
            )
            time.sleep(interval)
    return False



if __name__ == "__main__":
    model_name = "hha_xtuner_qwen35_35b"
    api_server_url = "http://10.102.250.69:23333"

    delete_from_routedapiproxy(model_name)
    register_to_routedapiproxy(model_name, api_server_url)
    backend_ok = check_chat_completions_with_retry(api_server_url, model_name)
    routed_ok = check_chat_completions_with_retry(ROUTED_API_PROXY_URL, model_name)
    if not backend_ok:
        raise SystemExit(f"backend chat completions check failed for {api_server_url}")
    if not routed_ok:
        raise SystemExit(f"routed proxy chat completions check failed for {ROUTED_API_PROXY_URL}")
