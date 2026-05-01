import os
import time
import ray
from transformers import AutoTokenizer
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers
import os
from xtuner.v1.rl.gateway import GatewayConfig
import httpx
import requests
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

    only_debug=False
    
    gateway_url='http://10.102.249.52:38100/v1'
    if not only_debug:
        ray.init(num_cpus=80, ignore_reinit_error=True)

        # MODEL_PATH = '/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
        MODEL_PATH = '/mnt/shared-storage-user/llmrazor-share/model/Qwen3.6-35B-A3B'

        resources_cfg = AcceleratorResourcesConfig(
            accelerator='GPU',
            num_workers=8,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,
        )
        max_prompt_length = 8196
        max_response_length = 256*1024
        context_length = max_prompt_length + max_response_length
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg, name="moe_api_pg")
        rollout_config = RolloutConfig(
                env="test_rollout_api_server_dense",
                model_path=MODEL_PATH,
                model_name=os.path.basename(MODEL_PATH).lower(),
                tokenizer_path=MODEL_PATH,
                tensor_parallel_size=2,
                expert_parallel_size=1,
                context_length=context_length,
                worker_log_dir='./worker_logs',
                dist_port_base=38000,
                api_host="127.0.0.1",
                api_port=28000,
                tool_call_parser="qwen3p5",
                reasoning_parser="qwen3",
        )
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
        gateway_config = GatewayConfig(port=38100, auto_start=False, capture_folder='/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/gateway/worker_logs')
        gateway_url = ray.get(rollout_controller.start_gateway.remote(gateway_config), timeout=1800)
        wait_for_gateway_ready(gateway_url)
        print(f"Gateway is ready at {gateway_url}")

        url = "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1/models/new"
        payload = {
                "model_name": 'xtuner_gateway_demo',
                "api_key": "sk-admin",
                "api_base": gateway_url,
        }
        headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        print("register model success", resp.json())
    
    check_chat(gateway_url, 'xtuner_gateway_demo')
    check_chat('http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1', 'xtuner_gateway_demo')
    check_claude_messages(gateway_url, "xtuner_gateway_demo")
    check_claude_messages(
        "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1",
        "xtuner_gateway_demo",
    )
    if not only_debug:
        import time
        time.sleep(1000000)
