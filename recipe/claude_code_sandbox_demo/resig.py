import requests

gateway_url= 'http://10.103.23.59:30001'
model_name='xtuner_vllm_qwen3.6-35b-a3b'

url = "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1/models/new"
payload = {
        "model_name": model_name,
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