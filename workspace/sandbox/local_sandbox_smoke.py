#!/usr/bin/env python3
import json
import os
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("LOCAL_SANDBOX_BASE_URL", "http://127.0.0.1:18080")
WORKSPACE = Path(os.environ.get("LOCAL_SANDBOX_SMOKE_WORKSPACE", "/tmp/local-sandbox-demo-workspace"))
WORKSPACE.mkdir(parents=True, exist_ok=True)


def post(path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.read()


sb = post("/sandboxes", {"workspace": str(WORKSPACE)})
run = post(
    f"/sandboxes/{sb['id']}/exec",
    {
        "timeout": 30,
        "cmd": "python -c \"from pathlib import Path; Path('inside.txt').write_text('ok'); Path('/tmp/blocked-by-local-sandbox.txt').write_text('bad')\"",
    },
)

while True:
    status = json.loads(get(f"/sandboxes/{sb['id']}/runs/{run['id']}").decode("utf-8"))
    if status["status"] == "exited":
        break
    time.sleep(0.2)

trace = get(f"/sandboxes/{sb['id']}/trace").decode("utf-8", errors="replace")
print(json.dumps({"sandbox": sb, "run": status, "inside_exists": (WORKSPACE / "inside.txt").exists()}, indent=2))
print("--- trace ---")
print(trace)
