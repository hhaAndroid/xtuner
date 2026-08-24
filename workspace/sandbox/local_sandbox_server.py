#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import secrets
import selectors
import shutil
import signal
import subprocess
import tarfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_ROOT = Path(os.environ.get("LOCAL_SANDBOX_ROOT", "/tmp/local-sandbox")).resolve()
GUARD_SO = Path(os.environ.get("LOCAL_SANDBOX_GUARD_SO", SCRIPT_DIR / "local_sandbox_guard.so")).resolve()
SANDBOXES: dict[str, dict] = {}
RUNS: dict[tuple[str, str], dict] = {}


def now() -> float:
    return time.time()


def json_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def safe_join(root: Path, user_path: str | None) -> Path:
    if not user_path or user_path == "/":
        return root
    rel = user_path[1:] if user_path.startswith("/") else user_path
    target = (root / rel).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"path escapes workspace: {user_path}")
    return target


def safe_extract_tar(body: bytes, dst: Path) -> None:
    dst = dst.resolve()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as tf:
        for member in tf.getmembers():
            target = (dst / member.name).resolve()
            if target != dst and dst not in target.parents:
                raise ValueError(f"tar member escapes destination: {member.name}")
        tf.extractall(dst)


def make_trace(logs_dir: Path, run_id: str, event: dict) -> None:
    event = {"ts": now(), "run_id": run_id, **event}
    line = json.dumps(event, ensure_ascii=False)
    with (logs_dir / "trace.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sandbox_env(sb: dict, extra_env: dict | None) -> dict:
    workspace = Path(sb["workspace"])
    state_dir = Path(sb["state_dir"])
    home = workspace / ".sandbox_home"
    tmp = workspace / ".sandbox_tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(extra_env or {})
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "LOCAL_SANDBOX_RW_PREFIXES": str(workspace.resolve()),
            "LOCAL_SANDBOX_GUARD_QUIET": env.get("LOCAL_SANDBOX_GUARD_QUIET", "0"),
        }
    )

    preload = str(GUARD_SO)
    if env.get("LD_PRELOAD"):
        preload = preload + ":" + env["LD_PRELOAD"]
    env["LD_PRELOAD"] = preload
    env["LOCAL_SANDBOX_STATE_DIR"] = str(state_dir)
    return env


def run_command(sid: str, run_id: str, cmd: str, timeout: int, extra_env: dict | None) -> None:
    sb = SANDBOXES[sid]
    workspace = Path(sb["workspace"])
    logs_dir = Path(sb["logs_dir"])
    run = RUNS[(sid, run_id)]
    stdout_path = logs_dir / f"{run_id}.stdout.log"
    stderr_path = logs_dir / f"{run_id}.stderr.log"

    wrapped = "\n".join(
        [
            "set -o pipefail",
            "ulimit -n 4096",
            "ulimit -u 512",
            "cd \"$LOCAL_SANDBOX_WORKSPACE\"",
            cmd,
        ]
    )
    env = sandbox_env(sb, extra_env)
    env["LOCAL_SANDBOX_WORKSPACE"] = str(workspace)

    argv = [
        "setpriv",
        "--no-new-privs",
        "timeout",
        str(timeout),
        "bash",
        "-lc",
        wrapped,
    ]

    run.update({"status": "running", "started_at": now(), "stdout": str(stdout_path), "stderr": str(stderr_path)})
    make_trace(logs_dir, run_id, {"type": "start", "cmd": cmd, "workspace": str(workspace)})

    proc = subprocess.Popen(
        argv,
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    run["pid"] = proc.pid

    sel = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

    with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a", encoding="utf-8") as err:
        while sel.get_map():
            for key, _ in sel.select(timeout=0.2):
                line = key.fileobj.readline()
                if line == "":
                    sel.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    out.write(line)
                    out.flush()
                else:
                    err.write(line)
                    err.flush()
                make_trace(logs_dir, run_id, {"type": key.data, "data": line.rstrip("\n")})
            if proc.poll() is not None:
                for key in list(sel.get_map().values()):
                    rest = key.fileobj.read()
                    if rest:
                        target = out if key.data == "stdout" else err
                        target.write(rest)
                        target.flush()
                        for line in rest.splitlines():
                            make_trace(logs_dir, run_id, {"type": key.data, "data": line})
                    sel.unregister(key.fileobj)

    rc = proc.wait()
    run.update({"status": "exited", "exit_code": rc, "finished_at": now()})
    make_trace(logs_dir, run_id, {"type": "exit", "exit_code": rc})


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalSandbox/0.1"

    def send_json(self, code: int, obj: object) -> None:
        data = json_bytes(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        if n == 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_POST(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            if parts == ["sandboxes"]:
                body = self.read_json()
                sid = "sbx-" + secrets.token_hex(8)
                state_dir = SERVER_ROOT / sid
                logs_dir = state_dir / "logs"
                workspace = Path(body["workspace"]).expanduser().resolve() if body.get("workspace") else state_dir / "workspace"
                workspace.mkdir(parents=True, exist_ok=True)
                logs_dir.mkdir(parents=True, exist_ok=True)
                sb = {
                    "id": sid,
                    "workspace": str(workspace),
                    "state_dir": str(state_dir),
                    "logs_dir": str(logs_dir),
                    "created_at": now(),
                }
                SANDBOXES[sid] = sb
                self.send_json(201, sb)
                return

            if len(parts) == 3 and parts[0] == "sandboxes" and parts[2] == "upload":
                sid = parts[1]
                sb = SANDBOXES[sid]
                query = urllib.parse.parse_qs(parsed.query)
                dst = safe_join(Path(sb["workspace"]), query.get("dst", ["."])[0])
                dst.mkdir(parents=True, exist_ok=True)
                n = int(self.headers.get("Content-Length", "0"))
                safe_extract_tar(self.rfile.read(n), dst)
                self.send_json(200, {"ok": True, "dst": str(dst)})
                return

            if len(parts) == 3 and parts[0] == "sandboxes" and parts[2] == "exec":
                sid = parts[1]
                body = self.read_json()
                run_id = "run-" + secrets.token_hex(8)
                RUNS[(sid, run_id)] = {"id": run_id, "sandbox_id": sid, "status": "queued", "created_at": now()}
                t = threading.Thread(
                    target=run_command,
                    args=(sid, run_id, body["cmd"], int(body.get("timeout", 1800)), body.get("env") or {}),
                    daemon=True,
                )
                t.start()
                self.send_json(202, RUNS[(sid, run_id)])
                return

            self.send_json(404, {"error": "not found"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) == 4 and parts[0] == "sandboxes" and parts[2] == "runs":
                key = (parts[1], parts[3])
                run = dict(RUNS[key])
                for field in ("stdout", "stderr"):
                    p = run.get(field)
                    if p and Path(p).exists():
                        data = Path(p).read_text(encoding="utf-8", errors="replace")
                        run[field + "_tail"] = data[-16000:]
                self.send_json(200, run)
                return

            if len(parts) == 3 and parts[0] == "sandboxes" and parts[2] == "trace":
                sb = SANDBOXES[parts[1]]
                trace = Path(sb["logs_dir"]) / "trace.jsonl"
                data = trace.read_bytes() if trace.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if len(parts) == 3 and parts[0] == "sandboxes" and parts[2] == "download":
                sb = SANDBOXES[parts[1]]
                query = urllib.parse.parse_qs(parsed.query)
                src = safe_join(Path(sb["workspace"]), query.get("path", ["."])[0])
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                    tf.add(src, arcname=src.name)
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_json(404, {"error": "not found"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_DELETE(self) -> None:
        try:
            parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]
            if len(parts) == 2 and parts[0] == "sandboxes":
                sid = parts[1]
                for (rsid, _), run in list(RUNS.items()):
                    if rsid == sid and run.get("status") == "running" and run.get("pid"):
                        try:
                            os.killpg(run["pid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                sb = SANDBOXES.pop(sid, None)
                if sb:
                    shutil.rmtree(sb["state_dir"], ignore_errors=True)
                self.send_json(200, {"ok": True})
                return
            self.send_json(404, {"error": "not found"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()
    SERVER_ROOT.mkdir(parents=True, exist_ok=True)
    if not GUARD_SO.exists():
        raise SystemExit(f"guard shared object does not exist: {GUARD_SO}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LocalSandboxServer listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
