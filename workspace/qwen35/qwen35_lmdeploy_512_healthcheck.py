#!/usr/bin/env python3
"""Launch one LMDeploy server, send 512 DAPO math prompts, and monitor health."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/"
    "models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)
DEFAULT_LMDEPLOY_PATH = "/mnt/shared-storage-user/huanghaian/code/lmdeploy"
DEFAULT_DATA_PATH = "/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "workspace/qwen35/qwen35_lmdeploy_512_results.jsonl"
DEFAULT_LOG_PATH = REPO_ROOT / "workspace/qwen35/qwen35_lmdeploy_server.log"
LOG_WRITE_LOCK = threading.Lock()


class HealthCheckFailure(RuntimeError):
    """Raised when the independent /health probe fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lmdeploy-path", default=DEFAULT_LMDEPLOY_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--server-log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--tee-server-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--client-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23333)
    parser.add_argument("--model-name", default="qwen3_5_moe")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--distributed-executor-backend", choices=["uni", "mp", "ray"], default="ray")
    parser.add_argument("--session-len", type=int, default=10240)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--cache-max-entry-count", type=float, default=0.8)
    parser.add_argument("--max-prefill-token-num", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--return-logprob", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-logprobs", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=3600.0)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--health-interval", type=float, default=10.0)
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=16)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--progress-mininterval", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    parser.add_argument("--logprobs-mode", default="raw_logprobs", choices=["none", "raw_logits", "raw_logprobs"])
    parser.add_argument("--uvicorn-log-level", default="warning")
    parser.add_argument("--kill-server-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-return-routed-experts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/usr/local/nvidia/bin/:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:" + env.get("LD_LIBRARY_PATH", "")
    )
    pythonpath_parts = [args.lmdeploy_path, str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    env["ENABLE_RETURN_ROUTED_EXPERTS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["UVICORN_LOG_LEVEL"] = args.uvicorn_log_level
    if args.use_fa3:
        env["XTUNER_USE_FA3"] = "1"
    return env


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    distributed_executor_backend = args.distributed_executor_backend
    if distributed_executor_backend is None and args.dp > 1:
        distributed_executor_backend = "ray"

    cmd = [
        sys.executable,
        "-m",
        "lmdeploy",
        "serve",
        "api_server",
        args.model_path,
        "--backend",
        "pytorch",
        "--model-name",
        args.model_name,
        "--server-name",
        args.host,
        "--server-port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--tp",
        str(args.tp),
        "--dp",
        str(args.dp),
        "--ep",
        str(args.ep),
        "--session-len",
        str(args.session_len),
        "--max-batch-size",
        str(args.max_batch_size),
        "--cache-max-entry-count",
        str(args.cache_max_entry_count),
        "--max-prefill-token-num",
        str(args.max_prefill_token_num),
        "--hf-overrides",
        json.dumps({"fp32_lm_head": True}),
        "--enable-abort-handling",
        "--log-level",
        args.log_level,
    ]
    if args.logprobs_mode != "none":
        cmd.extend(["--logprobs-mode", args.logprobs_mode])
    if distributed_executor_backend is not None:
        cmd.extend(["--distributed-executor-backend", distributed_executor_backend])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enable_return_routed_experts:
        cmd.append("--enable-return-routed-experts")
    if args.api_key:
        cmd.extend(["--api-keys", args.api_key])
    return cmd


def tee_process_output(proc: subprocess.Popen, log_file: Any, echo: bool) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        with LOG_WRITE_LOCK:
            log_file.write(line)
            log_file.flush()
        if echo:
            print(line, end="", flush=True)


def write_parent_log(log_file: Any, message: str, *, echo: bool = True) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - parent - INFO - {message}"
    with LOG_WRITE_LOCK:
        log_file.write(line + "\n")
        log_file.flush()
    if echo:
        print(message, flush=True)


def start_server(args: argparse.Namespace) -> tuple[subprocess.Popen, Any, threading.Thread]:
    log_path = Path(args.server_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    cmd = build_server_cmd(args)
    print("Launching LMDeploy server:")
    print(" ".join(cmd))
    print(f"Server log: {log_path}")
    write_parent_log(log_file, "Launching LMDeploy server:", echo=False)
    write_parent_log(log_file, " ".join(cmd), echo=False)
    write_parent_log(log_file, f"Server log: {log_path}", echo=False)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=build_env(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
        text=True,
    )
    output_thread = threading.Thread(
        target=tee_process_output,
        args=(proc, log_file, args.tee_server_log),
        daemon=True,
    )
    output_thread.start()
    return proc, log_file, output_thread


def stop_server(
    proc: subprocess.Popen,
    log_file: Any,
    output_thread: threading.Thread,
    kill: bool,
    shutdown_reason: str,
) -> None:
    try:
        if kill and proc.poll() is None:
            if shutdown_reason == "health_check_failed":
                write_parent_log(
                    log_file,
                    "HEALTH_CHECK_TRIGGERED_SHUTDOWN: sending SIGTERM to LMDeploy server process group; "
                    "subsequent lmdeploy SIGTERM/cleanup logs are expected cleanup, not the root cause.",
                )
            elif shutdown_reason == "normal_completion":
                write_parent_log(
                    log_file,
                    "NORMAL_COMPLETION_SHUTDOWN: all requests completed and results were written; "
                    "sending SIGTERM to LMDeploy server process group for cleanup.",
                )
            else:
                write_parent_log(
                    log_file,
                    "ERROR_TRIGGERED_SHUTDOWN: script is exiting after an error; "
                    "sending SIGTERM to LMDeploy server process group for cleanup.",
                )
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                write_parent_log(
                    log_file,
                    "SERVER_FORCE_KILL: LMDeploy server did not exit within 30s after SIGTERM; sending SIGKILL.",
                )
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
        output_thread.join(timeout=5)
    finally:
        log_file.close()


def auth_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    return headers


async def wait_until_ready(args: argparse.Namespace, proc: subprocess.Popen, base_url: str) -> None:
    deadline = time.monotonic() + args.startup_timeout
    headers = auth_headers(args)
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        last_log = 0.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"LMDeploy server exited early with code {proc.returncode}")
            try:
                resp = await client.get(f"{base_url}/health", headers=headers)
                if resp.status_code == 200:
                    print("LMDeploy server is healthy.")
                    return
            except httpx.HTTPError:
                pass
            now = time.monotonic()
            if now - last_log >= 15:
                print(f"Waiting for LMDeploy server at {base_url}/health ...")
                last_log = now
            await asyncio.sleep(1.0)
    raise TimeoutError(f"LMDeploy server was not healthy within {args.startup_timeout} seconds")


def read_first_n_items(path: str, n: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise KeyError(f"line {line_no} has no 'prompt' field")
            obj["_line_no"] = line_no
            items.append(obj)
            if len(items) >= n:
                break
    if len(items) < n:
        raise ValueError(f"Only found {len(items)} examples in {path}; need {n}")
    return items


def build_payload(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": args.model_name,
        "messages": item["prompt"],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "min_new_tokens": args.min_new_tokens,
        "stream": False,
        "skip_special_tokens": True,
        "return_token_ids": True,
        "return_routed_experts": args.enable_return_routed_experts,
    }
    if args.return_logprob:
        payload["logprobs"] = True
        payload["top_logprobs"] = args.top_logprobs
    return payload


async def health_monitor(
    args: argparse.Namespace,
    base_url: str,
    stop_event: asyncio.Event,
    failure: dict[str, BaseException],
    progress: dict[str, int | float],
    log_file: Any | None = None,
) -> None:
    headers = auth_headers(args)
    timeout = httpx.Timeout(args.health_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.health_interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                resp = await client.get(f"{base_url}/health", headers=headers)
                if resp.status_code != 200:
                    raise RuntimeError(f"health returned HTTP {resp.status_code}: {resp.text[:200]}")
                tqdm.tqdm.write("health ok")
                if log_file is not None:
                    write_parent_log(
                        log_file,
                        f"HEALTH_CHECK_OK: url={base_url}/health, timeout={args.health_timeout}s",
                        echo=False,
                    )
            except BaseException as exc:
                total = int(progress.get("total", 0))
                submitted = int(progress.get("submitted", 0))
                completed = int(progress.get("completed", 0))
                failed = int(progress.get("failed", 0))
                pending = max(0, total - completed - failed)
                elapsed = time.monotonic() - float(progress.get("started_at", time.monotonic()))
                failed_message = (
                    "HEALTH_CHECK_FAILED: "
                    f"url={base_url}/health, timeout={args.health_timeout}s, "
                    f"submitted={submitted}/{total}, completed={completed}, failed={failed}, "
                    f"pending={pending}, elapsed={elapsed:.1f}s, error={type(exc).__name__}: {exc}"
                )
                cancel_message = (
                    "HEALTH_CHECK_FAILED: cancelling pending inference requests; "
                    "LMDeploy server will be terminated by cleanup because health check is the root cause."
                )
                tqdm.tqdm.write(failed_message)
                tqdm.tqdm.write(cancel_message)
                if log_file is not None:
                    write_parent_log(log_file, failed_message, echo=False)
                    write_parent_log(log_file, cancel_message, echo=False)
                failure["error"] = HealthCheckFailure(
                    f"LMDeploy health check failed or timed out after {args.health_timeout}s"
                )
                failure["error"].__cause__ = exc
                stop_event.set()
                return


async def client_progress_monitor(
    args: argparse.Namespace,
    progress: dict[str, int | float],
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=args.progress_interval)
            return
        except asyncio.TimeoutError:
            pass
        total = int(progress["total"])
        submitted = int(progress["submitted"])
        completed = int(progress["completed"])
        failed = int(progress["failed"])
        pending = max(0, total - completed - failed)
        elapsed = time.monotonic() - float(progress["started_at"])
        tqdm.tqdm.write(
            f"client progress: submitted={submitted}/{total}, "
            f"completed={completed}, failed={failed}, pending={pending}, elapsed={elapsed:.1f}s"
        )


async def send_one(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    base_url: str,
    idx: int,
    item: dict[str, Any],
    progress: dict[str, int | float],
) -> dict[str, Any]:
    started = time.monotonic()
    payload = build_payload(args, item)
    progress["submitted"] = int(progress["submitted"]) + 1
    resp = await client.post(f"{base_url}/v1/chat/completions", headers=auth_headers(args), json=payload)
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "request_index": idx,
        "source_line": item["_line_no"],
        "extra_info": item.get("extra_info"),
        "ground_truth": item.get("reward_model", {}).get("ground_truth"),
        "elapsed_seconds": elapsed,
        "http_status": resp.status_code,
    }
    try:
        body = resp.json()
    except json.JSONDecodeError:
        result["error"] = resp.text
        resp.raise_for_status()
        return result

    if resp.status_code != 200:
        result["error"] = body
        resp.raise_for_status()

    result["response"] = body
    try:
        choice = body["choices"][0]
        result["text"] = choice["message"].get("content", "")
        result["finish_reason"] = choice.get("finish_reason")
        result["output_ids"] = choice.get("output_ids")
        result["routed_experts"] = choice.get("routed_experts")
        result["logprobs"] = choice.get("logprobs")
        if choice.get("logprobs") and isinstance(choice["logprobs"], dict):
            content_logprobs = choice["logprobs"].get("content") or []
            result["token_logprobs"] = [item.get("logprob") for item in content_logprobs]
            result["token_logprobs_len"] = len(content_logprobs)
        result["usage"] = body.get("usage")
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return result


async def run_requests(args: argparse.Namespace, base_url: str, log_file: Any | None = None) -> None:
    items = read_first_n_items(args.data_path, args.num_requests)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timeout = None if args.request_timeout is None else httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(max_connections=args.num_requests + 16, max_keepalive_connections=args.num_requests + 16)
    stop_event = asyncio.Event()
    progress: dict[str, int | float] = {
        "total": len(items),
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "started_at": time.monotonic(),
    }
    health_failure: dict[str, BaseException] = {}
    health_task = asyncio.create_task(
        health_monitor(args, base_url, stop_event, health_failure, progress, log_file)
    )
    progress_task = asyncio.create_task(client_progress_monitor(args, progress, stop_event))
    completed = 0
    started = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [
                asyncio.create_task(send_one(client, args, base_url, idx, item, progress))
                for idx, item in enumerate(items)
            ]

            def cancel_requests_on_health_failure(_: asyncio.Task) -> None:
                if health_failure:
                    for pending in tasks:
                        pending.cancel()

            health_task.add_done_callback(cancel_requests_on_health_failure)
            print(f"Submitted {len(tasks)} requests.")
            try:
                with output_path.open("w", encoding="utf-8") as fout:
                    progress_bar = tqdm.tqdm(
                        total=len(tasks),
                        desc="LMDeploy 512 requests",
                        unit="sample",
                        dynamic_ncols=True,
                        mininterval=args.progress_mininterval,
                        leave=True,
                    )
                    try:
                        for task in asyncio.as_completed(tasks):
                            if health_failure:
                                for pending in tasks:
                                    pending.cancel()
                                raise health_failure["error"]
                            try:
                                result = await task
                            except asyncio.CancelledError:
                                if health_failure:
                                    raise health_failure["error"]
                                raise
                            except BaseException:
                                progress["failed"] = int(progress["failed"]) + 1
                                progress_bar.update(1)
                                raise
                            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                            fout.flush()
                            completed += 1
                            progress["completed"] = completed
                            progress_bar.update(1)
                            if completed == 1 or completed % args.print_every == 0 or completed == len(tasks):
                                elapsed = time.monotonic() - started
                                tqdm.tqdm.write(f"completed {completed}/{len(tasks)} in {elapsed:.1f}s")
                    finally:
                        progress_bar.close()
            except BaseException:
                for pending in tasks:
                    pending.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    finally:
        stop_event.set()
        await health_task
        await progress_task

    if health_failure:
        raise health_failure["error"]
    print(f"Wrote results to {output_path}")
    if log_file is not None:
        write_parent_log(log_file, f"Wrote results to {output_path}", echo=False)


async def async_main(
    args: argparse.Namespace,
    proc: subprocess.Popen | None,
    log_file: Any | None = None,
) -> None:
    base_url = f"http://{args.client_host}:{args.port}"
    if proc is not None:
        await wait_until_ready(args, proc, base_url)
    await run_requests(args, base_url, log_file)


def main() -> int:
    args = parse_args()
    if args.port == 0:
        args.port = find_free_port()

    if args.dry_run:
        print(" ".join(build_server_cmd(args)))
        return 0

    proc, log_file, output_thread = start_server(args)
    shutdown_reason = "normal_completion"
    try:
        asyncio.run(async_main(args, proc, log_file))
        return 0
    except HealthCheckFailure:
        shutdown_reason = "health_check_failed"
        raise
    except BaseException:
        shutdown_reason = "error"
        raise
    finally:
        stop_server(proc, log_file, output_thread, args.kill_server_on_exit, shutdown_reason)


if __name__ == "__main__":
    raise SystemExit(main())
