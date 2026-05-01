"""SkillsBench evaluation script using XTuner Gateway + env-gateway sandboxes.

Evaluates a custom model served via XTuner Gateway on the SkillsBench benchmark,
using env-gateway sandboxes as the execution environment instead of local Docker.
"""

from __future__ import annotations
import time 
import argparse
import asyncio
import json
import logging
import shlex
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import tomllib

from env_gateway_sdk import EnvClient, GatewayClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("skillsbench_eval")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    gateway_url: str
    model_name: str | None  # None = let claude code pick default
    api_key: str | None     # ANTHROPIC_API_KEY (for official API)
    auth_token: str | None  # ANTHROPIC_AUTH_TOKEN (for proxy/gateway)
    tasks_dir: Path
    task_names: list[str]
    max_concurrent: int
    max_turns: int
    agent_timeout: int
    sandbox_ttl: int
    output_file: Path
    work_dir: Path | None  # If set, per-task artifacts are dumped here under {task_name}/
    env_gateway_url: str


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


@dataclass
class TaskInfo:
    name: str
    task_dir: Path
    instruction: str
    verifier_timeout: int
    agent_timeout: int


def _load_tasks(config: EvalConfig) -> list[TaskInfo]:
    tasks_dir = config.tasks_dir
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks-dir not found: {tasks_dir}")

    tasks: list[TaskInfo] = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "task.toml").exists():
            continue
        if not (task_dir / "instruction.md").exists():
            continue
        if not (task_dir / "tests" / "test.sh").exists():
            continue

        name = task_dir.name
        if config.task_names and name not in config.task_names:
            continue

        # Read instruction, stripping Harbor canary marker lines
        raw_instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
        instruction_lines = [
            line for line in raw_instruction.splitlines() if not line.startswith("HARBOR_CANARY:")
        ]
        instruction = "\n".join(instruction_lines).strip()

        # Parse task.toml for timeouts
        with open(task_dir / "task.toml", "rb") as f:
            toml_data = tomllib.load(f)

        verifier_timeout = int(toml_data.get("verifier", {}).get("timeout_sec", 500))
        task_agent_timeout = int(toml_data.get("agent", {}).get("timeout_sec", config.agent_timeout))

        tasks.append(
            TaskInfo(
                name=name,
                task_dir=task_dir,
                instruction=instruction,
                verifier_timeout=verifier_timeout,
                agent_timeout=task_agent_timeout,
            )
        )

    return tasks


# ---------------------------------------------------------------------------
# Task result
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_name: str
    api_key: str
    reward: float
    error: str | None
    agent_stdout: str
    agent_stderr: str
    agent_returncode: int | None
    verifier_stdout: str
    elapsed_sec: float
    started_at: str


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def _exec_checked(env_client: EnvClient, command: str, *, action: str, timeout_sec: int | None = None):
    if timeout_sec is None:
        result = env_client.exec(command)
    else:
        result = env_client.exec(command, timeout_sec=timeout_sec)
    return_code = getattr(result, "return_code", 0)
    if return_code not in (0, None):
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        details = [f"{action} failed with return_code={return_code}"]
        if stderr:
            details.append(f"stderr={stderr[:1000]}")
        if stdout:
            details.append(f"stdout={stdout[:1000]}")
        raise RuntimeError(" | ".join(details))
    return result


def _upload_dir_recursive(env_client: EnvClient, local_dir: Path, remote_dir: str) -> None:
    """Upload all files from local_dir to remote_dir in the sandbox."""
    env_client.exec(f"mkdir -p {remote_dir}")
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir)
        remote_path = f"{remote_dir}/{relative}"
        remote_parent = str(Path(remote_path).parent)
        if remote_parent != remote_dir:
            env_client.exec(f"mkdir -p {remote_parent}")
        env_client.upload(remote_path, file_path.read_bytes())


def _install_claude_code(env_client: EnvClient, timeout_sec: int = 300) -> None:
    """Install nvm + Node.js LTS + claude-code as root.

    Sets NVM_DIR=/root/.nvm so the installation is owned by root and accessible
    when claude runs as root.  Harbor signals the sandbox context via IS_SANDBOX=1
    which allows --permission-mode=bypassPermissions even for root.
    """
    nvm_src = "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh"

    # Step 1: Install system deps (curl + bash needed by nvm installer)
    _exec_checked(
        env_client,
        "sh -c 'set -e; "
        "if command -v apk >/dev/null 2>&1; then apk add --no-cache bash curl ca-certificates; "
        "elif command -v apt-get >/dev/null 2>&1; then apt-get update -qq && apt-get install -y -qq bash curl ca-certificates; "
        "elif command -v dnf >/dev/null 2>&1; then dnf install -y bash curl ca-certificates; "
        "elif command -v yum >/dev/null 2>&1; then yum install -y bash curl ca-certificates; "
        "fi'",
        action="install system dependencies",
        timeout_sec=60,
    )

    # Step 2: Install nvm under /root/.nvm
    _exec_checked(
        env_client,
        f"bash -c 'export NVM_DIR=\"/root/.nvm\" && curl -fsSL {nvm_src} | bash'",
        action="install nvm",
        timeout_sec=timeout_sec,
    )

    # Step 3: Install Node.js LTS
    _exec_checked(
        env_client,
        "bash -c 'export NVM_DIR=\"/root/.nvm\" && . \"$NVM_DIR/nvm.sh\" && nvm install --lts'",
        action="install Node.js LTS",
        timeout_sec=timeout_sec,
    )

    # Step 4: Install @anthropic-ai/claude-code and verify
    _exec_checked(
        env_client,
        "bash -c 'export NVM_DIR=\"/root/.nvm\" && . \"$NVM_DIR/nvm.sh\" && "
        "npm install -g @anthropic-ai/claude-code && claude --version'",
        action="install claude-code",
        timeout_sec=timeout_sec,
    )


_SOLUTION_EXCLUDE_PREFIXES = (
    "/root/.nvm",
    "/root/.npm",
    "/root/.claude",
    "/root/.config",
    "/root/.cache",
    "/root/.local",
    "/root/run_claude.sh",
)


def _download_agent_solution(env_client: EnvClient, task_name: str, solution_dir: Path) -> None:
    """Download files created by the agent in /root/ to solution_dir.

    Excludes known setup artifacts (nvm, npm, claude-code installation, etc.) so
    only the files the agent actually produced as part of solving the task are saved.
    """
    result = env_client.exec("find /root -maxdepth 4 -type f 2>/dev/null")
    if not result.stdout:
        logger.debug("[%s] No files found under /root", task_name)
        return

    remote_paths = [p.strip() for p in result.stdout.splitlines() if p.strip()]
    for remote_path in remote_paths:
        if any(remote_path.startswith(prefix) for prefix in _SOLUTION_EXCLUDE_PREFIXES):
            continue
        # Preserve directory structure relative to /root/
        relative = remote_path.removeprefix("/root/").lstrip("/")
        # Skip hidden files/dirs directly under /root/ (e.g. .bashrc, .profile, .goose)
        if relative.split("/")[0].startswith("."):
            continue
        try:
            dl = env_client.download_file(remote_path)
            if not (dl.ok and dl.content):
                continue
            local_path = solution_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(dl.content)
            logger.debug("[%s] Downloaded solution file: %s", task_name, remote_path)
        except Exception as exc:
            logger.debug("[%s] Could not download solution file %s: %s", task_name, remote_path, exc)


def _dump_task_artifacts(
    env_client: EnvClient,
    task_name: str,
    work_dir: Path,
    agent_stdout: str,
    agent_stderr: str,
) -> None:
    """Download all task artifacts from the sandbox and save them under work_dir/task_name/.

    Layout:
        {work_dir}/{task_name}/
            agent_trajectory.jsonl   # stream-json lines from claude --output-format=stream-json
            agent_stderr.txt         # claude's stderr
            solution/                # files created by the agent under /root/
            verifier/
                reward.txt           # binary pass/fail score written by test.sh
                ctrf.json            # CTRF test report (if present)
                test_stdout.txt      # combined stdout+stderr of test.sh
    """
    task_dir = work_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # 1. Agent trajectory (stream-json output from claude)
    if agent_stdout:
        (task_dir / "agent_trajectory.jsonl").write_text(agent_stdout, encoding="utf-8")

    # 2. Agent stderr
    if agent_stderr:
        (task_dir / "agent_stderr.txt").write_text(agent_stderr, encoding="utf-8")

    # 3. Agent solution files — files produced by the agent in /root/
    solution_dir = task_dir / "solution"
    solution_dir.mkdir(exist_ok=True)
    _download_agent_solution(env_client, task_name, solution_dir)

    # 4. Verifier artifacts — download from sandbox, keep whatever exists
    verifier_dir = task_dir / "verifier"
    verifier_dir.mkdir(exist_ok=True)

    _sandbox_files = [
        ("/logs/verifier/reward.txt", verifier_dir / "reward.txt"),
        ("/logs/verifier/ctrf.json", verifier_dir / "ctrf.json"),
        ("/logs/verifier/test_stdout.txt", verifier_dir / "test_stdout.txt"),
    ]
    for remote_path, local_path in _sandbox_files:
        try:
            result = env_client.download_file(remote_path)
            if result.ok and result.content:
                local_path.write_bytes(result.content)
        except Exception as exc:
            logger.debug("[%s] Could not download %s: %s", task_name, remote_path, exc)

    logger.info("[%s] Artifacts dumped to %s", task_name, task_dir)


def _parse_reward(env_client: EnvClient) -> float:
    """Parse task reward from /logs/verifier/reward.txt or ctrf.json."""
    # Try reward.txt first (binary pass/fail)
    result = env_client.download_file("/logs/verifier/reward.txt")
    if result.ok:
        try:
            return float(result.content.decode("utf-8").strip())
        except (ValueError, AttributeError):
            pass

    # Fall back to ctrf.json (partial credit)
    result = env_client.download_file("/logs/verifier/ctrf.json")
    if result.ok:
        try:
            ctrf = json.loads(result.content)
            tests = ctrf.get("results", {}).get("tests", [])
            if tests:
                passed = sum(1 for t in tests if t.get("status") == "passed")
                return passed / len(tests)
        except (json.JSONDecodeError, AttributeError, ZeroDivisionError):
            pass

    return 0.0


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------


def _evaluate_task(task: TaskInfo, config: EvalConfig) -> TaskResult:
    started_at = datetime.now(tz=timezone.utc).isoformat()
    start_time = time.monotonic()
    logger.info("[%s] Starting evaluation", task.name)

    # Unique key per task run so the gateway can isolate this run's trace records.
    task_api_key = f"sandbox_{uuid4().hex}"
    logger.info("[%s] api_key=%s", task.name, task_api_key)

    gateway_client = GatewayClient(base_url=config.env_gateway_url)
    env_client: EnvClient | None = None
    env_id: str | None = None

    try:
        # a) Create sandbox
        image_tag = f"hb_{task.name}"
        logger.info("[%s] Creating sandbox with image_tag=%s", task.name, image_tag)
        env = gateway_client.create(image_tag=image_tag, ttl_seconds=config.sandbox_ttl)

        if env is None:
            raise RuntimeError(f"Failed to create sandbox for image_tag={image_tag!r} (unknown tag or server error)")

        env_id = env.env_id
        # Set HTTP timeout longer than the agent timeout so exec() doesn't time out first
        env_client = EnvClient(env.url, timeout=float(config.agent_timeout + 300))

        logger.info("[%s] Waiting for sandbox to be ready (env_id=%s)", task.name, env_id)
        env_client.wait_ready(timeout=120, interval=10)
        env_client.keepalive(in_secs=30)

        # b) Install nvm + node + claude-code as root
        logger.info("[%s] Installing claude code", task.name)
        install_start_time = time.time()
        _install_claude_code(env_client)
        end_time = time.time()
        logger.info("[%s] Installed claude code in %s seconds", task.name, end_time - install_start_time)

        # c) Upload skills to /root/.claude/skills/ (claude runs as root)
        skills_dir = task.task_dir / "environment" / "skills"
        if skills_dir.exists():
            for skill_dir in sorted(skills_dir.iterdir()):
                if skill_dir.is_dir():
                    remote_skills_path = f"/root/.claude/skills/{skill_dir.name}"
                    logger.info("[%s] Uploading skill %s -> %s", task.name, skill_dir.name, remote_skills_path)
                    _upload_dir_recursive(env_client, skill_dir, remote_skills_path)

        # d) Write claude command to a script and run as root
        # IS_SANDBOX=1 is the Harbor signal that allows --permission-mode=bypassPermissions
        # even when running as root inside a container.
        logger.info("[%s] Running claude (timeout=%ss)", task.name, task.agent_timeout)
        env_vars = (
            "IS_SANDBOX=1 "
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 "
            f"ANTHROPIC_BASE_URL={config.gateway_url} "
            f"ANTHROPIC_API_KEY={task_api_key} "
            f"ANTHROPIC_AUTH_TOKEN={task_api_key} "
        )
        if config.model_name:
            env_vars += (
                f"ANTHROPIC_DEFAULT_SONNET_MODEL={config.model_name} "
                f"ANTHROPIC_DEFAULT_OPUS_MODEL={config.model_name} "
                f"ANTHROPIC_DEFAULT_HAIKU_MODEL={config.model_name} "
            )
        # Write the command to a script to avoid quoting issues
        script_content = (
            "#!/bin/bash\n"
            "export NVM_DIR=\"/root/.nvm\"\n"
            "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"\n"
            f"{env_vars}"
            f"claude --permission-mode=bypassPermissions --verbose "
            f"--output-format=stream-json --max-turns {config.max_turns} "
            f"-p {shlex.quote(task.instruction)}\n"
        )
        script_path = "/root/run_claude.sh"
        env_client.upload(script_path, script_content.encode())
        env_client.exec(f"chmod +x {script_path}")
        agent_result = env_client.exec(
            f"bash {script_path}",
            timeout_sec=task.agent_timeout,
        )
        agent_stdout = agent_result.stdout or ""
        agent_stderr = agent_result.stderr or ""
        agent_returncode = agent_result.return_code
        logger.info("[%s] Claude finished (returncode=%s)", task.name, agent_returncode)
        if agent_stderr:
            logger.warning("[%s] Claude stderr: %s", task.name, agent_stderr[:500])
        if agent_returncode not in (0, None):
            error = f"Claude exited with return_code={agent_returncode}"
            if agent_stderr.strip():
                error = f"{error}: {agent_stderr.strip()[:500]}"
            if config.work_dir is not None:
                _dump_task_artifacts(env_client, task.name, config.work_dir, agent_stdout, agent_stderr)
            elapsed_sec = time.monotonic() - start_time
            return TaskResult(
                task_name=task.name,
                api_key=task_api_key,
                reward=0.0,
                error=error,
                agent_stdout=agent_stdout,
                agent_stderr=agent_stderr,
                agent_returncode=agent_returncode,
                verifier_stdout="",
                elapsed_sec=elapsed_sec,
                started_at=started_at,
            )

        # e) Upload tests/ to /tests/
        logger.info("[%s] Uploading tests", task.name)
        _upload_dir_recursive(env_client, task.task_dir / "tests", "/tests")
        env_client.exec("chmod +x /tests/test.sh")

        # f) Run verification
        logger.info("[%s] Running verifier (timeout=%ss)", task.name, task.verifier_timeout)
        env_client.exec("mkdir -p /logs/verifier")
        env_client.exec(
            "bash /tests/test.sh > /logs/verifier/test_stdout.txt 2>&1",
            timeout_sec=task.verifier_timeout,
        )

        # Collect verifier stdout from file
        dl = env_client.download_file("/logs/verifier/test_stdout.txt")
        verifier_stdout = dl.content.decode("utf-8", errors="replace") if dl.ok else ""

        # g) Parse reward
        reward = _parse_reward(env_client)
        logger.info("[%s] Reward=%.4f", task.name, reward)

        # h) Dump artifacts to work_dir
        if config.work_dir is not None:
            _dump_task_artifacts(env_client, task.name, config.work_dir, agent_stdout, agent_stderr)

        elapsed_sec = time.monotonic() - start_time
        return TaskResult(
            task_name=task.name,
            api_key=task_api_key,
            reward=reward,
            error=None,
            agent_stdout=agent_stdout,
            agent_stderr=agent_stderr,
            agent_returncode=agent_returncode,
            verifier_stdout=verifier_stdout,
            elapsed_sec=elapsed_sec,
            started_at=started_at,
        )

    except Exception as exc:
        elapsed_sec = time.monotonic() - start_time
        logger.exception("[%s] Evaluation failed: %s", task.name, exc)
        # Still dump whatever was collected before the failure
        if config.work_dir is not None and env_client is not None:
            try:
                _dump_task_artifacts(env_client, task.name, config.work_dir, "", str(exc))
            except Exception:
                pass
        return TaskResult(
            task_name=task.name,
            api_key=task_api_key,
            reward=0.0,
            error=str(exc),
            agent_stdout="",
            agent_stderr="",
            agent_returncode=None,
            verifier_stdout="",
            elapsed_sec=elapsed_sec,
            started_at=started_at,
        )

    finally:
        if env_client is not None:
            try:
                env_client.close()
            except Exception:
                pass
        if env_id is not None:
            try:
                gateway_client.close(env_id)
                logger.info("[%s] Sandbox closed (env_id=%s)", task.name, env_id)
            except Exception as close_exc:
                logger.warning("[%s] Failed to close sandbox: %s", task.name, close_exc)


# ---------------------------------------------------------------------------
# Async orchestration
# ---------------------------------------------------------------------------


async def _run_all_tasks(tasks: list[TaskInfo], config: EvalConfig) -> list[TaskResult]:
    semaphore = asyncio.Semaphore(config.max_concurrent)
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=config.max_concurrent)

    async def _run_with_semaphore(task: TaskInfo) -> TaskResult:
        async with semaphore:
            return await loop.run_in_executor(executor, _evaluate_task, task, config)

    results = await asyncio.gather(
        *[_run_with_semaphore(t) for t in tasks],
        return_exceptions=False,
    )
    executor.shutdown(wait=False)
    return list(results)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _build_output(
    job_id: str,
    started_at: str,
    config: EvalConfig,
    results: list[TaskResult],
) -> dict:
    finished_at = datetime.now(tz=timezone.utc).isoformat()

    completed = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    pass_count = sum(1 for r in results if r.reward >= 1.0)
    total = len(results)
    mean_reward = sum(r.reward for r in results) / total if total > 0 else 0.0

    return {
        "job_id": job_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": {
            "gateway_url": config.gateway_url,
            "model_name": config.model_name,
            "tasks_dir": str(config.tasks_dir),
            "max_concurrent": config.max_concurrent,
            "max_turns": config.max_turns,
            "agent_timeout": config.agent_timeout,
            "sandbox_ttl": config.sandbox_ttl,
            "env_gateway_url": config.env_gateway_url,
        },
        "results": [asdict(r) for r in results],
        "summary": {
            "total_tasks": total,
            "completed": len(completed),
            "failed": len(failed),
            "pass_count": pass_count,
            "pass_rate": pass_count / total if total > 0 else 0.0,
            "mean_reward": mean_reward,
        },
    }


def _print_summary(output: dict) -> None:
    summary = output["summary"]
    results = output["results"]

    print("\n" + "=" * 60)
    print("SkillsBench Evaluation Results")
    print("=" * 60)
    print(f"  Model       : {output['config']['model_name']}")
    print(f"  Gateway     : {output['config']['gateway_url']}")
    print(f"  Total tasks : {summary['total_tasks']}")
    print(f"  Completed   : {summary['completed']}")
    print(f"  Failed      : {summary['failed']}")
    print(f"  Pass count  : {summary['pass_count']}")
    print(f"  Pass rate   : {summary['pass_rate']:.1%}")
    print(f"  Mean reward : {summary['mean_reward']:.4f}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x["task_name"]):
        status = "PASS" if r["reward"] >= 1.0 else ("FAIL" if r["error"] is None else "ERROR")
        reward_str = f"{r['reward']:.2f}"
        elapsed_str = f"{r['elapsed_sec']:.1f}s"
        error_str = f" | {r['error'][:80]}" if r["error"] else ""
        print(f"  [{status:5s}] {reward_str:4s}  {elapsed_str:7s}  {r['task_name']}{error_str}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on SkillsBench using XTuner Gateway + env-gateway sandboxes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gateway-url", required=True, help="XTuner gateway URL (e.g. http://host/v1)")
    parser.add_argument("--model-name", default=None, help="Model name to force (sets ANTHROPIC_DEFAULT_*_MODEL). Omit to use gateway default.")
    parser.add_argument("--api-key", default=None, help="API key (sets ANTHROPIC_API_KEY). Use for official Anthropic API.")
    parser.add_argument("--auth-token", default=None, help="Auth token (sets ANTHROPIC_AUTH_TOKEN). Use for proxy/gateway.")
    parser.add_argument(
        "--tasks-dir",
        required=True,
        type=Path,
        help="Path to skillsbench/tasks/ directory",
    )
    parser.add_argument(
        "--task-names",
        default="",
        help="Comma-separated task names to run (default: all tasks)",
    )
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent sandboxes")
    parser.add_argument("--max-turns", type=int, default=50, help="Max claude turns per task")
    parser.add_argument("--agent-timeout", type=int, default=1500, help="Timeout for claude execution (seconds)")
    parser.add_argument("--sandbox-ttl", type=int, default=3600, help="Sandbox TTL in seconds")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("skillsbench_results.json"),
        help="JSON results output path",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory to dump per-task artifacts (trajectory, verifier logs). "
             "Each task gets a subdirectory: {work_dir}/{task_name}/",
    )
    parser.add_argument(
        "--env-gateway-url",
        default="http://env-gateway.ailab.ailab.ai",
        help="Sandbox (env-gateway) URL",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    task_names: list[str] = [t.strip() for t in args.task_names.split(",") if t.strip()] if args.task_names else []

    config = EvalConfig(
        gateway_url=args.gateway_url.rstrip("/"),
        model_name=args.model_name,
        api_key=args.api_key,
        auth_token=args.auth_token,
        tasks_dir=args.tasks_dir,
        task_names=task_names,
        max_concurrent=args.max_concurrent,
        max_turns=args.max_turns,
        agent_timeout=args.agent_timeout,
        sandbox_ttl=args.sandbox_ttl,
        output_file=args.output_file,
        work_dir=args.work_dir,
        env_gateway_url=args.env_gateway_url.rstrip("/"),
    )

    logger.info("Discovering tasks in %s", config.tasks_dir)
    tasks = _load_tasks(config)

    if not tasks:
        logger.error("No tasks found matching criteria. Check --tasks-dir and --task-names.")
        return

    logger.info("Found %d task(s): %s", len(tasks), [t.name for t in tasks])

    job_id = str(uuid4())
    started_at = datetime.now(tz=timezone.utc).isoformat()
    logger.info("Job ID: %s | max_concurrent=%d", job_id, config.max_concurrent)

    results = asyncio.run(_run_all_tasks(tasks, config))

    output = _build_output(job_id, started_at, config, results)

    config.output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", config.output_file)

    _print_summary(output)


if __name__ == "__main__":
    main()
