"""Harbor bridge implementation for HarborAgentLoop.

Usage in HarborAgentLoopConfig:
    bridge_import_path="xtuner.v1.rl.agent_loop.harbor_bridge:generate_with_harbor"

This bridge is intentionally self-contained and subprocess-based so it can run
without adding a hard dependency on Harbor's internal Python APIs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from xtuner.v1.data_proto import RolloutState


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _message_to_text(message: list[dict[str, Any]] | str) -> str:
    if isinstance(message, str):
        return message

    lines: list[str] = []
    for turn in message:
        role = str(turn.get("role", "user")).upper()
        content = turn.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            content_text = "\n".join([p for p in text_parts if p])
        else:
            content_text = str(content)
        lines.append(f"[{role}] {content_text}".strip())

    return "\n\n".join(lines).strip()


def _extract_response_from_trajectory(trajectory_path: Path) -> str | None:
    if not trajectory_path.exists():
        return None

    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])

    # Pick the last agent step with textual message.
    for step in reversed(steps):
        if step.get("source") != "agent":
            continue
        msg = step.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg
        if isinstance(msg, list):
            parts: list[str] = []
            for item in msg:
                if isinstance(item, dict):
                    t = item.get("text") or item.get("content") or ""
                    if t:
                        parts.append(str(t))
                elif item:
                    parts.append(str(item))
            text = "\n".join(parts).strip()
            if text:
                return text
    return None


def _find_trial_dir(job_dir: Path) -> Path | None:
    if not job_dir.exists():
        return None
    for p in job_dir.iterdir():
        if p.is_dir():
            return p
    return None


def generate_with_harbor(rollout_state: RolloutState, context: dict[str, Any]) -> dict[str, Any]:
    """Generate one sample by delegating execution to Harbor.

    Expected context keys (all optional unless noted):
        harbor_repo (str): Harbor repo root. Default: /home/huanghaian/.openclaw/workspace/harbor
        harbor_bin (str): Harbor binary. Default: harbor
        bridge_workspace (str): Temp workspace root. Default: /tmp/xtuner_harbor_bridge
        task_template_path (str): Template task path. Default: {harbor_repo}/examples/tasks/hello-world
        job_name_prefix (str): Job name prefix. Default: xtuner-harbor
        env (str): Harbor env provider (docker/daytona/...). Default: docker
        agent (str): Harbor agent name. Default: terminus-2
        model_name (str): Model name passed to Harbor (recommended)
        api_base (str): Agent kwarg api_base
        llm_backend (str): Agent kwarg llm_backend, e.g. litellm
        extra_agent_kwargs (dict): Additional --ak key=value
        timeout_sec (int): Subprocess timeout. Default: 1800
        keep_job_dir (bool): Keep generated task dir. Default: False
    """

    harbor_repo = Path(context.get("harbor_repo", "/home/huanghaian/.openclaw/workspace/harbor")).expanduser()
    harbor_bin = str(context.get("harbor_bin", "harbor"))
    bridge_workspace = Path(context.get("bridge_workspace", "/tmp/xtuner_harbor_bridge")).expanduser()
    task_template = Path(
        context.get("task_template_path", str(harbor_repo / "examples/tasks/hello-world"))
    ).expanduser()

    job_name_prefix = str(context.get("job_name_prefix", "xtuner-harbor"))
    env_name = str(context.get("env", "docker"))
    agent_name = str(context.get("agent", "terminus-2"))
    model_name = context.get("model_name")
    timeout_sec = int(context.get("timeout_sec", 1800))
    keep_job_dir = bool(context.get("keep_job_dir", False))

    llm_backend = context.get("llm_backend")
    api_base = context.get("api_base")
    inference_api_key = context.get("inference_api_key")
    extra_agent_kwargs = dict(context.get("extra_agent_kwargs", {}))

    if not harbor_repo.exists():
        return {"error_msg": f"harbor_repo does not exist: {harbor_repo}"}
    if not task_template.exists():
        return {"error_msg": f"task_template_path does not exist: {task_template}"}

    _ensure_dir(bridge_workspace)

    run_id = f"{job_name_prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    task_dir = bridge_workspace / run_id

    try:
        shutil.copytree(task_template, task_dir)

        # Write instruction from current rollout message.
        instruction_text = _message_to_text(rollout_state.message)
        if not instruction_text:
            instruction_text = "Please solve the task and provide your final answer concisely."

        (task_dir / "instruction.md").write_text(instruction_text + "\n", encoding="utf-8")

        cmd = [
            harbor_bin,
            "run",
            "--path",
            str(task_dir),
            "--agent",
            agent_name,
            "--env",
            env_name,
            "--n-concurrent",
            "1",
            "--job-name",
            run_id,
        ]

        if model_name:
            cmd.extend(["--model", str(model_name)])

        # Agent kwargs (--ak key=value)
        if llm_backend:
            cmd.extend(["--ak", f"llm_backend={llm_backend}"])
        if api_base:
            cmd.extend(["--ak", f"api_base={api_base}"])
        for k, v in extra_agent_kwargs.items():
            cmd.extend(["--ak", f"{k}={v}"])

        env = None
        if inference_api_key:
            env = dict(os.environ)
            # Generic key for bridge consumers.
            env["INFERENCE_API_KEY"] = str(inference_api_key)
            # Common OpenAI-compatible gateway env.
            env.setdefault("OPENAI_API_KEY", str(inference_api_key))

        proc = subprocess.run(
            cmd,
            cwd=str(harbor_repo),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=env,
        )

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-4000:]
            stdout_tail = (proc.stdout or "")[-2000:]
            return {
                "error_msg": f"harbor run failed (code={proc.returncode})\nSTDERR:\n{stderr_tail}\nSTDOUT:\n{stdout_tail}",
                "finish_reason": "error",
            }

        job_dir = harbor_repo / "jobs" / run_id
        trial_dir = _find_trial_dir(job_dir)
        if trial_dir is None:
            return {"error_msg": f"No trial dir found for job: {run_id}", "finish_reason": "error"}

        trajectory_path = trial_dir / "agent" / "trajectory.json"
        response = _extract_response_from_trajectory(trajectory_path)

        # Fallback: try result.json for debugging text.
        if not response:
            result_path = trial_dir / "result.json"
            if result_path.exists():
                result_json = json.loads(result_path.read_text(encoding="utf-8"))
                # Keep fallback lightweight.
                response = json.dumps(result_json.get("verifier_result", {}), ensure_ascii=False)

        if not response:
            return {
                "error_msg": "Harbor finished but no assistant response found in trajectory/result.",
                "finish_reason": "error",
            }

        return {
            "response": response,
            "finish_reason": "stop",
        }

    except subprocess.TimeoutExpired:
        return {"error_msg": f"harbor run timeout after {timeout_sec}s", "finish_reason": "error"}
    except Exception as e:
        return {"error_msg": f"harbor bridge exception: {e}", "finish_reason": "error"}
    finally:
        if not keep_job_dir:
            shutil.rmtree(task_dir, ignore_errors=True)
