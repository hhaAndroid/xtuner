"""Claude Code black-box RL agent loop using sandbox environments.

This agent loop runs Claude Code inside an isolated sandbox (via env-gateway),
pointing it at the XTuner inference gateway as the LLM backend. After Claude Code
finishes, it retrieves all per-turn token IDs (prompt_ids, response_ids, logprobs)
from the gateway's trace store and converts them into trainable RolloutState objects.

If the rollout controller has router-replay enabled, each trace record will also
carry ``routed_experts``, which is forwarded transparently through
``chat_trace_records_to_rollout_states``.

Typical data flow per sample
-----------------------------
1. ``generate_sample`` is called with a ``RolloutState`` containing the task
   instruction and sandbox metadata (image_tag, timeouts, …).
2. A unique ``api_key`` is generated for this execution so that gateway trace
   records can be isolated per run.
3. A sandbox is created via env-gateway, Claude Code is installed (or assumed
   pre-installed) and run inside it with ``ANTHROPIC_BASE_URL`` pointing to the
   XTuner inference gateway and ``ANTHROPIC_API_KEY`` set to the unique key.
4. After Claude Code exits, the verifier (``/tests/test.sh``) is run to obtain
   a scalar reward.
5. The trace records (one per LLM API call / turn) are popped from the gateway
   trace store and converted to ``RolloutState`` objects via
   ``chat_trace_records_to_rollout_states``.  Each state receives the same reward.
6. The list of states (one per turn) is returned to the caller.
"""

from __future__ import annotations

import asyncio
import copy
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from pydantic import ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop, AgentLoopConfig
from xtuner.v1.rl.judger.native import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import chat_trace_records_to_rollout_states

_NVM_SRC = "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh"
_INSTALL_DEPS_CMD = (
    "sh -c 'set -e; "
    "if command -v apk >/dev/null 2>&1; then apk add --no-cache bash curl ca-certificates; "
    "elif command -v apt-get >/dev/null 2>&1; "
    "then apt-get update -qq && apt-get install -y -qq bash curl ca-certificates; "
    "elif command -v dnf >/dev/null 2>&1; then dnf install -y bash curl ca-certificates; "
    "elif command -v yum >/dev/null 2>&1; then yum install -y bash curl ca-certificates; "
    "fi'"
)


class SandboxClaudeCodeAgentLoopConfig(AgentLoopConfig):
    """Configuration for SandboxClaudeCodeAgentLoop.

    Args:
        env_gateway_url (str): URL of the env-gateway service used to create sandboxes.
        max_turns (int): Default maximum Claude Code turns per task.  Can be overridden
            per-sample via ``rollout_state.extra_fields["max_turns"]``.
        agent_timeout (int): Default timeout in seconds for the Claude Code process.
            Can be overridden per-sample via ``rollout_state.extra_fields["agent_timeout"]``.
        verifier_timeout (int): Default timeout in seconds for ``test.sh``.  Can be
            overridden per-sample via ``rollout_state.extra_fields["verifier_timeout"]``.
        sandbox_ttl (int): Default sandbox TTL in seconds.  Can be overridden per-sample.
        install_claude_code (bool): Install nvm + Node.js LTS + claude-code inside the
            sandbox on every run.  Set to ``False`` when the sandbox image already has
            claude-code to skip installation and speed up startup.
        permission_mode (str): Claude Code ``--permission-mode`` flag value.
        max_concurrent_sandboxes (int): Thread-pool size for concurrent sandbox operations.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    env_gateway_url: str = "http://env-gateway.ailab.ailab.ai"
    max_turns: int = 50
    agent_timeout: int = 500
    verifier_timeout: int = 500
    sandbox_ttl: int = 3600
    install_claude_code: bool = True
    permission_mode: str = "bypassPermissions"
    max_concurrent_sandboxes: int = 16

    def build_local(
        self,
        rollout_controller: RolloutController,
        judger: Judger | None = None,
        logger: Any = None,
    ) -> "SandboxClaudeCodeAgentLoop":
        """Build a local SandboxClaudeCodeAgentLoop instance.

        Args:
            rollout_controller (RolloutController): The XTuner rollout controller.
            judger (Judger | None): Optional judger that overrides the verifier reward.
            logger: Logger instance.

        Returns:
            SandboxClaudeCodeAgentLoop: The constructed agent loop.
        """
        return SandboxClaudeCodeAgentLoop(
            env_gateway_url=self.env_gateway_url,
            max_turns=self.max_turns,
            agent_timeout=self.agent_timeout,
            verifier_timeout=self.verifier_timeout,
            sandbox_ttl=self.sandbox_ttl,
            install_claude_code=self.install_claude_code,
            permission_mode=self.permission_mode,
            max_concurrent_sandboxes=self.max_concurrent_sandboxes,
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
        )


class SandboxClaudeCodeAgentLoop(AgentLoop):
    """Claude Code RL agent loop that executes tasks inside isolated sandboxes.

    Each call to ``generate_sample``:

    1. Creates a sandbox via env-gateway using the ``image_tag`` stored in
       ``rollout_state.extra_fields``.
    2. Optionally installs claude-code.
    3. Runs Claude Code with the task instruction, proxying LLM calls through
       the XTuner inference gateway.  A unique ``api_key`` per run is used so
       that the gateway can isolate this run's trace records.
    4. Runs the verifier (``/tests/test.sh``) and parses the reward from
       ``/logs/verifier/reward.txt`` or ``/logs/verifier/ctrf.json``.
    5. Pops the per-turn trace records from the gateway trace store.  Each
       record contains ``prompt_ids``, ``response_ids``, ``logprobs``, and
       optionally ``routed_experts`` (if router-replay is enabled on the
       rollout controller).
    6. Converts records to ``RolloutState`` objects via
       ``chat_trace_records_to_rollout_states`` and assigns the reward.

    Args:
        env_gateway_url (str): URL of the env-gateway service.
        max_turns (int): Default maximum Claude Code turns (fallback when not in extra_fields).
        agent_timeout (int): Default Claude Code execution timeout (seconds).
        verifier_timeout (int): Default verifier timeout (seconds).
        sandbox_ttl (int): Default sandbox TTL (seconds).
        install_claude_code (bool): Whether to install claude-code at startup.
        permission_mode (str): Claude Code permission mode.
        max_concurrent_sandboxes (int): Thread-pool size for sandbox operations.
        rollout_ctl (RolloutController): XTuner rollout controller.
        sample_params (SampleParams): LLM sampling parameters.
        hf_checkpoint (str): HuggingFace model checkpoint path.
        judger (Judger | None): Optional reward judger; overrides verifier reward.
        logger: Logger instance.

    Note:
        Task-specific paths (``task_dir``, ``image_tag``) and timeouts are read from
        ``rollout_state.extra_fields`` at runtime, populated by ``prepare_dataset.py``
        via ``SkillsBenchTokenizeFn``.
    """

    def __init__(
        self,
        env_gateway_url: str,
        max_turns: int,
        agent_timeout: int,
        verifier_timeout: int,
        sandbox_ttl: int,
        install_claude_code: bool,
        permission_mode: str,
        max_concurrent_sandboxes: int,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger: Any = None,
    ) -> None:
        super().__init__(
            rollout_ctl=rollout_ctl,
            sample_params=sample_params,
            hf_checkpoint=hf_checkpoint,
            judger=judger,
            logger=logger,
        )
        self.env_gateway_url = env_gateway_url.rstrip("/")
        self.max_turns = max_turns
        self.agent_timeout = agent_timeout
        self.verifier_timeout = verifier_timeout
        self.sandbox_ttl = sandbox_ttl
        self.install_claude_code = install_claude_code
        self.permission_mode = permission_mode
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_sandboxes)

    async def generate_sample(  # type: ignore[override]
        self, rollout_state: RolloutState, **kwargs
    ) -> list[RolloutState]:
        """Run Claude Code in a sandbox and return per-turn RolloutState objects.

        Args:
            rollout_state (RolloutState): Input state.  The task instruction is
                extracted from ``message``.  Sandbox parameters are read from
                ``extra_fields``: ``image_tag``, ``agent_timeout``,
                ``verifier_timeout``, ``sandbox_ttl``.

        Returns:
            list[RolloutState]: One state per LLM API call (turn).  All
                completed states share the same reward.  On failure, a single
                state with ``Status.FAILED`` is returned.
        """
        try:
            metadata = await self.rollout_ctl.get_rollout_metadata.remote()  # type: ignore[attr-defined]
            gateway_url: str | None = metadata.get("api_server_url")
            rollout_config = metadata.get("rollout_config")
            model_name: str = getattr(rollout_config, "model_name", None) or "rollout-controller"

            if not gateway_url:
                return [
                    self._failed_state(
                        rollout_state,
                        "Gateway is not started. Configure GatewayConfig(auto_start=True) "
                        "before using SandboxClaudeCodeAgentLoop.",
                    )
                ]

            extra = rollout_state.extra_fields
            image_tag: str = extra.get("image_tag", "")
            task_dir: str = extra.get("task_dir", "")
            task_agent_timeout = int(extra.get("agent_timeout", self.agent_timeout))
            task_verifier_timeout = int(extra.get("verifier_timeout", self.verifier_timeout))
            task_sandbox_ttl = int(extra.get("sandbox_ttl", self.sandbox_ttl))
            task_max_turns = int(extra.get("max_turns", self.max_turns))
            instruction = self._extract_instruction(rollout_state)

            if not image_tag:
                return [self._failed_state(rollout_state, "Missing image_tag in extra_fields.")]
            if not task_dir:
                return [self._failed_state(rollout_state, "Missing task_dir in extra_fields.")]
            if not instruction:
                return [self._failed_state(rollout_state, "Could not extract instruction from rollout_state.message.")]

            # Unique key so the gateway can isolate this run's trace records.
            api_key = f"sandbox_{uuid4().hex}"

            loop = asyncio.get_event_loop()
            result: dict[str, Any] = await loop.run_in_executor(
                self._executor,
                self._run_sandbox_task,
                gateway_url,
                model_name,
                api_key,
                image_tag,
                task_dir,
                instruction,
                task_agent_timeout,
                task_verifier_timeout,
                task_sandbox_ttl,
                task_max_turns,
            )

            sandbox_error: str | None = result.get("error")
            if sandbox_error:
                return [self._failed_state(rollout_state, sandbox_error)]

            rollout_extra_fields: dict[str, Any] = {
                "sandbox_api_key": api_key,
                "sandbox_agent_returncode": result.get("agent_returncode"),
                "sandbox_agent_stderr": self._truncate(result.get("agent_stderr", "")),
                "sandbox_verifier_stdout": self._truncate(result.get("verifier_stdout", "")),
                "sandbox_reward_raw": result.get("reward_raw", 0.0),
            }

            # Optionally let a custom judger override the verifier-based reward.
            reward: dict[str, Any] | None = None
            if self.judger is not None:
                judge_state = rollout_state.model_copy(deep=True)
                judge_state.extra_fields = {
                    **copy.deepcopy(rollout_state.extra_fields),
                    **copy.deepcopy(rollout_extra_fields),
                }
                judged_state = await self.judger.judge(judge_state)
                if judged_state.reward is not None:
                    reward = copy.deepcopy(judged_state.reward)

            if reward is None:
                reward_raw = float(result.get("reward_raw", 0.0))
                reward = {"score": reward_raw}

            records = await self._pop_trace_records(gateway_url, api_key)
            if not records:
                return [
                    self._failed_state(
                        rollout_state,
                        f"No gateway trace records found for api_key={api_key}. "
                        f"returncode={result.get('agent_returncode')}, "
                        f"stderr={self._truncate(result.get('agent_stderr', ''), 512)}",
                        extra_fields=rollout_extra_fields,
                    )
                ]

            # chat_trace_records_to_rollout_states extracts prompt_ids, response_ids,
            # logprobs and, when router-replay is enabled on the rollout controller,
            # routed_experts from each record.
            states = chat_trace_records_to_rollout_states(
                rollout_state=rollout_state,
                records=records,
                tokenizer=self.tokenizer,
                extra_fields=rollout_extra_fields,
            )
            if not states:
                return [
                    self._failed_state(
                        rollout_state,
                        "Gateway trace records contained no trainable turns.",
                        extra_fields=rollout_extra_fields,
                    )
                ]

            completed_states = [s for s in states if s.status == Status.COMPLETED]
            for state in completed_states:
                state.reward = copy.deepcopy(reward)

            return states

        except Exception as exc:
            return [self._failed_state(rollout_state, f"SandboxClaudeCodeAgentLoop failed: {exc}")]

    # ------------------------------------------------------------------
    # Synchronous sandbox helpers (run in thread-pool executor)
    # ------------------------------------------------------------------

    def _run_sandbox_task(
        self,
        gateway_url: str,
        model_name: str,
        api_key: str,
        image_tag: str,
        task_dir: str,
        instruction: str,
        agent_timeout: int,
        verifier_timeout: int,
        sandbox_ttl: int,
        max_turns: int,
    ) -> dict[str, Any]:
        """Create a sandbox, run Claude Code, run verifier, return results.

        Args:
            gateway_url (str): XTuner gateway URL (``ANTHROPIC_BASE_URL`` for claude).
            model_name (str): Model name exposed by the gateway.
            api_key (str): Unique key to isolate gateway trace records for this run.
            image_tag (str): Sandbox image tag (e.g. ``hb_offer-letter-generator``).
            task_dir (str): Absolute path to the task directory on the local filesystem.
                Populated by ``prepare_dataset.py`` and forwarded via ``extra_fields``.
            instruction (str): Task instruction passed to Claude Code via ``-p``.
            agent_timeout (int): Claude Code process timeout (seconds).
            verifier_timeout (int): Verifier process timeout (seconds).
            sandbox_ttl (int): Sandbox TTL (seconds).
            max_turns (int): ``--max-turns`` passed to the claude command.

        Returns:
            dict[str, Any]: Keys: ``error``, ``agent_returncode``, ``agent_stderr``,
                ``verifier_stdout``, ``reward_raw``.
        """
        from env_gateway_sdk import EnvClient, GatewayClient

        task_path = Path(task_dir)
        gateway_client = GatewayClient(base_url=self.env_gateway_url)
        env_id: str | None = None
        env_client: EnvClient | None = None

        try:
            env = gateway_client.create(image_tag=image_tag, ttl_seconds=sandbox_ttl)
            if env is None:
                return {"error": f"Failed to create sandbox for image_tag={image_tag!r}"}

            env_id = env.env_id
            env_client = EnvClient(env.url, timeout=float(agent_timeout + 300))
            env_client.wait_ready(timeout=120, interval=10)
            env_client.keepalive(in_secs=30)

            if self.install_claude_code:
                self._install_claude_code_sync(env_client)

            # Upload skills BEFORE running claude so the agent can use them.
            # Layout: {task_dir}/environment/skills/{skill_name}/
            # → sandbox: /root/.claude/skills/{skill_name}/
            skills_dir = task_path / "environment" / "skills"
            if skills_dir.is_dir():
                for skill_dir in sorted(skills_dir.iterdir()):
                    if skill_dir.is_dir():
                        remote_skills_path = f"/root/.claude/skills/{skill_dir.name}"
                        self._upload_dir_recursive(env_client, skill_dir, remote_skills_path)

            # Build and upload the claude runner script to avoid shell-quoting issues.
            env_vars = (
                "IS_SANDBOX=1 "
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 "
                f"ANTHROPIC_BASE_URL={gateway_url} "
                f"ANTHROPIC_API_KEY={api_key} "
                f"ANTHROPIC_AUTH_TOKEN={api_key} "
                f"ANTHROPIC_DEFAULT_SONNET_MODEL={model_name} "
                f"ANTHROPIC_DEFAULT_OPUS_MODEL={model_name} "
                f"ANTHROPIC_DEFAULT_HAIKU_MODEL={model_name} "
            )
            script_content = (
                "#!/bin/bash\n"
                'export NVM_DIR="/root/.nvm"\n'
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"\n'
                f"{env_vars}"
                f"claude --permission-mode={self.permission_mode} "
                f"--output-format=stream-json --max-turns {max_turns} "
                f"-p {shlex.quote(instruction)}\n"
            )
            env_client.upload("/root/run_claude.sh", script_content.encode())
            env_client.exec("chmod +x /root/run_claude.sh")
            agent_result = env_client.exec("bash /root/run_claude.sh", timeout_sec=agent_timeout)
            agent_returncode = agent_result.return_code
            agent_stderr = agent_result.stderr or ""

            # Upload tests and run verifier AFTER claude finishes.
            verifier_stdout = ""
            reward_raw = 0.0
            tests_dir = task_path / "tests"
            if tests_dir.is_dir():
                self._upload_dir_recursive(env_client, tests_dir, "/tests")
                env_client.exec("chmod +x /tests/test.sh")
                env_client.exec("mkdir -p /logs/verifier")
                env_client.exec(
                    "bash /tests/test.sh > /logs/verifier/test_stdout.txt 2>&1",
                    timeout_sec=verifier_timeout,
                )
                dl = env_client.download_file("/logs/verifier/test_stdout.txt")
                verifier_stdout = dl.content.decode("utf-8", errors="replace") if dl.ok else ""
                reward_raw = self._parse_reward_sync(env_client)

            return {
                "error": None,
                "agent_returncode": agent_returncode,
                "agent_stderr": agent_stderr,
                "verifier_stdout": verifier_stdout,
                "reward_raw": reward_raw,
            }

        except Exception as exc:
            return {"error": f"Sandbox task failed: {exc}"}

        finally:
            if env_client is not None:
                try:
                    env_client.close()
                except Exception:
                    pass
            if env_id is not None:
                try:
                    gateway_client.close(env_id)
                except Exception:
                    pass

    def _install_claude_code_sync(self, env_client: Any) -> None:
        env_client.exec(_INSTALL_DEPS_CMD, timeout_sec=60)
        env_client.exec(
            f"bash -c 'export NVM_DIR=\"/root/.nvm\" && curl -fsSL {_NVM_SRC} | bash'",
            timeout_sec=300,
        )
        env_client.exec(
            "bash -c 'export NVM_DIR=\"/root/.nvm\" && . \"$NVM_DIR/nvm.sh\" && nvm install --lts'",
            timeout_sec=300,
        )
        env_client.exec(
            "bash -c 'export NVM_DIR=\"/root/.nvm\" && . \"$NVM_DIR/nvm.sh\" && "
            "npm install -g @anthropic-ai/claude-code && claude --version'",
            timeout_sec=300,
        )

    @staticmethod
    def _upload_dir_recursive(env_client: Any, local_dir: Path, remote_dir: str) -> None:
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

    @staticmethod
    def _parse_reward_sync(env_client: Any) -> float:
        result = env_client.download_file("/logs/verifier/reward.txt")
        if result.ok:
            try:
                return float(result.content.decode("utf-8").strip())
            except (ValueError, AttributeError):
                pass

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

    # ------------------------------------------------------------------
    # Async gateway helpers
    # ------------------------------------------------------------------

    async def _pop_trace_records(self, gateway_url: str, api_key: str) -> list[dict[str, Any]]:
        """Pop per-turn trace records from the XTuner gateway for this run.

        Each record contains ``prompt_ids``, ``response_ids``, ``logprobs`` and,
        when router-replay is active, ``routed_experts``.

        Args:
            gateway_url (str): XTuner gateway base URL.
            api_key (str): The unique key that was passed to Claude Code for this run.

        Returns:
            list[dict[str, Any]]: Ordered list of trace records (one per LLM call).
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{gateway_url.rstrip('/')}/trace_store/pop",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        records = payload.get("records", [])
        return records if isinstance(records, list) else []

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _extract_instruction(self, rollout_state: RolloutState) -> str:
        for message in reversed(rollout_state.message):
            if message.get("role") == "user":
                return self._content_to_text(message.get("content"))
        return ""

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(item.get("text", "")) if isinstance(item, dict) and "text" in item else str(item)
                for item in content
            ]
            return "\n".join(p for p in parts if p)
        return str(content)

    def _failed_state(
        self,
        rollout_state: RolloutState,
        error_msg: str,
        *,
        extra_fields: dict[str, Any] | None = None,
    ) -> RolloutState:
        failed = rollout_state.model_copy(deep=True)
        failed.status = Status.FAILED
        failed.error_msg = error_msg
        if extra_fields:
            failed.extra_fields = {
                **copy.deepcopy(rollout_state.extra_fields),
                **copy.deepcopy(extra_fields),
            }
        return failed

    def _truncate(self, text: str, max_chars: int = 4096) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...<truncated>"
