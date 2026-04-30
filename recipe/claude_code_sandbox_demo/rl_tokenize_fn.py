"""Tokenize function for Claude Code black-box RL training on SkillsBench-style tasks.

Expected JSONL data format::

    {
        "data_source": "skillsbench",
        "prompt": [{"role": "user", "content": "<task instruction>"}],
        "reward_model": {},
        "extra_info": {
            "task_name": "offer-letter-generator",
            "image_tag": "hb_offer-letter-generator",   # optional, defaults to hb_{task_name}
            "agent_timeout": 900,                        # optional
            "verifier_timeout": 900,                     # optional
            "sandbox_ttl": 3600                          # optional
        }
    }
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizer

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.datasets.utils import CachableTokenizeFunction


class SkillsBenchTokenizeFn(CachableTokenizeFunction[RolloutState]):
    """Tokenize function for SkillsBench-style Claude Code RL tasks.

    Converts task data items into RolloutState objects for black-box RL training.
    The prompt is tokenized to get an approximate length for batching/filtering;
    the actual per-turn token IDs used for training are retrieved from the
    XTuner gateway trace store after the Claude Code agent loop completes.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        max_length (int | None): Maximum prompt token length. Items exceeding
            this length are filtered out (num_tokens set to 0 in cache mode).
        default_agent_timeout (int): Default timeout for Claude Code execution (seconds).
        default_verifier_timeout (int): Default timeout for the verifier (seconds).
        default_sandbox_ttl (int): Default sandbox time-to-live (seconds).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
        default_agent_timeout: int = 900,
        default_verifier_timeout: int = 900,
        default_sandbox_ttl: int = 3600,
        default_max_turns: int = 50,
    ):
        super().__init__(tokenizer)
        self.max_length = max_length
        self.default_agent_timeout = default_agent_timeout
        self.default_verifier_timeout = default_verifier_timeout
        self.default_sandbox_ttl = default_sandbox_ttl
        self.default_max_turns = default_max_turns

    def __call__(self, item: dict, **kwargs) -> RolloutState:
        """Convert a SkillsBench data item to a RolloutState.

        Args:
            item (dict): Data item. See module docstring for the expected format.
                ``extra_info`` must contain ``task_dir`` (absolute path to the task
                directory written by ``prepare_dataset.py``) which is forwarded to
                ``extra_fields`` for use by ``SandboxClaudeCodeAgentLoop``.

        Returns:
            RolloutState: The rollout state carrying the task instruction and
                sandbox metadata needed by SandboxClaudeCodeAgentLoop.
        """
        message = item["prompt"]
        extra_info: dict = dict(item.get("extra_info") or {})
        task_name = extra_info.get("task_name", "")

        raw_prompt = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
        )
        extra_info["raw_prompt"] = raw_prompt
        data = self.tokenizer(raw_prompt, add_special_tokens=False)
        prompt_token_ids: list[int] = data["input_ids"]
        num_tokens = len(prompt_token_ids)

        if self.state == "cache":
            if self.max_length is not None and num_tokens > self.max_length:
                num_tokens = 0  # filtered out by the dataset
        else:
            if self.max_length is not None:
                assert num_tokens <= self.max_length, f"num_tokens {num_tokens} > max_length {self.max_length}"

        # Inject sandbox-specific fields that SandboxClaudeCodeAgentLoop reads at runtime.
        extra_info.setdefault("image_tag", f"hb_{task_name}" if task_name else "")
        extra_info.setdefault("agent_timeout", self.default_agent_timeout)
        extra_info.setdefault("verifier_timeout", self.default_verifier_timeout)
        extra_info.setdefault("sandbox_ttl", self.default_sandbox_ttl)
        extra_info.setdefault("max_turns", self.default_max_turns)

        return RolloutState(
            prompt_ids=prompt_token_ids,
            message=message,
            task_name=task_name,
            reward_model=item.get("reward_model") or {},
            num_tokens=num_tokens,
            proxy_attn_flops=float(num_tokens),
            data_source=item.get("data_source"),
            extra_fields=extra_info,
        )

    def hash(self) -> str:
        return "SkillsBenchTokenizeFn"


class SkillsBenchTokenizeFnConfig(BaseModel):
    """Pydantic configuration for SkillsBenchTokenizeFn.

    Args:
        max_length (int | None): Maximum prompt token length.
        default_agent_timeout (int): Default Claude Code timeout in seconds.
        default_verifier_timeout (int): Default verifier timeout in seconds.
        default_sandbox_ttl (int): Default sandbox TTL in seconds.
    """

    model_config = ConfigDict(title="SkillsBench RL tokenize function config", extra="forbid")

    max_length: int | None = None
    default_agent_timeout: int = 900
    default_verifier_timeout: int = 900
    default_sandbox_ttl: int = 3600
    default_max_turns: int = 50

    def build(self, tokenizer: PreTrainedTokenizer, **_kwargs: Any) -> SkillsBenchTokenizeFn:
        """Build the tokenize function.

        Args:
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance.

        Returns:
            SkillsBenchTokenizeFn: The constructed tokenize function.
        """
        return SkillsBenchTokenizeFn(
            tokenizer=tokenizer,
            max_length=self.max_length,
            default_agent_timeout=self.default_agent_timeout,
            default_verifier_timeout=self.default_verifier_timeout,
            default_sandbox_ttl=self.default_sandbox_ttl,
            default_max_turns=self.default_max_turns,
        )
