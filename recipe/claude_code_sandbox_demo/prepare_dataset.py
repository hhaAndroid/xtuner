"""Convert raw SkillsBench tasks directory into the JSONL format consumed by
SkillsBenchTokenizeFn for Claude Code black-box RL training.

Each output line is a JSON object:

    {
        "data_source": "skillsbench",
        "prompt": [{"role": "user", "content": "<instruction>"}],
        "reward_model": {},
        "extra_info": {
            "task_name": "offer-letter-generator",
            "image_tag": "hb_offer-letter-generator",
            "agent_timeout": 900,
            "verifier_timeout": 900,
            "sandbox_ttl": 3600
        }
    }

Usage examples
--------------
# All tasks → single output file:
python prepare_dataset.py --tasks-dir /path/to/skillsbench/tasks --output train.jsonl

# 80/20 random train/eval split:
python prepare_dataset.py --tasks-dir /path/to/skillsbench/tasks \\
    --output train.jsonl --eval-output eval.jsonl --eval-ratio 0.2 --seed 42

# Only specific tasks:
python prepare_dataset.py --tasks-dir /path/to/skillsbench/tasks \\
    --output subset.jsonl --task-names citation-check,offer-letter-generator

# Override default timeouts:
python prepare_dataset.py --tasks-dir /path/to/skillsbench/tasks \\
    --output train.jsonl --default-agent-timeout 1200 --default-verifier-timeout 600
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


@dataclass
class TaskEntry:
    task_name: str
    task_dir: str        # absolute path – read by SandboxClaudeCodeAgentLoop at runtime
    image_tag: str
    instruction: str
    agent_timeout: int
    verifier_timeout: int
    sandbox_ttl: int
    max_turns: int


def _load_task(
    task_dir: Path,
    *,
    default_agent_timeout: int,
    default_verifier_timeout: int,
    sandbox_ttl: int,
    default_max_turns: int,
) -> TaskEntry | None:
    """Parse a single SkillsBench task directory.

    Returns None and prints a warning if required files are missing.
    """
    instruction_path = task_dir / "instruction.md"
    toml_path = task_dir / "task.toml"
    test_path = task_dir / "tests" / "test.sh"

    for required in (instruction_path, toml_path, test_path):
        if not required.exists():
            print(f"[SKIP] {task_dir.name}: missing {required.name}", file=sys.stderr)
            return None

    # Read instruction, stripping Harbor canary marker lines.
    raw = instruction_path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if not ln.startswith("HARBOR_CANARY:")]
    instruction = "\n".join(lines).strip()
    if not instruction:
        print(f"[SKIP] {task_dir.name}: empty instruction after filtering", file=sys.stderr)
        return None

    # Parse timeouts from task.toml.
    with open(toml_path, "rb") as f:
        toml_data = tomllib.load(f)
    agent_timeout = int(toml_data.get("agent", {}).get("timeout_sec", default_agent_timeout))
    verifier_timeout = int(toml_data.get("verifier", {}).get("timeout_sec", default_verifier_timeout))
    max_turns = int(toml_data.get("agent", {}).get("max_turns", default_max_turns))

    task_name = task_dir.name
    return TaskEntry(
        task_name=task_name,
        task_dir=str(task_dir.resolve()),
        image_tag=f"hb_{task_name}",
        instruction=instruction,
        agent_timeout=agent_timeout,
        verifier_timeout=verifier_timeout,
        sandbox_ttl=sandbox_ttl,
        max_turns=max_turns,
    )


def load_tasks(
    tasks_dir: Path,
    task_names: list[str],
    *,
    default_agent_timeout: int,
    default_verifier_timeout: int,
    sandbox_ttl: int,
    default_max_turns: int,
) -> list[TaskEntry]:
    """Scan tasks_dir and return valid TaskEntry objects.

    Args:
        tasks_dir (Path): Root directory containing one subdirectory per task.
        task_names (list[str]): If non-empty, only these task names are included.
        default_agent_timeout (int): Fallback agent timeout (seconds).
        default_verifier_timeout (int): Fallback verifier timeout (seconds).
        sandbox_ttl (int): Sandbox TTL (seconds).

    Returns:
        list[TaskEntry]: Loaded tasks sorted by task_name.
    """
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks-dir not found: {tasks_dir}")

    name_filter: set[str] = set(task_names)
    entries: list[TaskEntry] = []

    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if name_filter and task_dir.name not in name_filter:
            continue
        entry = _load_task(
            task_dir,
            default_agent_timeout=default_agent_timeout,
            default_verifier_timeout=default_verifier_timeout,
            sandbox_ttl=sandbox_ttl,
            default_max_turns=default_max_turns,
        )
        if entry is not None:
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# JSONL serialisation
# ---------------------------------------------------------------------------


def task_to_record(entry: TaskEntry, data_source: str = "skillsbench") -> dict:
    """Convert a TaskEntry to the JSONL record expected by SkillsBenchTokenizeFn."""
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": entry.instruction}],
        "reward_model": {},
        "extra_info": {
            "task_name": entry.task_name,
            "task_dir": entry.task_dir,
            "image_tag": entry.image_tag,
            "agent_timeout": entry.agent_timeout,
            "verifier_timeout": entry.verifier_timeout,
            "sandbox_ttl": entry.sandbox_ttl,
            "max_turns": entry.max_turns,
        },
    }



def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    print(f"Wrote {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SkillsBench tasks directory to RL training JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Path to skillsbench/tasks/ directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path (train split, or all tasks if no --eval-output)",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="If set, write eval split to this path and train split to --output",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Fraction of tasks to put in the eval split (used with --eval-output)",
    )
    parser.add_argument(
        "--task-names",
        default="",
        help="Comma-separated task names to include (default: all tasks)",
    )
    parser.add_argument(
        "--data-source",
        default="skillsbench",
        help="Value written to the data_source field of each record",
    )
    parser.add_argument(
        "--default-agent-timeout",
        type=int,
        default=900,
        help="Fallback agent timeout when task.toml does not specify one",
    )
    parser.add_argument(
        "--default-verifier-timeout",
        type=int,
        default=900,
        help="Fallback verifier timeout when task.toml does not specify one",
    )
    parser.add_argument(
        "--sandbox-ttl",
        type=int,
        default=3600,
        help="Sandbox TTL in seconds written to every record",
    )
    parser.add_argument(
        "--default-max-turns",
        type=int,
        default=50,
        help="Fallback max_turns when task.toml does not specify agent.max_turns",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    task_names: list[str] = [t.strip() for t in args.task_names.split(",") if t.strip()]

    print(f"Scanning {args.tasks_dir} …")
    entries = load_tasks(
        args.tasks_dir,
        task_names,
        default_agent_timeout=args.default_agent_timeout,
        default_verifier_timeout=args.default_verifier_timeout,
        sandbox_ttl=args.sandbox_ttl,
        default_max_turns=args.default_max_turns,
    )

    if not entries:
        print("No valid tasks found. Check --tasks-dir and --task-names.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(entries)} valid task(s).")

    records = [task_to_record(e, data_source=args.data_source) for e in entries]

    if args.eval_output is None:
        write_jsonl(records, args.output)
    else:
        rng = random.Random(args.seed)
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_eval = max(1, round(len(shuffled) * args.eval_ratio))
        eval_records = shuffled[:n_eval]
        train_records = shuffled[n_eval:]
        write_jsonl(train_records, args.output)
        write_jsonl(eval_records, args.eval_output)

    # Print a sample record for quick sanity-check.
    print("\nSample record:")
    sample = records[0].copy()
    sample["prompt"][0]["content"] = sample["prompt"][0]["content"][:120] + " …"
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
