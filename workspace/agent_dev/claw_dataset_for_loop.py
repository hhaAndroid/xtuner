#!/usr/bin/env python3
"""Minimal for-loop example for JsonlDataset + RLClawTokenizeFn."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402
from xtuner.v1.datasets.jsonl import JsonlDataset  # noqa: E402
from xtuner.v1.datasets.rl_tokenize_fn.claw_tokenize_fn import RLClawTokenizeFn  # noqa: E402


DEFAULT_ANNO_PATH = REPO_ROOT / "workspace/agent_dev/claw_tasks.jsonl"
DEFAULT_TASKS_ROOT = Path("/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/claw-bench/tasks")
DEFAULT_TOKENIZER_PATH = Path("/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-path", type=Path, default=DEFAULT_ANNO_PATH)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--tokenize-workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    tokenize_fn = RLClawTokenizeFn(
        root_path=str(args.tasks_root),
        tokenizer_fn=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    dataset = JsonlDataset(
        anno_path=str(args.anno_path),
        tokenize_fn=tokenize_fn,
        max_length=args.max_length,
    )

    print(f"dataset_len={len(dataset)}")
    print(f"anno_path={args.anno_path}")
    print(f"tasks_root={args.tasks_root}")

    for idx in range(min(args.limit, len(dataset))):
        item = dataset[idx]
        print(f"item: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
