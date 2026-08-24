# Copyright (c) OpenMMLab. All rights reserved.
"""Convert Orca RLHF preference data to XTuner v1 DPO JSONL format.

The target format is consumed by ``Qwen3VLDPOTokenizeFunction`` via
``VLMPreferenceJsonlDataset``:

    {
        "prompt": [{"role": "system", ...}, {"role": "user", ...}],
        "chosen": [{"role": "assistant", ...}],
        "rejected": [{"role": "assistant", ...}]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/mnt/shared-storage-user/llmrazor-share/data/orca_rlhf.jsonl")
DEFAULT_OUTPUT = Path("orca_rlhf_xtuner.jsonl")
REQUIRED_KEYS = ("system", "question", "chosen", "rejected")


def _as_text(value: Any, key: str, line_no: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"line {line_no}: expected {key!r} to be str, got {type(value).__name__}")
    return value.strip()


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def convert_record(record: dict[str, Any], line_no: int) -> dict[str, list[dict[str, Any]]]:
    missing_keys = [key for key in REQUIRED_KEYS if key not in record]
    if missing_keys:
        raise KeyError(f"line {line_no}: missing required keys: {', '.join(missing_keys)}")

    system = _as_text(record["system"], "system", line_no)
    question = _as_text(record["question"], "question", line_no)
    chosen = _as_text(record["chosen"], "chosen", line_no)
    rejected = _as_text(record["rejected"], "rejected", line_no)

    prompt = []
    if system:
        prompt.append({"role": "system", "content": text_content(system)})
    prompt.append({"role": "user", "content": text_content(question)})

    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": text_content(chosen)}],
        "rejected": [{"role": "assistant", "content": text_content(rejected)}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Orca RLHF JSONL to XTuner v1 preference JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input Orca RLHF JSONL path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output XTuner JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"line {line_no}: expected JSON object")
            converted = convert_record(record, line_no)
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} records to {args.output}")


if __name__ == "__main__":
    main()
