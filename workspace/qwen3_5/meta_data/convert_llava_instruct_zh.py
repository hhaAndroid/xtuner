#!/usr/bin/env python3
"""Convert llava_instruct OpenAI-format jsonl: image_url -> image, strip <IMG_CONTEXT>\\n from text.

Reads input path from llava.json (annotation field) by default; writes
llava_instruct_150k_zh_wh_new.jsonl next to this script.

Usage:
  python convert_llava_instruct_zh.py
  python convert_llava_instruct_zh.py -i /path/to/source.jsonl -o /path/to/out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os


def _strip_img_context_prefix(text: str) -> str:
    if not text:
        return text
    for prefix in ("<IMG_CONTEXT>\n", "<IMG_CONTEXT>\r\n"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text


def _convert_content_part(part: dict) -> None:
    if not isinstance(part, dict):
        return
    if part.get("type") == "image_url" and "image_url" in part:
        part["type"] = "image"
        part["image"] = part.pop("image_url")
    if part.get("type") == "text" and "text" in part and isinstance(part["text"], str):
        part["text"] = _strip_img_context_prefix(part["text"])


def convert_record(obj: dict) -> dict:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return obj
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _strip_img_context_prefix(content)
        elif isinstance(content, list):
            for part in content:
                _convert_content_part(part)
    return obj


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_meta = os.path.join(here, "llava.json")
    default_out = os.path.join(here, "llava_instruct_150k_zh_wh_new.jsonl")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta",
        default=default_meta,
        help="llava.json path (used to read annotation if -i omitted)",
    )
    parser.add_argument("-i", "--input", default=None, help="Source jsonl (overrides llava.json)")
    parser.add_argument("-o", "--output", default=default_out, help="Output jsonl path")
    args = parser.parse_args()

    in_path = args.input
    if in_path is None:
        with open(args.meta, encoding="utf-8") as f:
            meta = json.load(f)
        entry = next(iter(meta.values()))
        in_path = entry["annotation"]
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"annotation file not found: {in_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    n = 0
    with open(in_path, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = convert_record(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 10000 == 0:
                print(f"processed {n} lines...", flush=True)
    print(f"done: {n} lines -> {args.output}")


if __name__ == "__main__":
    main()
