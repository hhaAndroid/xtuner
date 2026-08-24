import json
from pathlib import Path


def main():
    src = Path("/mnt/shared-storage-user/llmrazor-share/data/slime_data/dapo-math-17k/dapo-math-17k.jsonl")
    dst = Path("/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/meta_data/dapo_math_17k_slime_xtuner.jsonl")

    total = 0
    with src.open() as fin, dst.open("w") as fout:
        for idx, line in enumerate(fin):
            item = json.loads(line)
            out = {
                "data_source": "math_dapo",
                "prompt": item["prompt"],
                "ability": "MATH",
                "reward_model": {
                    "ground_truth": item["label"],
                    "style": "deepscaler",
                },
                "extra_info": {
                    "index": idx,
                    "label": item["label"],
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            total += 1

    print(json.dumps({"src": str(src), "dst": str(dst), "total": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
