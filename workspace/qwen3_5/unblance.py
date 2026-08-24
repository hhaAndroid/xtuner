import re
import sys
import statistics
from collections import defaultdict

def parse_tgs_from_log(filepath: str) -> dict[int, list[float]]:
    pattern = re.compile(
        r'\[XTuner\]\[RANK\s*\d+\].*?Step\s+(\d+)/\d+.*?\btgs:\s*([\d.]+)'
    )
    step_tgs: dict[int, list[float]] = defaultdict(list)
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step_tgs[int(m.group(1))].append(float(m.group(2)))
    return step_tgs


def imbalance(tgs_list: list[float]) -> float:
    """Coefficient of variation (std/mean) — scale-free imbalance metric."""
    if len(tgs_list) < 2:
        return 0.0
    mean = statistics.mean(tgs_list)
    return statistics.stdev(tgs_list) / mean if mean > 0 else 0.0


def analyze(filepath: str) -> None:
    step_tgs = parse_tgs_from_log(filepath)
    if not step_tgs:
        print("No matching log lines found.")
        return

    step_scores: dict[int, float] = {
        step: imbalance(vals) for step, vals in step_tgs.items()
    }

    # ── 降序打印 ───────────────────────────────────────────────────────────────
    for step, score in sorted(step_scores.items(), key=lambda x: -x[1]):
        vals = step_tgs[step]
        print(
            f"Step {step:6d} | ranks={len(vals):3d} | "
            f"min={min(vals):8.1f}  max={max(vals):8.1f}  "
            f"mean={statistics.mean(vals):8.1f} | imbalance={score:.4f}"
        )

    scores = list(step_scores.values())
    worst_step = max(step_scores, key=step_scores.get)

    print("\n" + "=" * 70)
    print(f"Most imbalanced step : Step {worst_step}  "
          f"(imbalance = {step_scores[worst_step]:.4f})")
    worst_vals = sorted(step_tgs[worst_step])
    print(f"  TGS per rank       : {worst_vals}")
    print(f"  min={min(worst_vals):.1f}  max={max(worst_vals):.1f}  "
          f"mean={statistics.mean(worst_vals):.1f}")
    print(f"\nImbalance across all steps:")
    print(f"  Mean     : {statistics.mean(scores):.4f}")
    print(f"  Variance : {statistics.variance(scores):.6f}")


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "train.log")