"""Parse two training log files and plot step vs tgs / e2e_tgs as 2 subplots."""
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse training log and return (steps, tgs, e2e_tgs) arrays."""
    steps, tgs_list, e2e_list = [], [], []
    step_pat   = re.compile(r"Step\s+(\d+)/\d+")
    tgs_pat    = re.compile(r"(?<![_\w])tgs:\s*([\d.]+)")
    e2e_pat    = re.compile(r"e2e_tgs:\s*([\d.]+)")

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            sm = step_pat.search(line)
            tm = tgs_pat.search(line)
            em = e2e_pat.search(line)
            if sm and tm and em:
                steps.append(int(sm.group(1)))
                tgs_list.append(float(tm.group(1)))
                e2e_list.append(float(em.group(1)))

    if not steps:
        raise ValueError(f"No matching lines found in {file_path}")
    print(f"[{Path(file_path).name}] {len(steps)} steps parsed, "
          f"step range: {steps[0]}–{steps[-1]}")
    return np.array(steps), np.array(tgs_list), np.array(e2e_list)


def plot_tgs(files: list[str], output_path: str = "tgs_comparison.png") -> None:
    datasets = [parse_log(f) for f in files]

    # Clip to the minimum step count across experiments
    min_len = min(len(d[0]) for d in datasets)
    datasets = [(s[:min_len], t[:min_len], e[:min_len]) for s, t, e in datasets]
    labels = [Path(f).stem for f in files]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for ax, (metric_idx, title, ylabel) in zip(
        axes,
        [(1, "TGS (Tokens / GPU / Second)", "tgs"),
         (2, "E2E-TGS (End-to-End TGS)",    "e2e_tgs")],
    ):
        for i, (steps, tgs, e2e) in enumerate(datasets):
            values = tgs if metric_idx == 1 else e2e
            ax.plot(steps, values, color=colors[i % len(colors)],
                    linewidth=1.0, alpha=0.7, label=labels[i])
            # Smoothed trend (rolling mean, window = 5% of data)
            w = max(1, len(values) // 20)
            smoothed = np.convolve(values, np.ones(w) / w, mode="valid")
            x_smooth = steps[w - 1:]
            ax.plot(x_smooth, smoothed, color=colors[i % len(colors)],
                    linewidth=2.0, linestyle="--")

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training throughput comparison", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_tgs.py <log1.txt> <log2.txt> [output.png]")
        sys.exit(1)
    out = sys.argv[3] if len(sys.argv) > 3 else "tgs_comparison.png"
    plot_tgs(sys.argv[1:3], output_path=out)