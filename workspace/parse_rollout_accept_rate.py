#!/usr/bin/env python3
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ROLLOUT_START_RE = re.compile(
    r"\[XTuner\]\[(?P<ts>[^]]+)\]\[INFO\] Rollout (?P<rollout>\d+)/\d+ start"
)
ACCEPT_RATE_RE = re.compile(
    r"Decode batch, .*?accept rate: (?P<rate>\d+(?:\.\d+)?)"
)
MTP_LOSS_RE = re.compile(
    r"Rollout (?P<rollout>\d+) Step (?P<step>\d+): .*?reduced_mtp_loss=(?P<loss>\d+(?:\.\d+)?)"
)


@dataclass
class RolloutStats:
    rollout: int
    start_ts: str
    start_line: int
    accept_rates: list[float] = field(default_factory=list)
    mtp_losses: list[float] = field(default_factory=list)

    @property
    def mean_accept_rate(self) -> float:
        return sum(self.accept_rates) / len(self.accept_rates)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(log_path: Path) -> list[RolloutStats]:
    stats: list[RolloutStats] = []
    current: RolloutStats | None = None
    rollout_occurrence: dict[int, int] = {}
    rollout_to_stat: dict[tuple[int, int], RolloutStats] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = strip_ansi(raw_line.rstrip("\n"))

            rollout_match = ROLLOUT_START_RE.search(line)
            if rollout_match:
                rollout = int(rollout_match.group("rollout"))
                occurrence = rollout_occurrence.get(rollout, 0) + 1
                rollout_occurrence[rollout] = occurrence
                current = RolloutStats(
                    rollout=rollout,
                    start_ts=rollout_match.group("ts"),
                    start_line=lineno,
                )
                stats.append(current)
                rollout_to_stat[(rollout, occurrence)] = current
                continue

            mtp_match = MTP_LOSS_RE.search(line)
            if mtp_match:
                rollout = int(mtp_match.group("rollout"))
                occurrence = rollout_occurrence.get(rollout, 0)
                stat = rollout_to_stat.get((rollout, occurrence))
                if stat is not None:
                    stat.mtp_losses.append(float(mtp_match.group("loss")))
                continue

            if current is None:
                continue

            accept_match = ACCEPT_RATE_RE.search(line)
            if accept_match:
                current.accept_rates.append(float(accept_match.group("rate")))

    return stats


def print_stats(stats: list[RolloutStats]) -> int:
    printed = 0
    for idx, item in enumerate(stats, start=1):
        if not item.accept_rates:
            continue
        printed += 1
        mtp_loss = (
            f"{sum(item.mtp_losses) / len(item.mtp_losses):.6f}"
            if item.mtp_losses
            else "nan"
        )
        print(
            f"step={idx} "
            f"avg_accept_rate={item.mean_accept_rate:.6f} "
            f"mtp_loss={mtp_loss}"
        )
    return printed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the average decode accept rate for each rollout step."
    )
    parser.add_argument("log_path", type=Path, help="Path to the training log file")
    args = parser.parse_args()

    if not args.log_path.is_file():
        print(f"Log file not found: {args.log_path}", file=sys.stderr)
        return 1

    stats = parse_log(args.log_path)
    printed = print_stats(stats)
    if printed == 0:
        print("No rollout accept-rate records found.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
