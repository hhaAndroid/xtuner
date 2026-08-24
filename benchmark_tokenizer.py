"""Benchmark tokenizer speed: Qwen3 fast vs intern-s1-mini fast at different sequence lengths."""

import time
import statistics
from transformers import AutoTokenizer

QWEN3_PATH = "/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B"
INTERN_PATH = "/mnt/shared-storage-user/llmrazor-share/model/intern-s1-mini"
SEQ_LENGTHS = [8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000]
WARMUP_RUNS = 2
BENCH_RUNS = 3

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models require large amounts of data to train effectively. "
    "Natural language processing enables computers to understand human language. "
) * 20_000  # ~5.4M chars, enough for 512k tokens


def build_sample(tokenizer, target_tokens: int) -> str:
    """Return a string that encodes to approximately target_tokens tokens."""
    ids = tokenizer.encode(BASE_TEXT, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def bench(tokenizer, text: str, runs: int) -> list[float]:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tokenizer(text, add_special_tokens=False)
        times.append(time.perf_counter() - t0)
    return times


def fmt(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000:.1f}ms"


def main() -> None:
    print(f"Loading Qwen3 fast tokenizer from {QWEN3_PATH} ...")
    tok_qwen3 = AutoTokenizer.from_pretrained(QWEN3_PATH, use_fast=True)
    print(f"Loading intern-s1-mini fast tokenizer from {INTERN_PATH} ...")
    tok_intern = AutoTokenizer.from_pretrained(INTERN_PATH, use_fast=True, trust_remote_code=True)
    print("Done.\n")

    # Build samples using each tokenizer separately (different vocabs → different token counts)
    print("Building text samples ...")
    samples_qwen3: dict[int, str] = {}
    samples_intern: dict[int, str] = {}
    for length in SEQ_LENGTHS:
        samples_qwen3[length] = build_sample(tok_qwen3, length)
        samples_intern[length] = build_sample(tok_intern, length)
        actual_q = len(tok_qwen3.encode(samples_qwen3[length], add_special_tokens=False))
        actual_i = len(tok_intern.encode(samples_intern[length], add_special_tokens=False))
        print(f"  target {length:>7,} → qwen3: {actual_q:,} tokens  |  intern: {actual_i:,} tokens")
    print()

    # Header
    col_w = 12
    header = (
        f"{'seq_len':>10} | "
        f"{'qwen3_avg':>{col_w}} {'qwen3_p50':>{col_w}} | "
        f"{'intern_avg':>{col_w}} {'intern_p50':>{col_w}} | "
        f"{'qwen3/intern':>12}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for length in SEQ_LENGTHS:
        text_q = samples_qwen3[length]
        text_i = samples_intern[length]

        # Warmup
        bench(tok_qwen3, text_q, WARMUP_RUNS)
        bench(tok_intern, text_i, WARMUP_RUNS)

        qwen3_times = bench(tok_qwen3, text_q, BENCH_RUNS)
        intern_times = bench(tok_intern, text_i, BENCH_RUNS)

        q_avg = statistics.mean(qwen3_times)
        q_p50 = statistics.median(qwen3_times)
        i_avg = statistics.mean(intern_times)
        i_p50 = statistics.median(intern_times)
        ratio = q_avg / i_avg if i_avg > 0 else float("inf")

        print(
            f"{length:>10,} | "
            f"{fmt(q_avg):>{col_w}} {fmt(q_p50):>{col_w}} | "
            f"{fmt(i_avg):>{col_w}} {fmt(i_p50):>{col_w}} | "
            f"{ratio:>11.2f}x"
        )


if __name__ == "__main__":
    main()
