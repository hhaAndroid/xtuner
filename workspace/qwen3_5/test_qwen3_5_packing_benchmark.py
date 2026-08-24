"""Full 4-dimensional benchmark for VLM attention cost."""
import itertools
from collections import namedtuple
from dataclasses import dataclass
import os
import time
import torch
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
import argparse
from transformers import AutoTokenizer
import time
from xtuner.v1.utils import Config
import torch
import os
from xtuner.v1.loss import CELossContext
import json
import torch.distributed as dist
import sys
import numpy as np
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import get_logger
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    ParallelConfigException,
    StrEnum,
    get_logger,
    is_hf_model_path,
    log_format,
    profile_time_and_memory,
    record_git_info,
)

# ─── Benchmark constants ───────────────────────────────────────────────────────
TOTAL_TOKENS = 32768
FLASH_ATTN_BLOCK_SIZE = 128
IMG_TOKEN_ALIGN = 16      # image token count aligned to merge_size^2 * something
MIN_IMG_TOKENS = 256       # minimum tokens per image (adjust to your model)
MIN_TEXT_TOKENS = 2048     # minimum tokens per text sub-sequence

N_IMAGE_CASES = 10        # axis 2: how many different num_images to test
N_TEXT_CASES = 3         # axis 3: how many different num_text_seqs to test
N_RANDOM_TRIALS = 10      # axis 4: random draws per concrete (n_img, n_text) case

# Axis 1: image/text token ratio
IMG_RATIO_LIST = [0.0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


# ─── Result record ─────────────────────────────────────────────────────────────
@dataclass
class BenchmarkRecord:
    img_ratio: float
    num_images: int
    num_text_seqs: int
    trial: int
    elapsed: float
    llm_num_patch: float
    img_num_patch: float
    imgpatch_div_4: float
    sum_num_tokens: int
    sum_image_tokens: int


# ─── Token-splitting helpers ───────────────────────────────────────────────────
def _random_split(
    total: int,
    n: int,
    min_size: int,
    align: int,
    rng: np.random.Generator,
) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [total]

    reserve = n * min_size
    if reserve >= total:
        return [min_size] * n

    remainder = total - reserve
    fractions = rng.dirichlet(np.ones(n))
    # Floor to align (no minimum=align trick)
    extra = [(int(f * remainder) // align) * align for f in fractions]
    chunks = [min_size + e for e in extra]

    # Exact correction on first element (no re-align, guarantees sum=total)
    diff = total - sum(chunks)
    chunks[0] += diff
    return chunks


def _geom_range(start: int, stop: int, n: int) -> list[int]:
    """n geometrically spaced integers from start to stop (inclusive), deduplicated."""
    raw = np.geomspace(start, stop, n).round().astype(int)
    return sorted(set(raw.tolist()))


# 在文件顶部常量区增加：
IMAGE_TOKEN_ID = 248056       # Qwen3.5-VL image_token_id
PATCH_PIXEL_DIM = 1536        # 3 * 2 * 16 * 16 (in_channels * temporal_patch_size * patch_size^2)
SPATIAL_MERGE_SIZE = 2

def make_seq_ctx(
    img_ratio: float,
    num_images: int,
    num_text_seqs: int,
    seed: int,
    total_tokens: int = TOTAL_TOKENS,
    device: str = "cuda",
) -> SequenceContext:
    rng = np.random.default_rng(seed)

    img_budget = round(total_tokens * img_ratio)
    text_budget = total_tokens - img_budget

    # ── image sub-sequences ────────────────────────────────────────────────────
    if num_images > 0 and img_budget > 0:
        img_token_list = _random_split(img_budget, num_images, MIN_IMG_TOKENS, IMG_TOKEN_ALIGN, rng)
        img_seq_lens = img_token_list
        img_num_img_tokens: list[list[int]] = [[t] for t in img_token_list]
    else:
        img_token_list = []
        img_seq_lens = []
        img_num_img_tokens = []

    # ── text sub-sequences ─────────────────────────────────────────────────────
    if num_text_seqs > 0 and text_budget > 0:
        text_len_list = _random_split(text_budget, num_text_seqs, MIN_TEXT_TOKENS, FLASH_ATTN_BLOCK_SIZE, rng)
        text_seq_lens = text_len_list
        text_num_img_tokens: list[list[int]] = [[] for _ in text_len_list]
    else:
        text_seq_lens = []
        text_num_img_tokens = []

    all_seq_lens = img_seq_lens + text_seq_lens
    all_num_img_tokens = img_num_img_tokens + text_num_img_tokens
    assert all_seq_lens, f"Empty sequence for ratio={img_ratio} n_img={num_images} n_text={num_text_seqs}"

    # ── input_ids: 图片位置填 image_token_id，文本位置填随机 token ──────────────
    input_ids_parts = []
    for seq_len in img_seq_lens:
        # Must match image_token_id so get_placeholder_mask can locate visual slots
        input_ids_parts.append(torch.full((1, seq_len), IMAGE_TOKEN_ID, dtype=torch.long))
    for seq_len in text_seq_lens:
        input_ids_parts.append(torch.randint(0, 32000, (1, seq_len), dtype=torch.long))

    actual_total = sum(all_seq_lens)
    input_ids = torch.cat(input_ids_parts, dim=1).to(device)  # (1, total_len)

    # ── image_grid_thw & pixel_values ─────────────────────────────────────────
    # For n image tokens: t=1, h=2, w=2*n  →  merged_tokens = 1*(2/2)*(2n/2) = n ✓
    # num_patches = t*h*w = 4*n, satisfies origin_pixel_len % 4 == 0 ✓
    image_grid_thw_list = []
    pixel_values_list = []
    for n in img_token_list:
        t, h, w = 1, 2, 2 * n
        num_patches = t * h * w  # = 4 * n
        image_grid_thw_list.append([t, h, w])
        pixel_values_list.append(
            torch.randn(num_patches, PATCH_PIXEL_DIM, dtype=torch.bfloat16)
        )

    if image_grid_thw_list:
        image_grid_thw = torch.tensor(image_grid_thw_list, dtype=torch.long).to(device)
        pixel_values = torch.cat(pixel_values_list, dim=0).to(device)
    else:
        image_grid_thw = None
        pixel_values = None

    # ── position_ids: 3D (3, 1, seq_len) 满足 mrope ──────────────────────────
    # 对 benchmark 来说，所有维度用相同的递增位置即可
    pos = torch.arange(actual_total, dtype=torch.long, device=device)
    position_ids = pos.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()  # (3, 1, seq_len)

    # ── cu_seq_lens ────────────────────────────────────────────────────────────
    cu_seq_lens = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        torch.cumsum(torch.tensor(all_seq_lens, dtype=torch.int32), dim=0),
    ]).to(device).to(torch.int32)
    
    return SequenceContext(
        input_ids=input_ids,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=max(all_seq_lens),
        max_length_k=max(all_seq_lens),
        num_img_tokens=all_num_img_tokens,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        position_ids=position_ids,
        device=device,
    )


# ─── Axis-range builders (adapted per ratio) ──────────────────────────────────
def _num_images_range(img_budget: int) -> list[int]:
    """10 geometrically spaced image counts feasible for this image budget."""
    max_images = img_budget // MIN_IMG_TOKENS
    if max_images <= 1:
        return [1] if img_budget >= MIN_IMG_TOKENS else []
    return _geom_range(1, min(max_images, 64), N_IMAGE_CASES)


def _num_text_seqs_range(text_budget: int) -> list[int]:
    """10 geometrically spaced text-seq counts feasible for this text budget."""
    max_seqs = text_budget // MIN_TEXT_TOKENS
    if max_seqs <= 1:
        return [1] if text_budget >= MIN_TEXT_TOKENS else []
    return _geom_range(1, min(max_seqs, 256), N_TEXT_CASES)


def benchmark_packing(model, seq_ctx, loss_ctx) -> None:
    """Benchmark packing with fixed total length 32k."""
    num_tokens = seq_ctx.cu_seq_lens_k[1:] - seq_ctx.cu_seq_lens_k[:-1]
    flash_attn_block_size = 128
    
    if len(seq_ctx.num_img_tokens) < len(num_tokens):
        # If num_img_tokens is shorter than num_tokens, pad it with zeros
        seq_ctx.num_img_tokens = seq_ctx.num_img_tokens + [[0]] * (len(num_tokens) - len(seq_ctx.num_img_tokens))

    assert len(seq_ctx.num_img_tokens) == len(num_tokens), "num_img_tokens should have the same length as num_tokens, but got {} and {}".format(len(seq_ctx.num_img_tokens), len(num_tokens))
    sum_llm_num_patch = 0
    sum_img_num_patch = 0
    sum_img_num_patch_div_4=0
    for num_token, num_img_token in zip(num_tokens, seq_ctx.num_img_tokens):
        llm_num_patch = (round(int(num_token) / flash_attn_block_size)) ** 2
        img_num_patch1 = sum((n / flash_attn_block_size) ** 2 for n in num_img_token)
        img_num_patch = sum((n*4 / flash_attn_block_size) ** 2 for n in num_img_token)
        sum_llm_num_patch += llm_num_patch
        if isinstance(img_num_patch,torch.Tensor):
            img_num_patch = img_num_patch.item()
            img_num_patch1 = img_num_patch1.item()
        sum_img_num_patch += img_num_patch
        sum_img_num_patch_div_4 += img_num_patch1

    start_time = time.time()
    with torch.no_grad():
        output = model(
                    seq_ctx=seq_ctx,
                    loss_ctx=loss_ctx,
                    )
        loss=output.loss
    torch.cpu.synchronize()
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time, sum_llm_num_patch, round(sum_img_num_patch,2), round(sum_img_num_patch_div_4,2)

def _is_deterministic_split(total: int, n: int, min_size: int) -> bool:
    """Returns True when _random_split produces the same output regardless of rng."""
    return n <= 1 or total <= 0 or n * min_size >= total

# ─── Main benchmark loop ───────────────────────────────────────────────────────
def run_benchmark(model, loss_cfg, rank: int = 0, world_size: int = 1, device: str = "cuda", logger=None) -> list[BenchmarkRecord]:
    results: list[BenchmarkRecord] = []

    # Pre-count total cases across ALL ranks (same logic as before)
    total_cases = 0
    for img_ratio in IMG_RATIO_LIST:
        img_budget = round(TOTAL_TOKENS * img_ratio)
        text_budget = TOTAL_TOKENS - img_budget
        num_images_list = _num_images_range(img_budget) if img_ratio > 0.0 else [0]
        num_text_seqs_list = _num_text_seqs_range(text_budget) if img_ratio < 1.0 else [0]
        for num_images, num_text_seqs in itertools.product(num_images_list, num_text_seqs_list):
            img_det = _is_deterministic_split(img_budget, num_images, MIN_IMG_TOKENS)
            text_det = _is_deterministic_split(text_budget, num_text_seqs, MIN_TEXT_TOKENS)
            total_cases += 1 if (img_det and text_det) else N_RANDOM_TRIALS

    if rank == 0:
        logger.info(f"Total benchmark cases: {total_cases} across {world_size} GPUs (~{total_cases // world_size} per GPU)")

    global_idx = 0   # ← global serial index across all cases
    local_count = 0

    for img_ratio in IMG_RATIO_LIST:
        img_budget = round(TOTAL_TOKENS * img_ratio)
        text_budget = TOTAL_TOKENS - img_budget
        num_images_list = _num_images_range(img_budget) if img_ratio > 0.0 else [0]
        num_text_seqs_list = _num_text_seqs_range(text_budget) if img_ratio < 1.0 else [0]

        for num_images, num_text_seqs in itertools.product(num_images_list, num_text_seqs_list):
            img_det = _is_deterministic_split(img_budget, num_images, MIN_IMG_TOKENS)
            text_det = _is_deterministic_split(text_budget, num_text_seqs, MIN_TEXT_TOKENS)
            n_trials = 1 if (img_det and text_det) else N_RANDOM_TRIALS

            for trial in range(n_trials):
                if global_idx % world_size == rank:      # ← this rank owns this case
                    seq_ctx = make_seq_ctx(img_ratio, num_images, num_text_seqs, seed=trial, device=device)
                    shifted_labels = seq_ctx.input_ids.clone()
                    loss_ctx = loss_cfg.build(shifted_labels=shifted_labels)
                    loss_ctx = CELossContext.build_batches(
                        [loss_ctx], cu_seq_lens_list=[seq_ctx.cu_seq_lens_q]
                    )[0]
                    elapsed, llm_patch, img_patch, imgpatch_div_4 = benchmark_packing(model, seq_ctx, loss_ctx)
                    
                    num_tokens = seq_ctx.cu_seq_lens_k[1:] - seq_ctx.cu_seq_lens_k[:-1]
                    sum_num_tokens = sum(num_tokens).item()
                    sum_image_tokens = sum(sum(t) for t in seq_ctx.num_img_tokens)

                    rec = BenchmarkRecord(
                        img_ratio=img_ratio, 
                        num_images=num_images, 
                        num_text_seqs=num_text_seqs,
                        trial=trial, 
                        elapsed=elapsed,
                        llm_num_patch=llm_patch, 
                        img_num_patch=img_patch, 
                        imgpatch_div_4=imgpatch_div_4,
                        sum_num_tokens=sum_num_tokens, 
                        sum_image_tokens=sum_image_tokens
                        )
                    results.append(rec)
                    local_count += 1

                    logger.info(
                        f"[rank={rank} {local_count} gidx={global_idx}/{total_cases}] "
                        f"num_img/llm_token={sum_image_tokens}/{sum_num_tokens} img_ratio={img_ratio} "
                        f"time={elapsed:.3f}s llm_patch={llm_patch} img_patch={img_patch:.2f} "
                        f"imgpatch_div_4={imgpatch_div_4:.2f} num_images={num_images} "
                        f"num_text_seqs={num_text_seqs} trial={trial}"
                    )
                global_idx += 1

    return results

def warmup(model, loss_cfg, n_warmup: int = 10, device: str = "cuda") -> None:
    """Run n_warmup forward passes before benchmarking to warm up CUDA kernels."""
    print(f"Warming up ({n_warmup} iterations)...")
    # Use a mid-range case: 50% image ratio, 4 images, 4 text seqs
    for i in range(n_warmup):
        seq_ctx = make_seq_ctx(0.5, 4, 4, seed=i, device=device)
        shifted_labels = seq_ctx.input_ids.clone()
        loss_ctx = loss_cfg.build(shifted_labels=shifted_labels)
        loss_ctx = CELossContext.build_batches(
            [loss_ctx],
            cu_seq_lens_list=[seq_ctx.cu_seq_lens_q],
        )[0]
        with torch.no_grad():
            output = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
            _ = output.loss
        torch.cuda.synchronize()
    print("Warmup done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/configs/sft_qwen35vl_35b_config.py')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    rank = args.rank
    world_size = args.world_size
    # 注意：不调用 dist.init_process_group()
    torch.cuda.set_device(0)
    device = f"cuda:{0}"

    trainer_cfg = Config.fromfile(args.cfg)['trainer']
    tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path, trust_remote_code=True)
    loss_cfg = trainer_cfg.loss_cfg
        
    log_dir = trainer_cfg.work_dir
    log_level = os.environ.get("XTUNER_LOG_LEVEL", "INFO").upper()
    logger = get_logger()
    logger.remove()
    logger.add(log_dir / f"rank{rank}.log", format=log_format(), backtrace=True, catch=True)
    # Set log level to hide debug output
    logger.add(sys.stderr, format=log_format(rank=rank), level=log_level)

    QWEN3_VL_MOE_PATH = '/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
    with torch.device("meta"):
        model_cfg = Qwen3_5_VLMoE35BA3Config(compile_cfg=False)
        model = model_cfg.build().to(torch.bfloat16)
    model.from_hf(QWEN3_VL_MOE_PATH)
    model.to(device).eval()

    warmup(model, loss_cfg, n_warmup=10, device=device)

    benchmark_results = run_benchmark(model, loss_cfg, rank=rank, world_size=world_size, device=device, logger=logger)

    # Save per-rank results
    out_path = f"benchmark_results_rank{rank}.json"
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in benchmark_results], f, indent=2)
    logger.info(f"[rank={rank}] Saved {len(benchmark_results)} records → {out_path}")

