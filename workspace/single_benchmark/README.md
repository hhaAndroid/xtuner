# Qwen3.5-35B-A3B MTP Token-Id Benchmark

Start the LMDeploy server:

```bash
workspace/single_benchmark/start_lmdeploy_mtp.sh
```

Send one token-id request to LMDeploy in both non-streaming and streaming modes:

```bash
python workspace/single_benchmark/request_lmdeploy.py --case both
```

Start the SGLang server:

```bash
workspace/single_benchmark/start_sglang_mtp.sh
```

The SGLang script defaults to two GPUs with `TP_SIZE=2 EP_SIZE=1 DP_SIZE=1`,
`SPECULATIVE_ALGORITHM=EAGLE`, `SPECULATIVE_DRAFT_MODEL_PATH=MODEL_PATH`,
`SPECULATIVE_NUM_STEPS=3`, `SPECULATIVE_NUM_DRAFT_TOKENS=4`,
`MAMBA_SCHEDULER_STRATEGY=extra_buffer`, and disabled piecewise CUDA graph.
This follows the SGLang path used by the RL config in xtuner. Override these
environment variables if the target machine has a different layout.

Send the same token-id request to SGLang:

```bash
python workspace/single_benchmark/request_sglang.py --case both
```
