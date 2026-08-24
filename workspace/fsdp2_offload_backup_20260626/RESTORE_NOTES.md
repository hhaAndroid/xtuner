# FSDP2 offload debug/fix backup

This directory backs up the changes from the FSDP2 deferred all-gather memory investigation.

Backed up files:

- `worker.py.with_fsdp2_offload_debug`: full snapshot of `xtuner/v1/rl/trainer/worker.py` before temporarily removing the FSDP2 offload fix/debug code.
- `fsdp2_deferred_allgather_memory_review.md`: investigation write-up.

When restoring, do not blindly overwrite `worker.py` if the branch has moved. Re-apply only the FSDP2 offload pieces:

- `import gc`
- `TrainingWorker.offload_model()` gated debug calls and `_release_deferred_fsdp_all_gathers(...)`
- `_log_offload_memory_debug(...)`
- `_log_live_cuda_tensors(...)`
- `_format_tensor_referrers(...)`

Do not treat the routed-expert `ray.internal.free(...)` diff as part of this backup unless explicitly requested.
