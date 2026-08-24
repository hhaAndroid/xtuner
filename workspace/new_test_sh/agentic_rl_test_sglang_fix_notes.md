# Agentic RL + SGLang 两步跑通记录

日期：2026-08-05

## 1. 结论

目标脚本 `workspace/new_test_sh/agentic_rl_test_sglang.sh` 已在 8 张 GPU 上真实完成 2 个 train step，进程退出码为 0，并保存了 `hf-step-2` checkpoint。

本次运行参数为：

```bash
TRAIN_BATCH_SIZE=8
PROMPT_REPEAT_K=2
MAX_CONCURRENT_SAMPLES=16
TOTAL_TRAIN_STEPS=2
AUTO_RESUME=False
```

运行产物：

- 总日志：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang/codex_2step_20260805_1149/training_log_080511.txt`
- 两步 rollout：`.../20260805114919/train_rollout/train_rollout_{1,2}.jsonl`
- 最终 checkpoint：`.../20260805114919/hf/hf-step-2`
- Prefix cache 指标快照：`.../codex_2step_20260805_1149/prefix_cache_metrics_snapshot.txt`

## 2. 修改内容

### 2.1 XTuner

`xtuner/v1/rl/rollout/session_server.py`

- 给 SessionServer 增加 `rollout_backend`，区分 LMDeploy 和 SGLang 协议。
- 适配 SGLang OpenAI 请求：修正 `top_k=0` 与 SGLang `top_k=-1` 的语义差异，并转换 logprob 参数。
- 保留 messages，供 SGLang reasoning/tool parser 使用，同时发送预先 tokenized 的 `input_ids`。
- SSE 响应逐 chunk 从 SGLang 转发给 agent client；训练 trace 在旁路累积，不阻塞正文流式输出。
- 解析 SGLang 终止 chunk 中的 `sglext.output_ids`、`output_token_logprobs` 和 `routed_experts`，以重建精确训练轨迹。
- routed experts shared-store actor 根据 backend 选择 `sglang` 或 `lmdeploy` namespace。

`xtuner/v1/rl/rollout/worker.py`

- 创建 SessionServer actor 时把 `rollout_backend` 传进去。

`examples/v1/config/agentic_rl_qwen3p5vl_mtp_ep_code_sglang.py`

- routed experts 固定开启，不再考虑关闭分支。
- Rollout 使用 SGLang EAGLE/MTP：2 speculative steps、top-k 1、3 draft tokens。
- 训练侧启用 `MTPConfig(num_layers=4, share_weights=True)`。
- 启用 `MambaRadixCache` 和 LPM 调度；通过 `SGLANG_MAMBA_RADIX_CACHE_STRATEGY` 支持
  `extra_buffer`/`no_buffer` 两种策略，默认使用已通过两步验证的 `extra_buffer`。
- 开启 SGLang metrics，允许长 completion 根据实际 context 自动截断。
- `extra_buffer` 自动开启 overlap schedule，`no_buffer` 自动关闭；A/B 结果另见
  `workspace/new_test_sh/agentic_rl_sglang_mamba_cache_ab.md`。

`workspace/new_test_sh/agentic_rl_test_sglang.sh`

- routed experts 固定为 `True`。
- 默认 train step 数改为 2，仍可由环境变量覆盖。

### 2.2 SGLang 独立仓库

仓库：`/mnt/shared-storage-user/huanghaian/code/slime_package/xtuner_sglang/sglang`

- `entrypoints/openai/protocol.py`：增加 `return_token_ids`，并在 `SglExt` 中增加 output ids/logprobs。
- `entrypoints/openai/serving_chat.py`：流式和非流式响应都能按请求返回精确 output ids/logprobs；流式情况下放在终止 `sglext` chunk 中。
- `managers/tokenizer_manager.py`、`detokenizer_manager.py`：routed experts 改走 Ray shared store。
- 新增 `managers/routed_experts_shared_store.py`：保存大体积 routed-expert ndarray，HTTP 响应只传 key；失败时回退 base64。

这样解决了两个主要问题：OpenAI SSE 原本缺少训练所需的精确 token 元数据，以及 routed experts 直接 base64 编码导致的巨大 HTTP 数据传输。

## 3. 运行验证

- SGLang 实际加载 `Qwen3_5ForCausalLMMTP`；EAGLE 实测 accept rate 多数约为 0.64～0.94。
- 训练模型中实际存在 `MTPBlock/MTPLayer`。
- Step 1：`reduced_mtp_loss=0.9552`，`grad_norm=0.1658`，随后权重成功同步回 rollout workers。
- Step 2：两个训练 batch 的 `reduced_mtp_loss=0.8450/0.8778`，梯度均非零。
- 日志最终出现 `Train step 2/2`、checkpoint 保存成功以及 `Ray shutdown successfully`。
- Routed-expert cache 实际分配：每个 TP rank device cache 为 `(8192, 40, 8)`，host cache 约 3.17 GiB。
- 四个 SGLang endpoint 合计处理 507,133 个 prompt tokens，其中 331,231 个命中 device prefix cache，总命中率 65.31%；单次多轮请求最高观察到约 99.63%。

## 4. 尚未解决或需要继续确认的问题

以下问题不阻塞本次两步训练，但建议后续处理：

1. **Oversample 取消链路仍可能卡 300 秒。** Step 2 已取得 8 个有效 group 后，有 1 个额外 agent task 没有响应 pause/abort，最终由 `PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S=300` 强制 cancel。它使 Step 2 的 `produce_batch` 达到 589 秒。需要给 rollout/session/tool request 统一 request-id，并追踪到底卡在 agent tool、SessionServer drain 还是客户端取消传播。

2. **Shared store 可能留下 orphan object。** 正常响应会 `get + pop`，但客户端断开或 oversample 被取消时，终止 metadata 可能无人消费；当前 detached actor 没有 TTL、容量上限或按 step 清理机制，长时间训练可能产生 Ray object-store 泄漏。

3. **65.31% 是服务级指标，不是有效训练样本专属指标。** 它包含 cold first turn、完成请求和后来被 abort 的 oversample 请求。若要严格评估训练数据的 cache 收益，需要按 session/request-id 记录 prompt/cached tokens，并只汇总最终入 batch 的 group。

4. **Prefix cache A/B 仍需增加重复次数。** `extra_buffer + overlap` 后续也已通过两步流程，单次测试吞吐优于 `no_buffer`，且未发现 Mamba slot 泄漏；但两次 cache metrics 不是严格同口径，不能据此判断哪种策略必然有更高命中率。

5. **HTTP 是完整 SSE 流式，但 engine chunk 当前为 cumulative 模式。** `sglang_incremental_streaming_output=False` 时 SGLang 内部返回累计文本，OpenAI serving 层再切成 delta；agent 仍逐 chunk 接收。如果要求 engine 原生 incremental chunk，需要设为 `True` 后重新验证 reasoning/tool parser 和终止 metadata。

6. **少量非致命日志噪声。** 包括一个 `Tool 'None' is not defined`、activation-checkpoint 的 `inputs have requires_grad=False` warning，以及未使用音频 processor 的 TorchCodec/FFmpeg import traceback。它们未阻塞本次训练，且实际梯度非零，但前两个值得单独定位以免影响样本质量或掩盖真正异常。

7. **修改尚未整理成提交，也未跑完整测试套件。** SGLang 修改位于另一个 git 仓库；`workspace/` 又被 `.git/exclude` 忽略，整理提交时需要分别处理。当前已完成 SessionServer/rollout 的 58 个定向测试，但不能替代完整测试套件。

目前没有已知的两步训练正确性阻塞项；最优先建议处理第 1、2 项，即取消传播和 shared-store 生命周期。

## 5. LMDeploy 后端兼容性审计（2026-08-06）

本次 SGLang 支持对 LMDeploy 有共享代码改动，但已按 backend 隔离：

- `SessionServer._adapt_worker_request()` 只在 `rollout_backend == "sglang"` 时转换 `top_k` 和 logprob 参数；`lmdeploy` 以及兼容旧调用的 `None` 都保持请求字典逐字段不变。
- LMDeploy 仍使用原有 `input_ids`、`return_token_ids`、`return_logprob`、`return_routed_experts` 协议；SGLang 的 `sglext` 解析只是额外 fallback，不覆盖 LMDeploy choice 中的原字段。
- 流式响应同时存在 LMDeploy 的 `output_token_logprobs` 和标准 OpenAI `logprobs` 时，训练 trace 仍优先使用前者，保持原有精确 token-id 对齐逻辑。
- routed-expert key 在 LMDeploy 后端仍从 Ray `namespace="lmdeploy"` 的 `shared_store` actor 读取；只有 SGLang 后端切到 `namespace="sglang"`。
- `enable_return_routed_experts=True` 的现有 LMDeploy 配置行为不变；为 `False` 时不再无条件向后端索取 routed experts，这与配置语义一致，也减少不支持该扩展时的风险。
- SGLang 独立仓库的修改不会被 LMDeploy worker 导入；LMDeploy 脚本仍通过自己的 `LMDEPLOY_PATH` 启动原后端。

验证结果：

- 本地 LMDeploy `0.15.0` 可以导入，`RolloutConfig.rollout_backend` 正确解析为 `lmdeploy`，worker 仍分派到 `LMDeployWorker`，启动方式保持 `ray`、`rollout_cross_node_comm=True`。
- `tests/rl/test_session_server_sglang.py` 新增 LMDeploy 请求不变、流式扩展字段聚合和 shared-store namespace 回归测试。
- 该测试文件与 `tests/rl/test_rollout_logic.py` 合计 `58 passed`；`git diff --check` 通过。

结论：目前没有发现会破坏 LMDeploy 后端的改动，并已用测试固定关键兼容边界。后续完整 GPU 两步结果见第 7 节。

## 6. LMDeploy/SGLang 速度 A/B 配置对齐（2026-08-06）

为避免把训练配置差异误判成后端速度差异，两个启动脚本和配置现在共享以下默认值，并可用同名环境变量一起覆盖：

- `TRAIN_BATCH_SIZE=16`、`PROMPT_REPEAT_K=8`、`MAX_CONCURRENT_SAMPLES=128`、`TOTAL_TRAIN_STEPS=2`；
- 训练 `TRAIN_EP_SIZE=4`；
- rollout `ROLLOUT_TENSOR_PARALLEL_SIZE=2`、`ROLLOUT_EXPERT_PARALLEL_SIZE=1`；
- `ROLLOUT_CONTEXT_LENGTH=69632`、每实例最大 batch 128；
- `SPECULATIVE_NUM_DRAFT_TOKENS=3`；
- routed experts、MTP 训练、swap optimizer 均开启；
- 两个脚本都固定使用 `pt2121_all_en_sglang` 环境。

Prefix cache 也统一为默认开启，并把 Mamba/SSM decode state checkpoint 间隔设为 256 tokens：SGLang 使用 Radix cache 的 `extra_buffer`（可切 `no_buffer`），LMDeploy 使用 BlockTrie prefix cache。LMDeploy 的 `LMDEPLOY_PREFIX_CACHE_STATE_BUDGET` 默认是 0，含义是允许 checkpoint 借用空闲 runtime state slot，而不是关闭 state cache；需要为高并发保留独立 checkpoint 容量时可增加该值。

仍然保留的差异属于后端实现本身：LMDeploy 使用 `qwen3_5_mtp`，SGLang 使用 EAGLE（另有 2 speculative steps 和 top-k 1）；tool/reasoning parser、scheduler、cache 数据结构和 overlap 实现也各自不同。它们应视为实际后端能力的一部分，不能做到参数名级完全相同。

静态验证结果：两个 shell 脚本均通过 `bash -n`，两个配置均能在目标 Conda 环境完整加载；解析后的 batch/repeat/concurrency、训练 EP、rollout TP/EP、context、draft tokens、prefix-cache state interval 和 routed-expert 开关一致。LMDeploy `PytorchEngineConfig` 也确认收到 `enable_prefix_caching=True`、`prefix_cache_decode_state_interval=256`。

## 7. LMDeploy 配置修复与两步实跑（2026-08-06）

### 7.1 修复内容

之前 `extra_rollout_config` 中下面几个 LMDeploy 参数只留在 server kwargs，未传入
`PytorchEngineConfig`，因此配置文件虽然写了，engine 实际并没有生效：

- `lmdeploy_enable_prefix_caching`
- `lmdeploy_prefix_cache_state_budget`
- `lmdeploy_prefix_cache_decode_state_interval`
- `lmdeploy_enable_metrics`

现已在 `xtuner/v1/rl/rollout/lmdeploy.py` 中把这些字段按 backend 路由到 engine config：

- PyTorch backend 接收全部四项；
- TurboMind backend 接收它支持的 `enable_prefix_caching` 和 `enable_metrics`，不会错误传入
  PyTorch 专属的 Mamba state 参数；
- 参数从 server kwargs 中移除，避免把 engine 参数错误传给 API server；
- speculative config 仍沿用原来的独立处理，不受影响。

增加了 PyTorch/TurboMind 两组路由回归测试。定向测试结果为 `58 passed`，
`git diff --check` 通过。

### 7.2 真实 engine 配置确认

完整 GPU 日志中实际构造出的 `PytorchEngineConfig` 为：

- `enable_prefix_caching=True`
- `prefix_cache_state_budget=0`
- `prefix_cache_decode_state_interval=256`
- `enable_metrics=True`
- `enable_return_routed_experts=True`

实际 speculative config 为 `qwen3_5_mtp`、3 draft tokens。主模型 `CacheConfig` 中存在
130 个 state cache slot，SSM state 同时包含 BF16 conv state 和 FP32 recurrent state。日志中另一个
`enable_prefix_caching=False` 的 `CacheConfig` 属于 MTP draft model，不是主模型配置失效。

### 7.3 两步跑通结果

运行参数与 SGLang 快速基线一致：batch 8、repeat 2、最大并发 16、训练 EP4、rollout TP2/EP1、
context 69632、MTP draft tokens 3、routed experts 开启。

LMDeploy 运行产物：

- 日志：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_lmdeploy/codex_2step_prefix_20260806_0230/training_log_080602.txt`
- rollout：`.../20260806023030/train_rollout/train_rollout_{1,2}.jsonl`
- checkpoint：`.../20260806023030/hf/hf-step-2`

进程退出码为 0，日志最终出现 `Train step 2/2`、checkpoint 保存成功和
`Ray shutdown successfully`。MTP rollout、MTP training、流式 SSE、routed experts、两次训练后的权重同步均已实际执行。

| 指标 | LMDeploy step 1 | SGLang step 1 | LMDeploy step 2 | SGLang step 2 |
|---|---:|---:|---:|---:|
| response mean tokens | 21,200.8 | 20,436.3 | 41,092.5 | 41,890.4 |
| rollout TGS | 366.4 | 584.1 | 362.1 | 468.6 |
| produce_batch | 260.60s | 174.77s | 320.79s | 522.26s |
| whole step | 363.18s | 242.12s | 453.32s | 649.10s |

粗略结论：

- 同样本长度接近的情况下，SGLang 的 rollout TGS 高 29%～59%，也就是 LMDeploy 低
  23%～37%；取最快的 step 1，SGLang 为 584.1 TGS，约是 LMDeploy 366.4 TGS 的 1.59 倍。
- LMDeploy step 1 的 `produce_batch` 比 SGLang 慢 49%；step 2 则比 SGLang 快 39%，因为
  SGLang 该步有 oversample 取消长尾，而 LMDeploy 没有等满 300 秒。
- 两步合计 `produce_batch`：LMDeploy 581.39s，SGLang 697.03s，LMDeploy 少 16.6%；两步
  whole-step 合计分别为 816.50s 和 891.22s，LMDeploy 少 8.4%。这个合计优势来自 SGLang
  step 2 的取消长尾，不能解读为 LMDeploy engine 吞吐更高。

### 7.4 LMDeploy prefix cache 命中

多轮请求主要集中到两个 LMDeploy 实例：step 1 结束时它们的累计命中率分别约 26.5% 和
27.1%。step 2 的有效样本几乎是单轮（tool turns mean 0.1875），加上大量新前缀后，两实例的
全程累计命中率降到约 19.8% 和 20.1%。其余两个实例以首轮/单轮请求为主，日志没有报告有效
prefix hit。

因此这次 RL 流量中可以认为“有多轮复用的实例约 20%～27% 命中”，但不能把这两个实例的
百分比直接当成四实例全局命中率。当前 LMDeploy 脚本没有在进程退出前保存四个 endpoint 的
Prometheus counter 快照；若要和 SGLang 的 65.31% 做严格全局比较，需要补一个退出前 metrics
snapshot，并按 cached prompt tokens / prompt tokens 加权。
