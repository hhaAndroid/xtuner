# Agentic RL：Mamba radix-cache A/B 记录

## A：no_buffer 稳定基线（已完成）

- 运行日期：2026-08-05
- 工作目录：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang/codex_2step_20260805_1149`
- 固化配置：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang/codex_2step_20260805_1149/config.py`
- 参数：`TRAIN_BATCH_SIZE=8`、`PROMPT_REPEAT_K=2`、`MAX_CONCURRENT_SAMPLES=16`、`TOTAL_TRAIN_STEPS=2`
- Mamba cache：`no_buffer`
- Overlap schedule：关闭
- 结果：2/2 train steps 完成，退出码 0，保存 `hf-step-2`
- Prefix cache：507,133 prompt tokens，331,231 cached tokens，命中率 65.31%
- Rollout throughput：step 1 为 413.01 token/s，step 2 为 414.63 token/s
- Step wall time：step 1 为 338.45 秒，step 2 为 715.10 秒
- 已知异常：step 2 有一个 oversample agent task 等待 300 秒后被强制取消；该等待不一定来自 Mamba cache，但计入 step wall time。

## B：extra_buffer + overlap

- 运行日期：2026-08-06
- 工作目录：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang/codex_2step_extra_buffer_20260806_0001`
- 固化配置：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang/codex_2step_extra_buffer_20260806_0001/config.py`
- 完整日志：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang/codex_2step_extra_buffer_20260806_0001/training_log_080601.txt`
- 参数：与 A 相同，即 `TRAIN_BATCH_SIZE=8`、`PROMPT_REPEAT_K=2`、`MAX_CONCURRENT_SAMPLES=16`、`TOTAL_TRAIN_STEPS=2`、`AUTO_RESUME=False`
- Mamba cache：`extra_buffer`
- Overlap schedule：开启
- 结果：2/2 train steps 完成，退出码 0，保存 `hf-step-2`
- 运行时确认：4 个实例均使用 `MambaRadixCache`；Radix cache、EAGLE MTP rollout、routed expert 返回均开启。
- Rollout throughput：step 1 为 584.11 token/s，step 2 为 468.56 token/s。
- Step wall time：step 1 为 242.12 秒，step 2 为 649.10 秒。
- MTP 训练：step 1 的 `reduced_mtp_loss=0.9061`、`grad_norm=0.0899`；step 2 仍计算出 `reduced_mtp_loss=0.8602/0.9134`，但该批次所有 advantage 为 0，因此 LLM loss 和总 grad norm 都为 0，没有产生有效参数更新。这是采样奖励退化，不是 extra-buffer 启动或数值错误。

## 单次 A/B 结果

| 指标 | A：no_buffer | B：extra_buffer | 单次变化 |
| --- | ---: | ---: | ---: |
| step 1 rollout token/s | 413.01 | 584.11 | +41.4% |
| step 2 rollout token/s | 414.63 | 468.56 | +13.0% |
| step 1 wall time | 338.45s | 242.12s | -28.5% |
| step 2 wall time | 715.10s | 649.10s | -9.2% |
| step 1 produce_batch | 273.24s | 174.77s | -36.0% |
| step 2 produce_batch | 589.14s | 522.26s | -11.4% |

这个结果支持“当前 16 并发负载下，允许 overlap 的 `extra_buffer` 更快”，但只有各一次运行，且两次生成的平均 response 长度并不完全相同，不能当作稳定 benchmark 结论。

## Prefix/Mamba cache 观察

- B 的 step 1 四实例完整快照：339,852 prompt tokens，152,512 cached tokens，命中率 44.88%。
- B 的 step 2 收尾期间只成功直连抓到 3/4 个实例：339,646 prompt tokens，160,128 cached tokens，部分命中率 47.15%。`:25025` 在该暂停窗口直连 metrics 返回 connection-refused，因此不能把 47.15% 当成完整两步命中率。
- A 的 65.31% 是 step 2 后的四实例完整累计值，与 B 的 step 1 或 3/4 部分快照不是严格同口径；当前没有证据说明 `extra_buffer` 会提高 prefix-cache 命中率。B 的已观测命中反而较低，后续应增加自动的四端点、同阶段快照再做判断。
- B 在活跃推理时的 Mamba usage 约为 2%--6%，没有接近耗尽。step 1 flush 后四实例均为 `used=0`、`evictable=0`、`available=444/474`，说明两步之间没有 Mamba state slot 泄漏。
- `extra_buffer` 为 EAGLE 请求保留 ping-pong 状态：日志中单个 active request 通常占 4 个 Mamba slot，A 的 `no_buffer` 通常占 2 个。因此 B 的 `max_running_requests` 从 A 的 126 降到 88--94。当前最大并发只有 16，不构成瓶颈；高并发场景需要重新评估容量。
- 原始快照：`work_dirs_rl/agentic_rl_qwen3p5vl_mtp_ep_code_sglang/codex_2step_extra_buffer_20260806_0001/prefix_cache_metrics_snapshot.txt`。

## 仍未解决

- Step 2 达到 8 个目标组后，仍有 3 个 oversample agent task 没有响应 abort，等待满 300 秒后才被强制取消。A 也有同类问题（1 个 task），所以尚不能归因于 `extra_buffer`，但它是当前 wall time 的最大噪声和浪费。
- Step 2 暂停期间对 `:25025/metrics` 的外部直连曾返回 connection-refused；训练控制器随后报告没有 failed rollout worker，且该实例参与了最终 cache flush，整个任务也退出码 0。现有证据不足以判断是 metrics 可达性的瞬态问题，还是 pause/abort 链路的局部阻塞。
- 单次样本的 reward/response 长度波动较大；若要决定默认策略，建议至少再做 3 次固定数据顺序的 A/B，并让脚本在每个 rollout pause 后自动保存四端点 metrics。

## 面向 RL 的两种模式比较

`no_buffer` 和 `extra_buffer` 都是在 Radix prefix cache 已开启的前提下，选择如何保存和复用 Mamba recurrent state；它们不是“开/关 prefix cache”的区别。现有流式返回、EAGLE MTP 和 routed expert 路径在两种模式下都可工作，策略切换本身也不会把流式推理改成非流式。

| 维度 | `no_buffer` | `extra_buffer` |
| --- | --- | --- |
| Mamba state 保存方式 | 从当前 live state 复制/转交给 radix cache | 使用额外的 ping-pong tracking buffer 定期保存快照 |
| Overlap schedule | 必须关闭，避免 cache copy 与 forward stream 竞争 | 可以开启，快照与正在执行的 forward 解耦 |
| Rollout 吞吐 | 少了 overlap，长解码和并发请求下通常较慢 | 更容易隐藏调度和状态管理开销；本次两步分别快 41.4% 和 13.0% |
| Mamba slot/显存压力 | 较低，可容纳更多 active requests | 较高；本次 EAGLE 配置中每个 active request 通常从 2 个 slot 增至 4 个 |
| 本次实例请求容量 | `max_running_requests=126` | `max_running_requests=88--94` |
| Prefix state 粒度 | 通常可保存更接近当前长度的 state，粒度更细 | 按 tracking 边界保存，本配置默认间隔为 256 tokens，最后一个边界后的短 suffix 可能不能立即复用 |
| 实现与排障 | 状态路径较简单，适合作为保守 baseline | 多一组 buffer 生命周期，容量和释放行为需要监控 |
| EAGLE MTP | 本次已跑通 | 本次已跑通；应使用 `extra_buffer`，不要改成不支持 speculative decoding 的 `extra_buffer_lazy` |

### RL 场景下何时使用 `extra_buffer`

建议把 `extra_buffer` 作为当前配置的默认值，尤其适用于：

- rollout 是异步并发的，包含长 response、多轮 tool call 或 agent trajectory，GPU decode/调度吞吐是主要瓶颈；
- 使用 EAGLE MTP，希望在 speculative decoding 的同时保留 overlap schedule；
- 并发量明显低于实例的 `max_running_requests`，Mamba slot 还有充足余量；
- prefix reuse 很重要，但不要求任意 token 位置都成为最细粒度的 Mamba state 快照。

当前 RL 任务属于这一类：每实例的 Mamba usage 实测只有约 2%--6%，全局最大 rollout 并发为 16，远低于 88--94 的实例容量，而两步 rollout 都获得了实际加速。因此目前优先推荐 `extra_buffer`。

### RL 场景下何时使用 `no_buffer`

以下情况更适合切回 `no_buffer`：

- 准备显著提高 `MAX_CONCURRENT_SAMPLES`，active requests 已逼近 `max_running_requests`；
- GPU/Mamba state 内存紧张，出现 slot 分配失败、排队持续增加，或希望用同样资源容纳更多并发轨迹；
- rollout 并发很低、生成很短，overlap 带来的收益不足以抵消额外 buffer；
- 正在排查 abort、cache flush、state 生命周期或后端兼容问题，需要路径更简单的稳定基线；
- 工作负载特别依赖最新 suffix 的细粒度复用，并且实测证明 256-token tracking 边界影响了命中收益。

最后一项只是实现上的潜在倾向，不能用当前 65.31% 和 44.88% 直接证明：两次 metrics 抓取阶段、实例数量和生成内容并不一致。

### RL 中评估时要特别注意

- 每个训练 step 的权重同步/cache flush 会清掉上一步的 cache，因此主要收益来自同一 rollout step 内的重复 system prompt、`PROMPT_REPEAT_K`、多轮 tool history 和相似前缀，而不是跨训练 step 复用。
- Prefix 命中率通常更受请求路由、同前缀是否落到同一实例、并发到达顺序和生成长度影响。选择 Mamba buffer 策略时，应同时看吞吐和容量，不能只看一次 prefix-cache 百分比。
- 建议同时监控 `prompt_tokens_total/cached_tokens_total`、`mamba_used_tokens`、`mamba_available_tokens`、`mamba_evictable_tokens`、排队请求数和 rollout token/s。
- 可把 Mamba usage 持续达到约 70%--80% 作为需要降并发或 A/B 切回 `no_buffer` 的经验预警线；这只是运维阈值，不是 SGLang 的硬性保证。
- 两种模式都出现过 oversample task 等待 300 秒的问题，现阶段不要把该尾延迟作为 `extra_buffer` 或 `no_buffer` 的固有优缺点。

## 当前结论

`extra_buffer + overlap` 已经通过真实的两步端到端流程，且这次两步 rollout 都快于 `no_buffer`。对当前 16 并发的 agentic RL + MTP 任务，推荐继续使用 `extra_buffer`；当并发扩大到 Mamba slot 容量成为瓶颈、显存紧张，或需要保守排障基线时，再切换到 `no_buffer`。Prefix cache 命中率暂时不能证明 `extra_buffer` 优于 baseline，oversample 的 300 秒取消尾延迟也仍需单独处理。

## 配置切换方法

配置只接受 `extra_buffer` 和 `no_buffer` 两个值，并自动选择兼容的 overlap 设置：

```bash
# 默认：extra_buffer，overlap schedule 开启
SGLANG_MAMBA_RADIX_CACHE_STRATEGY=extra_buffer \
  bash workspace/new_test_sh/agentic_rl_test_sglang.sh

# baseline：no_buffer，overlap schedule 自动关闭
SGLANG_MAMBA_RADIX_CACHE_STRATEGY=no_buffer \
  bash workspace/new_test_sh/agentic_rl_test_sglang.sh
```

未设置该环境变量时默认使用 `extra_buffer`。传入其他值会在加载配置时直接报错，避免静默落入不受支持的组合。
