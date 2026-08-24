# Abort 请求链路与慢退出问题分析

本文记录 XTuner rollout 在 pause / abort 场景下的慢退出问题分析。关注点不是单纯让 `/abort_request` 更快返回，而是定位 pause 之后最后少量 pending rollout task 卡在哪一层，并判断已有修复是否真正覆盖对应问题。

相关并发背景可参考：[rollout_concurrency_gsm8k.md](./rollout_concurrency_gsm8k.md)。

## 背景链路

共卡方案中，一轮训练前会通过 `AgentLoopManager.produce_batch()` 生成 rollout 数据。一次调用内大体包含三段：

1. producer schedule rollout group，并通过 agent loop 调 rollout controller。
2. rollout controller 将请求路由到 rollout worker。
3. rollout worker 通过 httpx 向 LMDeploy 发 `/generate`。

当权重同步或主动 pause 发生时，XTuner 需要停止继续调度新 rollout，并通知 rollout worker abort。已经 schedule 出去的 rollout task 仍需要被回收；正常返回的 completed / aborted group 会写入 replay buffer，超过 cleanup timeout 后才会强制 cancel 剩余 task。

因此慢退出要分层判断：

- producer 是否还在 pause 后继续 schedule 新 group。
- 已 schedule task 是否卡在 XTuner / Ray 层。
- worker 内部 HTTP 请求是否卡在 httpx 连接池。
- 请求是否已经到达 LMDeploy API / engine waiting / running 队列。
- `/abort_request abort_all=True` 是否只清理了 LMDeploy 已知请求，而清不到 XTuner 侧尚未发送的旧请求。

## Item 1: HTTP 连接池隐藏排队

### 问题描述

早期配置中，单个 `RolloutWorker` 的 httpx 连接池上限由下面两个配置决定：

```text
current_http_concurrency =
  ceil(rollout_max_batch_size_per_instance * allow_over_concurrency_ratio)
```

如果 producer 在一轮 rollout 中实际创建的 `/generate` HTTP 请求数，高于所有 active rollout worker 可承载的 httpx 连接数，就会出现隐式排队：

```text
已创建的 worker generate task
├─ 一部分拿到 httpx connection，已经发到 LMDeploy
└─ 一部分卡在 XTuner worker 的 httpx client 连接池，LMDeploy 暂时看不到
```

会出现这个的原因是因为假设整个训练需要 512条样本，总共给所有 worker 发送这么多请求，假设是完全均分模式，每张卡需要负责 512/8 条样本。但是实际上发送给 lmdeploy 的最大并发请求数是current_http_concurrency 数确定的。和这个 global batch size 其实没有关系，rollout_max_batch_size_per_instance 是独立配置的。因为即使是均分，也可能出现发给worker 的请求远远大于current_http_concurrency ，从而会在 xtuner worker 的 httpx 连接池中排队，lmdeploy 并没有接受到这部分请求。这还是在均分模式下的问题，如果实际上不均分，则问题可能更严重。

此时发送 `/abort_request abort_all=True` 时，abort 请求本身可以通过独立 HTTP client 发出去(因为和前面说的 httpx 连接池不是同一个)，但 LMDeploy 只能清理当时已经到达 LMDeploy 的请求。仍在 XTuner httpx 连接池里排队的旧 `/generate` 请求不在 LMDeploy 管理范围内。

后续一旦已有连接释放，httpx 排队请求可能继续发送到 LMDeploy，于是日志上表现为：

```text
abort_request 已经发送
LMDeploy 之后仍然看到新的 /generate 到达
```

这不必然说明 producer 在 pause 之后继续生产新业务请求；也可能是 pause 前已经创建的旧 task 在 HTTP 层晚到 LMDeploy。

### 修复方案

在 RL trainer 初始化阶段加入 HTTP concurrency sanity check，按当前训练配置估算本轮 rollout 可能产生的 HTTP 请求数：

```text
scheduled_http_requests =
  sum((task_batch_size + ceil(task_batch_size * over_sample_threshold)) * prompt_repeat_k)
```

再按 active rollout worker 数估算单 worker 需要的 HTTP 并发：

```text
required_http_concurrency =
  ceil(scheduled_http_requests / active_rollout_worker_count) 假设均匀模式
```

如果当前配置不足：

```text
current_http_concurrency =
  ceil(rollout_max_batch_size_per_instance * allow_over_concurrency_ratio)
```

则自动调大：

```text
allow_over_concurrency_ratio =
  required_http_concurrency / rollout_max_batch_size_per_instance
```

同时打印 warning，记录旧值、新值、required concurrency、worker 数，以及各 task 的 batch / repeat / oversample 配置，方便复盘为什么配置被放大。

### 合理性判断

这个修复是合理的短期保护。它解决的是“producer 已经创建的请求数超过 httpx pool，导致部分请求藏在 XTuner 侧，LMDeploy abort 看不到”的问题。

修复后，在请求大体均匀分布到 active rollout workers 的前提下，每个 worker 的 httpx 连接池至少能覆盖本轮预估并发。这样旧请求更早进入 LMDeploy，`abort_all=True` 能看到并清理更多请求，降低 abort 后又有旧 `/generate` 晚到的概率。

但它不是完整的根因修复：

- 它按平均值 `scheduled_http_requests / active_rollout_worker_count` 估算。如果真实路由严重不均，例如大量请求因 session sticky 集中到同一个 worker，单个 worker 仍可能超过该估算值。
- 它没有改变 producer 的 schedule 行为，也没有建立 per-worker 显式 in-flight 限流。
- 它没有让“尚未真正发送到 LMDeploy 的请求”在 abort flag 置位后直接取消；只是尽量减少这类请求留在 httpx pool 的机会。
- 它会把隐式排队从 XTuner/httpx 层转移到 LMDeploy 可见的 API / engine waiting 层。这样更利于 abort，但可能增加 LMDeploy waiting 队列和请求元数据压力。

### 风险评估

该方案不一定导致 LMDeploy OOM，因为真正的大头通常由 engine running batch、prefill、KV cache、context length 等控制；LMDeploy waiting 队列里的请求主要占用请求对象、tokens、payload 和元数据。

但如果 `allow_over_concurrency_ratio` 被自动调得过大，风险会升高：

- 更多请求瞬时进入 LMDeploy。
- API waiting / engine waiting 队列变长。
- 超长 prompt、多模态 payload、routed experts 返回等重请求会放大内存压力。
- 如果 LMDeploy 对 waiting 请求也有较重的预处理或缓存，可能出现额外内存和延迟问题。

因此这个修复更适合作为 sanity check 和止血措施。更长期的方向应是：

- producer schedule 上限与 rollout backend capacity 对齐。
- router 按 worker 当前负载和容量分配请求。
- worker 侧建立显式 per-worker in-flight semaphore，而不是依赖 httpx 连接池隐式排队。
- abort 后，对尚未发送到 LMDeploy 的本地 pending 请求直接返回 `ABORTED`，避免继续晚到。

### 当前结论

“HTTP 连接池隐藏排队问题已经修复”这个说法需要限定语义：

```text
已降低均匀分发场景下 XTuner/httpx 连接池成为隐式队列的概率。
```

它不能等价为：

```text
已彻底消除 pause 后旧请求晚到 LMDeploy 的所有可能。
```

如果后续仍看到 abort 后有 `/generate` 到达，需要继续区分这些请求是：

1. pause 前已创建、卡在 XTuner/httpx 后晚到；
2. pause 后 producer 仍然 schedule 的新请求；
3. router/session sticky 导致单 worker 超过平均估算；
4. LMDeploy 内部 abort 后仍残留或重新接收的请求。

## Item 2: abort 后仍有少量 client.send 请求到达 LMDeploy

### 问题描述

即使 Item 1 的 HTTP 连接池容量已经调大，仍不能保证第一次 `/abort_request` 覆盖所有 pause 前创建的旧请求。

原因是 `/generate` 和 `/abort_request` 是两类独立 HTTP 请求，二者到达 LMDeploy 的顺序存在竞态窗口：

```text
T0: producer 已经创建 rollout task
T1: rollout worker 中的 self.client.send(/generate) 已经开始或即将开始发送
T2: pause 触发，worker 设置 abort flag，并发送 /abort_request
T3: LMDeploy 先收到 /abort_request
T4: 少量旧 /generate 后到达 LMDeploy，并创建新的 session / request
```

这批 `/generate` 请求不是 pause 后 producer 新 schedule 的业务请求，而是 pause 前已经创建的旧 task。它们没有卡在 httpx 连接池容量不足上，而是处在 HTTP 发送或网络到达 LMDeploy 的在途窗口中。

第一次 abort 对这类请求天然覆盖不好：LMDeploy 收到第一次 `/abort_request` 时，这些请求还没有到达，也就没有对应 session 可以被清理。

### 修复方案

在 `pause_produce()` 等待 pending rollout task 回收期间，周期性补发 pause / abort：

```text
while pending tasks not drained:
  periodically call agent_loop.pause()
  agent_loop.pause() -> rollout_ctl.pause_generation()
  rollout_ctl.pause_generation() -> worker.pause_generation()
  worker.pause_generation() -> POST /abort_request {"abort_all": true}
```

这样第一次 abort 之后才到达 LMDeploy 的旧 `/generate`，有机会被后续 periodic abort 覆盖。

当前实现中，`AsyncProduceStrategy` 设置：

```text
PERIODIC_ABORT_INTERVAL_S = 5.0
PENDING_TASK_COLLECT_TIMEOUT_S = 60.0
```

因此在 pending 回收窗口内，理论上会持续补发 abort，直到 pending 清空或 cleanup timeout 后强制 cancel。

### 合理性判断

这个修复是合理的，并且比 Item 1 更直接针对“abort 和 /generate 到达顺序竞态”。

它承认一个事实：只发一次 abort 不具备“未来请求也自动被取消”的语义。对于 abort 后才到达 LMDeploy 的旧请求，必须要么：

- XTuner 在本地阻止它继续发送；
- 要么 LMDeploy 能识别全局 abort generation / epoch，拒绝迟到请求；
- 要么 XTuner 在等待 pending 期间继续补发 abort，把迟到请求再清掉。

当前方案选择第三种，是低侵入、容易落地的补偿机制。它不要求 LMDeploy 改协议，也不要求立即重构 worker 侧发送队列。

### 局限

这个方案仍然是补偿式修复，不是严格根因修复：

- periodic abort 存在时间粒度。例如 interval 是 5 秒，某个旧 `/generate` 在两次 abort 之间到达，可能仍会运行一小段时间。
- 如果 LMDeploy 内部请求创建后很快进入不可中断阶段，后续 abort 仍可能不够及时。
- 如果 pending task 卡住的原因不是 LMDeploy session 残留，而是 worker / Ray / judger / response parsing / object ref drain，补发 abort 只能覆盖其中一部分。
- periodic abort 依赖 `pause_produce()` 仍在等待 pending。如果 pending 已经被错误地认为清空，后续不会继续补发。
- 在 128 卡场景中，fanout 本身、Ray 调度、LMDeploy 接收 abort 的延迟都会放大，需要单独验证。

### 风险评估

周期性补发 abort 的副作用相对可控：

- `/abort_request abort_all=True` 应该是幂等语义；重复发送通常不会改变正确性。
- 额外 HTTP 请求量很小，相比 rollout `/generate` 负载可以忽略。
- 主要风险是日志噪声和对 LMDeploy control path 的轻微压力。

但需要注意，如果某些 backend 的 abort 实现不是严格幂等，或者 abort 会触发较重的锁/全局扫描，5 秒一次的全量 abort 在大规模场景下仍应观察控制面延迟。

### 当前结论

这个发现成立。它和 Item 1 是两个不同层次的问题：

```text
Item 1: 请求还在 XTuner/httpx pool，LMDeploy 看不到。
Item 2: 请求已经在发送路上，但到达 LMDeploy 晚于第一次 abort。
```

periodic abort 能覆盖一部分“第一次 abort 后才创建 session”的残留请求。8/16 卡实验里有效，说明方向是对的。

但对于 128 卡 forced cleanup 场景，不能只凭 8/16 卡结果断言已完全修复。还需要用日志确认：

- 第一次 abort 后是否仍有 `/generate` 到达；
- 后续 periodic abort 是否覆盖这些迟到 session；
- pending task 最终是正常返回 `ABORTED`，还是等到 cleanup timeout 后被 force cancel；
- force cancel 剩余数量是否已经降到可接受范围。

因此当前判断是：

```text
periodic abort 是合理且必要的补偿修复；
它能降低迟到 /generate 残留 session 的概率；
128 卡 forced cleanup 是否完全修复仍需要实测确认。
```

## Item 3: abort 后 completed response 进入 judger

### 问题描述

worker 观察到 `receive_abort_request` 后，仍可能收到之前 `/generate` 的 HTTP response。这个 response 从 HTTP 和 LMDeploy finish reason 看可能是正常 completed response，并且已经带有大量生成 token。

这个现象并不矛盾。当前 `_safe_post_request()` 的逻辑是：

```text
1. 先创建 self.client.send(/generate) task。
2. 同时等待 send_task 和 abort_task。
3. 如果 abort_task 先完成，不是立刻丢弃 send_task；
   而是在 abort_timeout 内再等一下 send_task。
4. 如果 send_task 在 abort_timeout 内返回，仍会把 HTTP response 交给后续解析。
```

因此 abort flag 置位之后返回的 response 可能确实已经生成了很多 token。它不是“没开始推理就被 abort”的样本。

如果这类 response 继续按 `COMPLETED` 状态流下去，`SingleTurnAgentLoop.generate_sample()` 会认为它是正常完成样本，并触发 judger。批量打分路径里，如果 group 内没有样本被标记成 `ABORTED`，也会继续调用 batch judger。

### 修复方案

当前 worker 的处理顺序是合理的：

```text
http_result.response 返回
  -> _safe_handle_response() 正常解析 response / token / logprob / routed experts
  -> 再检查 receive_abort_request
  -> 如果 abort flag 已设置，将 finish_reason/status 覆盖为 abort/ABORTED
```

也就是说，修复不是在 `_safe_post_request()` 返回后直接丢弃 response，而是先保留 LMDeploy 已经返回的有效内容，再把语义状态改成 `ABORTED`。

这一点很重要。直接丢弃 response 会丢掉已经生成的 token、logprob 和 routed experts，也会让后续无法判断这些样本到底是未开始、已部分生成，还是已生成完成但晚于 abort 返回。

### 源码判断

当前分支中，`RolloutWorker.generate()` 在拿到 HTTP response 后先调用 `_safe_handle_response()`，随后检查 `receive_abort_request`：

```text
rollout_state = await self._safe_handle_response(rollout_state, http_result.response)
if self.receive_abort_request.is_set():
    rollout_state.finish_reason = "abort"
    rollout_state.status = Status.ABORTED
    return rollout_state
```

`_safe_handle_response()` 会先解析并写入：

- `response`
- `response_ids`
- `logprobs`
- `routed_experts`
- `finish_reason`
- `status`

随后外层再把 abort 后晚到的 completed response 改成 `ABORTED`。这与“保留已生成 token，但状态归为 ABORTED”的结论一致。

judger 侧也能对上这个语义。`SingleTurnAgentLoop.generate_sample()` 中，只有 `rollout_state.status == Status.COMPLETED` 才会进入 per-sample judger；否则直接返回。batch judge 路径中，如果 group 内存在 `Status.ABORTED` 样本，也会跳过 batch judger。

### 合理性判断

这个发现成立，修复也合理。

它解决的是第三层问题：即使请求已经到达 LMDeploy，并且最终返回了一个 HTTP completed response，只要这个 response 是在 worker 已观察到 abort 之后才返回，就不应该再作为正常 completed sample 进入 judger。

更准确地说，这个修复改变的不是 LMDeploy 是否继续生成，而是 XTuner 对晚到 response 的归类：

```text
修复前:
abort 后晚到 response -> COMPLETED -> 可能进入 judger

修复后:
abort 后晚到 response -> 解析并保留内容 -> ABORTED -> 跳过 judger
```

因此它不会让 `/abort_request` 更快，也不会消除所有 pending task，但能避免 pause 后返回的旧样本污染 reward / judge 流程，并减少 judger 继续处理旧样本带来的尾部等待。

### 局限

这个方案仍有几个需要确认的边界：

- 它依赖所有 agent loop / judger 入口都尊重 `Status.ABORTED`。当前 `SingleTurnAgentLoop` 是尊重的，但其他自定义 agent loop 也需要保持同样约定。
- batch judge 当前是“group 内只要有一个 `ABORTED` 就跳过整个 group 打分”。这是保守策略，可以避免混入 pause 后样本，但会牺牲同组里其他 completed sample 的打分。
- 如果 `_safe_handle_response()` 在解析阶段因为 malformed response 抛异常，外层这段 abort 覆盖逻辑不会执行；这种情况会走失败/重试/异常路径，而不是“保留 token 后标 ABORTED”。
- replay buffer 和后续统计逻辑需要接受“`ABORTED` 但携带非空 `response_ids`”这种状态。`aborted_zero_token_count=0` 正说明这批样本不是空 abort，而是已有生成内容的 abort。

### 当前结论

当前判断是：

```text
abort 后 completed response 进入 judger 这个问题确实存在；
先解析 response 再覆盖为 ABORTED 是正确修复；
16 卡实验中 620 个 abort 后晚到 response 被归为 ABORTED，
且 aborted_zero_token_count=0，能支持这个判断。
```

它和前两项的关系是：

```text
Item 1: 请求卡在 XTuner/httpx pool，LMDeploy 第一次 abort 看不到。
Item 2: 请求已在 HTTP 在途窗口，晚于第一次 abort 到达 LMDeploy。
Item 3: 请求已经返回 response，但返回时 worker 已经观察到 abort，不能再进 judger。
```

## Item 4: mixed completed / aborted group 被 batch judge 拖住

### 问题描述

`enable_batch_judge=True` 时，`generate_group()` 不是每个 sample 单独 judge，而是在 group 内所有 sample 都完成 generate 后，再统一调用 batch judger。

这会引出一个 group 级别的问题：

```text
sample A -> COMPLETED
sample B -> ABORTED
sample C -> COMPLETED
...
generate_group() gather 返回 group_samples
```

对单个 sample 来说，`generate_sample()` 中的 early return 可以保证 `ABORTED` sample 不进入 per-sample judge。但开启 batch judge 后，真正的 judge 入口在 `generate_group()` gather 之后。如果这里只判断“单个 sample 是否自己 early return”，是不够的；mixed group 仍可能整体进入 batch judge。

这类 group 已经越过了 LMDeploy / HTTP / rollout worker / controller-agent loop 边界，说明卡点不再是推理 backend，而是回到 agent loop 后被 batch judger 继续拖住。

### 修复方案

在 batch judge 路径中，`generate_group()` 先 gather group 内所有 sample，然后根据 group 状态决定是否 judge：

```text
group_samples = await gather(generate_sample(...))

if enable_batch_judge:
  if group 内不存在 ABORTED sample:
    run batch judger
  else:
    skip batch judger and return group_samples
```

当前分支源码中的逻辑是：

```text
if self.judger is not None and self.enable_batch_judge:
    if not any(sample.status == Status.ABORTED for sample in group_samples):
        group_samples = await self.run_judger(group_samples)
return group_samples
```

因此只要 group 内存在 `Status.ABORTED`，就会跳过 batch judge，直接返回给 producer 回收。

### 合理性判断

这个发现成立，修复也是必要的。

它和 Item 3 的区别是：

```text
Item 3: 单个 response 晚于 abort 返回，不能把它标成 COMPLETED 进入 judge。
Item 4: group 内已经有 ABORTED sample，整个 group 不应再进入 batch judge。
```

对于 pause 后回收的 group，只要包含 aborted sample，它就不再是一个完整、正常完成的 rollout group。继续进入 batch judge 会带来两个问题：

- 语义上不合理：judger 会处理 pause 后应被回收的样本。
- 性能上拖尾：最后少量 pending group 已经完成 generate，却被 batch judge 的耗时继续阻塞，导致 producer 等不到 group 返回。

因此“mixed / aborted group 直接跳过 batch judge”是符合 pause 语义的。16 卡实验中 124 个 mixed / aborted group 跳过 batch judge，且 gather 到返回最大约 0.10s，说明该修复确实把这部分 tail latency 从 judger 层移除了。

### 局限

当前实现是“只要存在 `Status.ABORTED` 就跳过 batch judge”，而不是严格写成“只有全部 sample 都是 `Status.COMPLETED` 才进入 batch judge”。

两者在 abort 场景下等价性足够好，但语义上仍有差异：

```text
当前实现:
没有 ABORTED -> 进入 batch judge

更严格的实现:
所有 sample 都是 COMPLETED -> 才进入 batch judge
```

如果未来 group 内出现 `FAILED`、`TRUNCATED` 或其他非 `COMPLETED` 状态，但没有 `ABORTED`，当前逻辑仍可能进入 batch judge。这不一定是本次 abort 慢退出问题的根因，但从 batch judge 的输入契约看，更稳妥的条件应是 all completed。

另外，跳过整个 group 的 batch judge 是保守策略。它避免了 aborted sample 污染 reward，但也意味着 mixed group 中其他 completed sample 不会被 judge。对于 pause / cleanup 场景这是合理取舍；对于非 pause 的普通失败场景，则需要根据 replay buffer 和训练样本使用策略再判断。

### 当前结论

当前判断是：

```text
mixed completed / aborted group 被 batch judge 拖住这个问题成立；
跳过包含 ABORTED sample 的 batch judge 是正确修复；
它定位的是 generate 已完成后、producer 回收前的 judger 层卡点。
```

建议后续把判断条件从“没有 ABORTED”进一步收紧为“全部 COMPLETED”，或者至少确认其他非完成状态不会进入 batch judge。

## Item 5: all completed group 在 pause 后仍可能卡 judger

### 问题描述

Item 4 解决的是 mixed group：

```text
group 内存在 ABORTED sample -> 跳过 batch judge
```

但它覆盖不了另一类情况：

```text
group 内所有 sample 都已经 COMPLETED
generate_group() 已进入 batch judge
此时 pause / abort 发生
```

这类 group 没有 `ABORTED` sample，因此不会触发 Item 4 的 skip 逻辑。如果 batch judge 本身很慢，producer 虽然已经通知 rollout controller abort，LMDeploy / HTTP / worker / controller 边界也都返回了，仍然要等这个 `generate_group()` 从 judger 返回。

因此如果日志显示最后少量 pending group 已经越过推理链路，却迟迟没有回到 producer，就需要继续看 agent loop 内部是否卡在 judger。

### 结论判断

这个问题在设计上成立。

pause 不能只通知 rollout controller，因为 rollout controller 只负责推理侧 `/generate` 的 abort。已经离开 rollout controller、进入 judger 的 group，不再受 LMDeploy `/abort_request` 影响。

如果目标是“pause 后尽快回收所有已 schedule group”，那么 agent loop 层必须能让 in-flight judger 停下来，或者至少给 judger 一个明确的 cancel timeout，避免 producer 被少量长耗时 judge 拖住。

### 当前分支源码核对

当前分支里，producer 的 `pause_produce()` 已经不是直接只 pause rollout controller，而是周期性调用：

```text
ctx.agent_loop.pause()
```

`AgentLoopActor.pause()` 没有走 `generate` concurrency group，是 actor 默认方法；这符合“不新增 control concurrency group，pause 走 actor 默认方法”的方向。

`SingleTurnAgentLoop.pause()` 的实际实现是：

```text
self._pause_event.set()
await super().pause()
self._pause_event.clear()
```

其中 `super().pause()` 会调用 `rollout_ctl.pause_generation()`。judger 侧不是通过 `Judger.pause()` 接口直接暂停，而是 `SingleTurnAgentLoop.run_judger()` 创建两个 task：

```text
judge_task = self.judger.judge(...)
pause_task = self._pause_event.wait()
```

如果 pause 先到，`run_judger()` 会再等待 `DEFAULT_JUDGER_CANCEL_TIMEOUT_S = 5.0` 秒。如果 judge 仍未返回，则 cancel `judge_task`，并把 sample / group 标成：

```text
status = Status.ABORTED
finish_reason = "abort"
reward = None
```

所以当前分支已经有“让 SingleTurnAgentLoop 内 in-flight judger 受 pause 影响”的机制。

### 与描述中修复方案的差异

你描述的修复方案包含：

- `Judger.pause()`
- `NativeJudger.pause()`
- `RemoteJudger.pause()`
- `JudgerPool.pause()`
- `ComposedJudger.pause()`
- `XTUNER_JUDGER_CANCEL_TIMEOUT_S`

但按当前分支源码看，这些接口没有实际出现。`Judger` base class 仍只有 `judge()`；`NativeJudger`、`RemoteJudger`、`JudgerPool`、`ComposedJudger` 都没有 `pause()`；cancel timeout 也是代码常量 `DEFAULT_JUDGER_CANCEL_TIMEOUT_S = 5.0`，不是环境变量。

当前实现更接近 PR 里较低侵入的方案：

```text
不在 Judger base class 定义通用 pause；
只在 SingleTurnAgentLoop 中把 judger 调用包装成可被 pause_event 打断的 task。
```

这能覆盖 `SingleTurnAgentLoop` 的单样本 judge 和 batch judge，但不等价于“所有 Judger 类型自己都能感知 pause 并清理内部资源”。

### 合理性判断

从修复方向看，这是合理的。它解决的是第四层之后的 tail：

```text
LMDeploy / HTTP / worker / controller 已返回
group 已进入 agent loop judger
producer 仍等待 generate_group() 返回
```

给 judger 引入 pause / cancel 语义是必要的，否则 rollout abort 再及时，也无法处理中途已经进入 reward 判断的 all completed group。

但是要注意当前实现的语义不是“pause 后所有已进入 judger 的 group 一律变成 ABORTED”。当前 `run_judger()` 在 pause 到达后，会给 judge task 一个 5 秒宽限：

```text
pause 到达
  -> 如果 judge 在 5 秒内正常返回：保留 judged result
  -> 如果 judge 超过 5 秒未返回：cancel 并标 ABORTED
```

因此它更准确地说是“限制 judger 拖尾时间”，而不是严格“pause 后停止所有 judge 并按 aborted group 回收”。

如果训练语义要求 pause 之后返回的 all completed group 也不能进入 reward 统计和训练样本，那么当前实现还不够严格；需要在 pause 被观察到后无条件标 `ABORTED`，最多只把 cancel timeout 用于 drain / cleanup，而不是允许 judge 在 timeout 内正常完成后继续作为 completed sample 返回。

如果训练语义允许“pause 之前已经进入 judger、且很快 judge 完成”的 group 作为 completed group 回收，那么当前实现是一个更温和的折中，可以减少无谓丢样本，同时把长尾 judge 限制在 5 秒左右。

### 风险与待验证点

这项修复还不能仅靠源码判断完全确认，需要实验日志验证：

- pause 后剩余 pending group 是否还卡在 `run_judger()`。
- judge cancel 后，group 是否按 `ABORTED` 返回给 producer，而不是等 cleanup timeout 被 force cancel。
- 被 cancel 的 judger 是否真的停止了外部工作。对本地 async HTTP judge，取消 asyncio task 通常能中断等待；但如果是 Ray remote judge 或同步阻塞 reward handler，仅取消外层 task 未必能终止远端/阻塞中的实际工作。
- 当前没有 `Judger.pause()` 协议，因此 `RemoteJudger`、`JudgerPool`、`ComposedJudger` 的内部 active judge ref / branch judge 是否被真正取消，需要单独确认。
- pause_event 是瞬时事件，`pause()` finally 会 clear。已经在 `run_judger()` 里的任务能观察到；但如果某个 group 在 pause_event clear 之后才进入 `run_judger()`，就依赖后续 periodic pause 再次打断。

### 当前结论

当前判断是：

```text
all completed group 在 pause 后卡 judger 这个问题在设计上成立；
让 AgentLoop.pause() 覆盖 in-flight judger 是正确方向；
当前分支实现能限制 SingleTurnAgentLoop judger 拖尾，
但不是你描述的完整 Judger.pause 协议。
```

这项是否已经修复 16 / 128 卡 forced cleanup，需要看实验中 pause 后最后 pending group 的栈或日志。如果还能看到 pending 明确停在 judger，说明当前 wrapper/cancel 还不够；如果 pending 能在 judger cancel timeout 后稳定返回 `ABORTED`，则说明该层 tail 已基本覆盖。

## Item 6: 本地 cancellation drain 可能无界等待

### 问题描述

abort / cleanup 路径里有两类本地 cancel：

```text
worker:
  abort timeout 后取消原始 HTTP send_task

producer:
  cleanup timeout 后取消剩余 pending rollout task
```

如果取消后继续无界等待 task drain，那么“兜底 cleanup”本身也可能变成新的卡点。例如 HTTP client、底层连接关闭、Ray future 或某个 coroutine 的 cancellation handling 不及时，都会导致本地 `await` 长时间不返回。

这类问题的特征是：上层已经决定放弃等待，但还卡在“取消后的清理等待”里。

### 当前分支源码核对

当前分支已经有统一的 bounded drain：

```text
async def cancel_and_drain(tasks, *, timeout: float = 5.0):
    for task in tasks:
        task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return
```

也就是说，取消后最多等待 5 秒 drain；如果 task 仍不结束，`cancel_and_drain()` 会直接返回，不会继续无界卡住调用方。

worker 的 `_safe_post_request()` 中有两处关键使用：

```text
abort_task 先完成，send_task 在 abort_timeout 内仍未返回
  -> cancel_and_drain([send_task])
  -> 返回 REQUEST_ABORTED

_safe_post_request() 自己被取消
  -> cancel_and_drain([send_task, abort_task])
  -> 返回 REQUEST_ABORTED
```

producer 的 `_PendingTasks.cancel_all()` 也会先原子 claim 并清空 pending set，然后：

```text
await cancel_and_drain(list(tasks))
return len(tasks)
```

因此 cleanup timeout 到达后，producer 不会因为取消 pending task 后的 drain 无界等待而永久卡死。

### 合理性判断

这个问题成立，修复也合理。

它是一个兜底层修复，不解决前面几项的业务根因，但能保证最后的强制 cleanup 有时间边界：

```text
正常路径:
pending task 自己返回 completed / aborted

异常路径:
pending task 超过 cleanup timeout
  -> producer cancel_all()
  -> cancel drain 最多等 5 秒
  -> pause_produce 继续返回
```

如果没有这个 timeout，cleanup timeout 的语义会被削弱：表面上 60 秒后触发 force cancel，但实际可能在 cancel drain 里继续卡很久。

### 局限

这个修复只保证本地调用方不无界等待，不保证被取消的底层工作已经真正停止。

需要区分两层语义：

```text
本地 asyncio task 不再阻塞 producer / worker 返回
```

不等于：

```text
HTTP 底层连接、Ray 远端 actor method、LMDeploy 内部 request 都已经立即停止
```

具体风险包括：

- `cancel_and_drain()` 超时后直接返回，未完成 cancellation 的 task 可能仍在 event loop 中处于 cancelling 状态。
- 对 Ray remote call，取消本地 await 不一定等价于取消远端 actor 正在执行的方法。
- 对 HTTP 请求，取消 `client.send()` 通常会关闭/回收连接，但底层库或网络栈异常时不应假设一定立刻完成。
- 当前 timeout 是默认参数 `5.0`，不是单独配置项；如果不同环境需要调整，需要显式暴露配置。
- `cancel_and_drain()` 超时后没有日志，排查时不容易知道是否曾经发生 drain timeout。

### 当前结论

当前判断是：

```text
本地 cancellation drain 无界等待这个风险成立；
当前分支用 cancel_and_drain(timeout=5.0) 做了合理兜底；
它能避免 cleanup 兜底再次卡死在本地 await drain 上。
```

16 卡第三次实验没有触发分钟级 cancellation drain tail，说明这层兜底至少没有成为新的显著尾部卡点。但 128 卡场景仍建议观察：

- cleanup timeout 后 `cancel_all()` 到 `pause_produce completed` 的耗时；
- worker `_safe_post_request()` abort timeout 后是否还能快速返回；
- 是否需要给 `cancel_and_drain()` 的 timeout 超时分支加 warning 统计。

## Item 7: FILTERED group 保留 routed experts 导致内存持续增长

### 问题描述

这项不是 abort 请求链路本身的问题，而是同一轮 128 卡排查中暴露出的 rollout payload 生命周期问题。

现象是 LMDeployWorker、RayEngineWorker 和 RayWorkerWrapper 的 RSS 持续增长，并且与 `leftover_filtered` 单调增长高度相关。更准确的链路是：

```text
completed group 生成完成
  -> producer 业务有效性过滤 is_valid_sample_fn(group) 返回 false
  -> group 被标记为 FILTERED
  -> group 仍写入 replay buffer
  -> FILTERED group 不再进入 trainer 消费路径
  -> trainer 侧读取 routed_experts 后 free ObjectRef 的逻辑不会执行
  -> replay buffer 长期持有 routed_experts ObjectRef
  -> object store / 相关 worker RSS 持续增长
```

这个问题在 `enable_return_routed_experts=True` 且 partial rollout 开启时更明显，因为每个 sample 可能持有较大的 `routed_experts` Ray ObjectRef。

因此本轮内存增长的主因不应简单归结为“训练用完 routed experts 后没有 free”。训练正常消费路径里，trainer worker 确实会在 `ray.get()` routed experts 后调用 `ray.internal.free(..., local_only=False)`。真正的问题是 filtered group 根本不会走到 trainer 消费路径。

### 当前分支源码核对

当前分支中，producer 写 replay buffer 前已经加了清理：

```text
is_completed = get_group_status(group) == Status.COMPLETED
produced_tokens = sum(len(item.response_ids) for item in group if item.response_ids is not None)

if is_completed:
    is_valid = self.is_valid_sample_fn(group)
    if not is_valid:
        for item in group:
            item.status = Status.FILTERED
            reset_rollout_response(item)

await self.replay_buffer.put(...)
self.progress.add_produced(..., tokens=produced_tokens)
```

这里有两个关键点：

- `produced_tokens` 在 `reset_rollout_response()` 之前统计，所以清空 response 不会把 produced token 指标变成 0。
- 清理发生在 `replay_buffer.put()` 之前，所以 replay buffer 不会持有 filtered sample 的完整 rollout payload。

`reset_rollout_response()` 当前会清理：

```text
if rollout_state.routed_experts is Ray ObjectRef:
    ray.internal.free([routed_experts], local_only=False)

rollout_state.tokens = prompt_ids
rollout_state.response = ""
rollout_state.response_ids = []
rollout_state.logprobs = []
rollout_state.routed_experts = None
rollout_state.finish_reason = None
rollout_state.response_mask = []
rollout_state.response_model_steps = []
rollout_state.reward = None
rollout_state.error_msg = None
```

这与“filtered group 不再保留 response / response_ids / logprobs / reward / routed_experts”的修复目标一致。

replay buffer 里原本已有的清理主要覆盖 `Status.EXPIRED`：`put()` 发现 group 已经过期会 reset；`refresh_staleness()` 把 completed / aborted 翻成 expired 时也会 reset。`FILTERED` 如果不在 producer 侧提前 reset，就会长期保留 payload。

### 合理性判断

这个发现成立，修复也合理。

核心原因是 `FILTERED` 的语义和 `COMPLETED` 不同：

```text
COMPLETED:
  可能被 trainer 消费，保留 rollout payload 是必要的。

FILTERED:
  已判定不会用于训练，继续保留完整 rollout payload 没有收益。
```

因此在写 replay buffer 前清理 filtered sample，是更正确的生命周期收口点。它既保留了 filtered group 作为计数、诊断和进度统计对象，又释放了大对象引用，避免 replay buffer 变成 routed experts 的长期持有者。

这也解释了为什么 `leftover_aborted` 本次不是主要来源：如果它基本稳定在约 128，而 `leftover_filtered` 单调增长并伴随 RSS 增长，那么增长项更像是 filtered payload 的累积，而不是 aborted backlog。

LMDeploy shared store 也不应优先判为主因。当前 XTuner LMDeploy decode 路径是通过 shared store key 取回 routed experts，再 `ray.put()` 到 Ray object store；如果 LMDeploy shared store 的 `get()` 是 pop 语义，那么正常 decode 后 key 不会继续留在 LMDeploy shared store。后续长期持有的更可能是 XTuner/Ray 侧 `RolloutState.routed_experts` ObjectRef。

### 局限与风险

这项修复主要针对 `FILTERED` group，仍有几个边界需要观察：

- `reset_rollout_response()` 只对 `routed_experts` 是 Ray ObjectRef 的情况显式 `ray.internal.free()`。当前 `RolloutState.routed_experts` 类型是 `np.ndarray | RayObjectRef | None`，主路径能覆盖；如果未来变成 list / 嵌套结构，需要扩展释放逻辑。
- `ray.internal.free(local_only=False)` 只是释放 Ray object store 引用；如果还有其他 Python 对象引用持有同一个 ObjectRef 或已 materialized 的 numpy array，内存不会立刻完全下降。
- 已经写入 replay buffer 的历史 filtered group，如果是在修复前产生的，除非额外扫一遍并 reset，否则不会被这段 producer 新逻辑 retroactive 清掉。
- 如果 `FAILED` group 也可能携带历史 `routed_experts`，并且同样不会进入 trainer 消费路径，它确实是同类风险。当前修复没有展开处理 FAILED，需要后续用 leftover_failed 和 RSS 关联性确认。
- 清空 response / reward 后，filtered group 只适合做状态计数和重试/统计，不再适合做详细样本内容分析。如果后续需要 debug filtered 原因，应在 reset 前记录轻量诊断字段。

### 当前结论

当前判断是：

```text
FILTERED group 保留 routed experts 导致内存持续增长这个问题成立；
主因更像是 replay buffer 持有 filtered rollout payload，
而不是 trainer 正常消费路径没有 free routed experts；
在 producer 标 FILTERED 时立即 reset_rollout_response() 是正确修复。
```

下一轮 16 / 128 卡实验需要确认：

- `leftover_filtered` 可以继续作为计数增长；
- 但 LMDeployWorker / RayEngineWorker / RayWorkerWrapper RSS 不再随 filtered 数量单调增长；
- `reset_rollout_response()` 的 ObjectRef free 是否有可观测效果；
- `FAILED` group 是否也存在类似 routed experts 生命周期问题。

## 架构性结论

这 7 个问题表面上分散在 HTTP pool、LMDeploy abort、late response、batch judge、judger cancel、cancellation drain、replay buffer payload cleanup 等不同位置，但它们不是 7 个彼此独立的 bug。

更底层的共同问题是：当前 rollout 过程本质上是一个复杂状态机，但代码里没有把它显式建模成统一的 lifecycle / state machine。

现在的 `RolloutState` 更像一个被多层模块共享修改的数据对象：

```text
rollout worker:
  写 response / response_ids / logprobs / routed_experts / finish_reason / status

agent loop:
  调 judger，写 reward，也可能改 status

producer:
  判断 group 是否有效，写 FILTERED，写 replay buffer

replay buffer:
  根据 staleness 写 EXPIRED，并清理 payload

trainer:
  消费 completed group，并在训练路径里释放 routed experts

pause / abort:
  又跨 producer / agent loop / rollout controller / worker / LMDeploy / judger 修改或解释状态
```

每一层的局部逻辑单看都有道理，但系统缺少一个全局 scope 来回答这些问题：

- 一个 rollout group 当前处在哪个 phase。
- 哪一层是当前 phase 的 owner。
- 哪些事件可以触发状态转换。
- 哪些状态是 terminal status。
- terminal status 下哪些 payload 必须保留，哪些必须清理。
- pause / abort 后，late request、late response、late judge result 是否还允许生效。
- 写入 replay buffer 前是否已经满足该状态对应的字段不变量。

因此当前修复呈现出明显的补丁式状态机特征：

```text
HTTP pool hidden queue
  -> 初始化时调大 http concurrency

abort 后仍有 late /generate
  -> pause_produce 期间 periodic abort

abort 后 completed response 晚到
  -> parse response 后再改 ABORTED

mixed completed / aborted group 进入 batch judge
  -> group 内有 ABORTED 就 skip batch judge

all completed group 已经进入 judger
  -> SingleTurnAgentLoop 用 pause_event cancel judger task

cleanup cancel drain 可能卡死
  -> cancel_and_drain 增加 timeout

FILTERED group 持有 routed experts
  -> 标 FILTERED 时 reset_rollout_response
```

这些补丁都是合理的，也都能降低对应问题的发生概率或尾部影响。但它们共同说明：当前系统没有统一的 rollout lifecycle 不变量，只能在问题暴露的层面继续补判断。

### 更准确的抽象

rollout group 应该被建模为带全局 scope 的状态机，而不是普通可变数据对象。一个更合理的抽象至少应包含：

```text
RolloutGroupScope
  group_id
  generation_epoch / abort_epoch
  model_step
  owner
  phase
  terminal_status
  payload_state
```

其中 `phase` 描述它正在系统哪一层：

```text
CREATED
SCHEDULED
HTTP_PENDING
BACKEND_WAITING
BACKEND_RUNNING
RESPONSE_RECEIVED
JUDGING
BUFFERED
CONSUMED
CLEANED
```

`terminal_status` 描述最终业务状态：

```text
COMPLETED
ABORTED
FILTERED
FAILED
EXPIRED
```

`payload_state` 描述重字段生命周期：

```text
FULL_PAYLOAD
LIGHT_METADATA_ONLY
FREED
```

所有状态变化都应通过统一的 transition 入口，而不是各模块直接写字段：

```text
transition(group, event)
```

例如：

```text
EVENT_ABORT(epoch=X)
EVENT_HTTP_SENT
EVENT_RESPONSE_RECEIVED
EVENT_JUDGE_STARTED
EVENT_JUDGE_CANCELLED
EVENT_FILTER_REJECTED
EVENT_STALENESS_EXPIRED
EVENT_TRAIN_CONSUMED
EVENT_PAYLOAD_CLEANED
```

这样才能定义全局不变量。

### 应有的不变量

后续如果要重构，建议先明确下面这些不变量：

```text
1. pause / abort 是 epoch 级控制事件。
   某个 epoch 被 abort 后，该 epoch 的 late response / late judge result 不得再把 group 提升为可训练 COMPLETED。

2. 只有 current epoch 且 terminal_status=COMPLETED 且 reward 有效的 group 才能进入 trainer。

3. ABORTED / FILTERED / FAILED / EXPIRED 默认不得持有 routed_experts 等重 payload。
   如需保留 debug 信息，只保留轻量 metadata。

4. HTTP 层不能存在 XTuner 不可观测、不可取消的隐藏队列。
   本地 pending、已发送、backend waiting、backend running 必须能分层统计。

5. judger 属于 rollout lifecycle 的一部分。
   pause 后 in-flight judge 要么 bounded cancel，要么返回时必须做 epoch/status 二次校验。

6. replay buffer 不应同时承担“训练数据缓存”和“所有失败/过滤样本的重 payload 归档”职责。
   不可训练样本进入 buffer 前应先清理重字段。

7. cleanup timeout 必须是最后兜底，并且兜底自身也必须 bounded。
```

有了这些不变量，很多现在散落的补丁可以收敛成统一的状态转换规则。

### 对当前修复的评价

当前 PR 的修复方向是务实的。它没有推倒重写状态机，而是在已经暴露的高风险边界上补了必要防线：

- 减少 XTuner/httpx 隐藏排队。
- 用 periodic abort 覆盖 late /generate。
- 防止 abort 后 late completed response 进入 judger。
- 防止 mixed aborted group 被 batch judge 拖住。
- 给 SingleTurnAgentLoop 的 judger 增加 bounded cancel。
- 给 cancellation drain 增加 timeout。
- 收口 filtered routed experts 生命周期。

这些修复能显著改善 8 / 16 卡场景，也有希望降低 128 卡 forced cleanup 的尾部。但从架构上看，它们仍然是局部补丁，不是全局语义收敛。

因此，对 128 卡是否“完全修复”的判断不应只看 `/abort_request` 是否返回快，而应继续按状态机视角观测最后 pending group：

```text
它们还没发到 backend？
已经发到 backend 但没被 abort？
response 已回来但状态没降级？
已经回到 agent loop 但卡 judger？
已经进入 replay buffer 但 payload 没清？
还是 cleanup cancel drain 自己卡住？
```

如果没有这些 phase 级别指标，后续任何 forced cleanup 问题都会继续变成跨层猜测。

### 后续建议

短期建议：

- 在日志和 metrics 里补齐 group phase 级别统计，至少区分 producer pending、HTTP sending、backend waiting/running、response handled、judging、buffer put、cleanup cancel。
- 给 `cancel_and_drain()` timeout 分支加 warning 和计数。
- 把 batch judge 条件从“没有 ABORTED”收紧为“全部 COMPLETED”。
- 明确 `FAILED` group 是否需要像 `FILTERED` 一样清理 routed experts。
- 对 pause 后 judge 在 cancel timeout 内正常返回的 group，确认它是否允许继续作为 completed group 进入训练。

中长期建议：

- 引入 `generation_epoch / abort_epoch`，让 request、response、judge result、buffer write 都携带并校验 epoch。
- 建立显式 per-worker in-flight 限流和可观测队列，不再依赖 httpx pool 作为隐式 backpressure。
- 将 status 修改、payload cleanup、replay buffer 写入收敛到统一 transition API。
- 把 replay buffer 中的训练 payload 和不可训练状态 metadata 分离，避免不可训练样本长期持有大对象。
- 将 judger 纳入 rollout lifecycle，而不是作为 generate 之后的附属调用。

最终目标不是让每个局部补丁都更复杂，而是让系统在任意时刻都能回答：

```text
这个 rollout group 处于哪个 phase？
属于哪个 epoch？
当前 owner 是谁？
是否仍可训练？
是否还持有重 payload？
如果 pause 已发生，为什么它还没有返回？
```

只有这些问题变成系统内的一等状态，abort / pause 的排查才不会继续依赖跨层日志拼图。
