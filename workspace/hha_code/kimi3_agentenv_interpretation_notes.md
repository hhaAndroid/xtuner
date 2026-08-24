# Kimi K3 AgentENV 与 Agentic RL 机制解读

本文整理对 `k3_tech_report.pdf` 中 agentic RL、AgentENV、partial rollout、统一白盒 RL 环境等内容的理解，重点围绕以下问题：

- Kimi K3 所说的 paused rollout / resume 是否等价于暂停黑盒 agent 的流式推理？
- AgentENV 到底恢复的是 sandbox 状态，还是 LLM/agent 状态？
- 4.2.1 的 unified white-box RL environment 是否表示把 Claude Code 这类黑盒 agent 变成白盒？
- 如果要训练或评测 Claude Code 这类黑盒 agent，应该如何理解其边界？
- AgentENV 是否支持有状态工具调用？

## 1. PDF 基本情况

已确认 `workspace/hha_code/k3_tech_report.pdf` 是 LaTeX/pdfTeX 生成的技术报告，不是扫描件。

- 页数：47 页
- 可抽取文本：约 187K 字符
- 空文本页：0 页
- 标题：`KIMI K3: OPEN FRONTIER INTELLIGENCE TECHNICAL REPORT OF KIMI K3`

因此报告正文可以直接解析和检索。图表中的文字如果是 PDF 文本/矢量对象，一般也可读；若是位图图像，则需要页面渲染或 OCR 才能进一步分析像素细节。

## 2. Partial Rollout 与 Paused Rollouts

报告在 agentic RL 部分提到：

> Paused rollouts are enqueued and prioritized for resumption at the start of the next iteration, powered by our sandbox infrastructure.

结合上下文，这里的机制大致是：

1. rollout 阶段对 `N` 个 prompt 各采样 `K` 条 completion，维护 `N x K` 条 active trajectories。
2. 不等待所有长轨迹结束，只要有比例 `lambda` 的轨迹完成，就提前结束本轮 generation。
3. 未完成的长轨迹被暂停并入队。
4. 下一次 iteration 开始时，优先恢复这些未完成轨迹。
5. 已完成 prompt 的所有 `K` 条响应会被送去 policy optimization。

这个设计主要用于降低 long-horizon task 的尾延迟。长轨迹可能跨越多个 RL iteration，因此它们还提到数据陈旧性和 per-token regularization，用来稳定这种极端 off-policy 的训练过程。

关键理解：这里的 pause/resume/fork 是训练系统层面的 rollout 调度机制，不应简单理解为“任意黑盒 agent 在 token streaming 中间可以被无损暂停再恢复”。

## 3. AgentENV 恢复的边界

AgentENV 是一个基于 Firecracker microVM 的 sandbox runtime，公开文档和论文都强调：

- microVM 隔离
- memory + disk snapshot
- pause / resume
- fork
- snapshot
- E2B-compatible HTTP API
- 支持大规模、低延迟启动和恢复 sandbox

论文 5.3.2 中还说，sandbox 可以在 agent 等待模型 inference result 时暂停，因为等待模型结果可能占 sandbox lifetime 的很大比例。

这说明 AgentENV 主要恢复的是：

- VM 内存
- VM 文件系统增量
- sandbox 中的进程状态
- sandbox 中运行的服务状态
- 工具执行环境状态

它不能天然恢复：

- 远端 LLM 服务的 HTTP streaming 连接
- inference server 内部正在生成的请求状态
- Claude Code 等黑盒 agent 的内部 planner 状态
- SDK 半截 stream 的协议状态
- 半截 assistant message 的语义级状态

所以如果一个完整黑盒 agent 框架运行在 sandbox 内，并且它自己正在通过 HTTP stream 接收 token，此时暂停 sandbox 或中断推理引擎连接，恢复后并不能保证从第 N 个 token 继续。更可能出现 stream EOF、timeout、SDK exception，或者 agent 进入自己的重试逻辑。

## 4. 更合理的系统形态：Agent 在外，Sandbox 是工具环境

更合理的 Kimi 式架构应该是：

```text
rollout manager / agent harness / inference scheduler
        |
        | 生成完整 action 或 tool call
        v
AgentENV sandbox
        |
        | 执行代码、读写文件、运行服务、返回 observation
        v
rollout manager 记录 trajectory 并继续下一步推理
```

也就是说：

- agent harness 和 rollout manager 在 sandbox 外部。
- LLM 推理请求由外部 inference scheduler 管理。
- sandbox 只在工具执行时 resume。
- 当模型正在生成下一步 action 时，sandbox 可以 pause，因为此时工具环境多数时候 idle。
- pause/resume 的安全点应放在 turn/action/tool-call 边界，而不是 token streaming 中间。

这种设计就能解释论文里 “paused rollout 入队，下个 iteration 优先恢复” 的说法：

- 恢复外部 trajectory 状态。
- 恢复或启动对应 sandbox。
- 恢复相关 prefix KV/cache。
- 继续从上一次完整交互边界执行。



## 5. 4.2.1 Unified White-Box RL Environment 的含义

报告 4.2.1 原文核心含义如下：

训练时如果只使用单一固定的 agent harness，模型可能过拟合到某一种 tool schema、system prompt、context management mechanism 或 interaction protocol。为了解决这个问题，Kimi K3 开发了一个统一的白盒 RL 环境，把 agent harness 表示成一组可配置、可组合的模块，包括：

- tool interfaces
- system prompts
- context management strategies
- skills
- memories
- subagents
- 其他组件

通过配置组合这些模块，这个环境可以实例化主流 harness，例如：

- Kimi Code
- Claude Code
- Codex
- OpenClaw
- Hermes
- 以及全新的 harness

RL 训练时，他们会针对不同任务组动态构造不同 harness configuration，让 Kimi K3 接触这些模块的不同组合，而不是只学习某一个固定 harness 的惯例。

这里的 “white-box” 不应理解成把官方 Claude Code 这类闭源黑盒 agent 反编译、插桩或透明化。更准确的理解是：

```text
不是：拿到 Claude Code 的内部实现、session state、planner、tool executor
而是：自己实现一个可控的 unified harness 框架，其中某些配置模拟 Claude Code / Codex / OpenClaw 等交互形态
```

因此它是“环境/harness 的白盒化”，不是“黑盒 agent 本体的白盒化”。

## 6. Claude Code 能否不在 sandbox 内？

如果要训练、评测或采样 Claude Code 这类黑盒 agent，Claude Code 可以不运行在 AgentENV sandbox 内，但需要改变工具接入方式。

一种可行架构是：

```text
Claude Code 进程：运行在外部 host / rollout worker
        |
        | MCP / wrapper tools
        v
AgentENV sandbox：真正执行 bash、读写文件、跑测试、开服务
```

Claude Code 支持 MCP，因此可以写一个 MCP server，把 AgentENV 的能力封装为 Claude Code 可调用的工具，例如：

- `sandbox_exec`
- `sandbox_read_file`
- `sandbox_write_file`
- `sandbox_pause`
- `sandbox_resume`
- `sandbox_snapshot`
- `sandbox_fork`
- `sandbox_open_port`

但这里有一个风险：Claude Code 的原生 Bash/Edit/Read 工具默认操作的是本地环境。若希望它只操作 AgentENV sandbox，需要通过权限、配置、prompt 或 MCP tool design 尽量限制本地工具，并让它只使用远程 sandbox tools。

如果只把 Claude Code 放在外面，但工具仍然是本地工具，那它不会天然变成 Kimi 那种可调度 rollout 架构。

## 7. 黑盒 Claude Code 训练/评测的现实边界

对 Claude Code 这种黑盒 agent，可以分三类目标：

### 7.1 Eval / Benchmark / 数据采样

可行。

让 Claude Code 通过 MCP 调用 AgentENV 工具，外部 orchestrator 记录 transcript、sandbox state、reward 和任务结果。这更像黑盒 agent benchmark。

### 7.2 Partial Rollout Pause/Resume

部分可行，但 pause 点应在完整 turn/tool 边界。

例如：

- 使用 `claude -p --max-turns N` 跑一小段。
- 等它完成一个完整响应或完整若干 turn。
- snapshot/pause sandbox。
- 下次用 Claude Code 的 continue/resume 能力继续。

但这不是 token-level streaming resume。

### 7.3 像训练自研 agent 一样精确控制 action 级 rollout

不太适合直接使用官方 Claude Code 黑盒。

原因：

- 不一定能拦截“模型刚生成 tool call、尚未执行”的内部边界。
- 不一定能保证只使用指定工具。
- 内部 session、重试、上下文管理、tool planner 都不可控。
- 中途 stream 断开后的恢复语义不可控。

如果目标是 RL training，更合理的是实现一个 Claude-Code-like white-box harness，而不是直接训练官方 Claude Code 黑盒。

## 8. AgentENV 是否支持有状态工具调用？

支持，但支持的是 sandbox 级有状态，而不是 LLM stream 级有状态。

AgentENV 支持以下状态保留：

- 同一 sandbox 内文件系统状态保留。
- 后台进程和服务可以继续运行。
- 可 pause/resume 保存并恢复 microVM 内存和磁盘状态。
- 可 snapshot/fork。
- 可通过 HTTP/WebSocket 访问 sandbox 内服务。
- 兼容 E2B API，因此可以对同一个 sandbox 反复 exec、connect、pause、resume。

典型有状态工具调用：

```text
tool_call_1: 写入 /app/main.py
tool_call_2: npm install
tool_call_3: 后台启动 dev server
tool_call_4: curl localhost:3000
tool_call_5: 修改文件并重新运行测试
```

这些状态都在同一个 VM/sandbox 里。

但普通 `exec` 通常是一次命令一次进程，所以如下命令不一定保留 shell 进程状态：

```bash
exec("cd /tmp")
exec("pwd")
```

第二次不一定还在 `/tmp`，除非显式传 `cwd`，或把状态写入文件、服务、REPL server 等长期状态中。

如果需要 Python/Node REPL 这种连续上下文，建议在 sandbox 内启动一个长期运行的 interpreter/server，然后每次 tool call 向这个 server 发送请求，而不是每次启动一个新 shell。

## 9. 当前结论

Kimi K3 的设计重点不是“让任意黑盒 agent 可以在 token streaming 中间无损暂停恢复”，而是把 agentic RL 的关键状态拆开管理：

- rollout / trajectory 状态在外部调度器中管理；
- LLM request / KV cache 在推理系统中管理；
- 工具执行环境状态在 AgentENV sandbox 中管理；
- agent harness 由统一白盒 RL 环境模块化配置；
- pause/resume 放在完整 turn/action/tool 边界，而不是半截 token stream 边界。

对我们自己的系统设计，这意味着：

1. 如果做自研 agent RL，应优先实现 white-box harness，把工具、上下文、记忆、子 agent、sandbox 调用都模块化。
2. 如果做黑盒 Claude Code 评测，可以把 Claude Code 放在 sandbox 外，并通过 MCP 调 AgentENV。
3. 如果要做 partial rollout，不要依赖暂停黑盒 agent 进程本身，而应在完整交互边界 checkpoint。
4. AgentENV 是很适合做有状态工具环境的，但不是 LLM stream 的恢复方案。

## 10. 参考链接

- AgentENV 文档：[https://kvcache-ai.github.io/AgentENV/](https://kvcache-ai.github.io/AgentENV/)
- AgentENV GitHub：[https://github.com/kvcache-ai/AgentENV](https://github.com/kvcache-ai/AgentENV)
- Firecracker snapshot 文档：[https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md)
- Claude Code CLI 文档：[https://code.claude.com/docs/en/cli-usage](https://code.claude.com/docs/en/cli-usage)
- Claude Code MCP 文档：[https://code.claude.com/docs/en/mcp](https://code.claude.com/docs/en/mcp)
- Claude Code sandbox 文档：[https://code.claude.com/docs/en/sandboxing](https://code.claude.com/docs/en/sandboxing)

## 11. Reasoning Effort RL 与 MOPD 的关系

Kimi K3 在 4.1.2 中提到 Reasoning Effort RL。它的核心目标不是简单限制 `max_tokens`，而是在 RL 阶段显式训练模型的“推理努力程度”控制能力。

他们将 RL experts 按两个维度组织：

```text
3 个任务域：
- general tasks
- general agents
- coding agents

3 个 reasoning effort levels：
- low
- high
- max
```

交叉后得到 `3 x 3 = 9` 个 expert models。

### 11.1 Per-Problem Budget Control

对每个问题 `x`，他们先用 cold-start model 估计一个初始 token budget：

```text
b0(x)
```

训练时再引入一个预算倍率：

```text
tau
```

如果某条 trajectory `y` 的总 token budget `T(y)` 超过：

```text
tau * b0(x)
```

就直接把这条 trajectory 的 reward 覆盖成 `-1`：

```text
if T(y) <= tau * b0(x):
    reward = original_task_reward
else:
    reward = -1
```

这里的 `T(y)` 按任务类型不同而不同：

- 对 general tasks，`T(y)` 统计 thinking tokens。
- 对 agentic tasks，`T(y)` 统计累计输出 tokens，包括 reasoning traces 和 tool-call arguments。

这个区别很重要。agentic task 中模型可能通过超长工具调用参数、超多中间步骤、超长 reasoning trace 来提高成功率。如果只约束 thinking tokens，不能有效限制 agentic rollout 的真实成本。

### 11.2 为什么用 Per-Problem Budget

他们不是给所有问题设置一个固定 token 上限，而是为每个问题估计 `b0(x)`。这样可以避免两类问题：

- 简单问题被允许过度思考。
- 复杂问题被统一硬上限过早截断。

`b0(x)` 代表这个问题在 cold-start model 下大概需要多少预算，`tau` 则控制当前 effort 档位允许使用多少倍预算。

所以这个机制更像：

```text
这个问题本身大概需要多少 token？
当前 effort level 允许它用多少倍？
```

而不是：

```text
所有问题统一最多生成 N 个 token。
```

### 11.3 Curriculum：从 Max 到 High/Low

训练不是一开始就直接压 low-effort。论文描述的流程是：

1. 先训练 `max-budget` variant，使用较大的 `tau`，让模型充分探索长链路推理和复杂 agentic 行为。
2. 即使是 max-budget，也仍然设置最大预算上限，用来抑制 excessive overthinking。
3. 然后逐步 anneal `tau` 到更小，得到 `high-effort` 和 `low-effort` expert models。
4. `tau` 的调整按 domain 配置，并有人类参与指导。

这说明 low/high/max 不是单纯 prompt 风格，而是经过不同预算约束下 RL 训练出来的不同策略分布。

## 12. 为什么还要用 MOPD 合并到一个模型

Reasoning Effort RL 训练出 9 个 expert 后，Kimi K3 没有直接部署 9 个模型，而是在 4.1.3 中使用 Multi-Teacher On-Policy Distillation, 即 MOPD，把这些能力合并到一个 unified model。

原因主要有几类。

### 12.1 部署成本

Kimi K3 是 2.8T 参数 MoE 模型。即使每 token 激活参数远小于总参数，完整权重仍然非常大。

如果直接保留：

```text
3 个 domain x 3 个 effort = 9 个 expert models
```

线上部署会面临很高的：

- 权重存储成本
- GPU/CPU/NVMe 内存管理成本
- 热更新成本
- cache 管理成本
- 推理调度成本
- 运维复杂度

合并成一个模型后，线上服务复杂度显著降低。

### 12.2 避免显式 Domain Routing

如果保留 9 个 expert，请求进入系统后需要先判断：

```text
这是 general task？
general agent？
coding agent？
应该用 low / high / max？
```

这个 router 本身也可能出错。统一模型可以将多 domain 能力内化到同一个模型中，effort 作为条件输入，domain 能力则通过蒸馏融合。

### 12.3 能力迁移

不同 expert 学到的能力不一定应该被隔离：

- coding agent 可能学到长程工具执行和代码环境操作。
- general agent 可能学到任务分解和跨应用状态管理。
- general tasks 可能学到 reasoning、faithfulness、vision、search 等能力。

MOPD 的目标就是把这些 specialized policies consolidated 到统一模型中，让能力可以互相迁移，而不是固化在不同 expert 模型里。

### 12.4 产品形态更简单

最终用户或产品系统不希望理解背后有 9 个专家模型。更自然的接口是：

```text
model = Kimi K3
reasoning_effort = low / high / max
```

这也和当前很多 reasoning model 的产品接口类似：同一个模型，通过 effort 参数控制推理预算和策略。

## 13. 合并后 Effort 是否还能手动控制

从论文 4.1.3 的公式看，MOPD 后的 student policy 是条件化的：

```text
pi_theta(yt | e, x, y<t)
```

这里 `e` 明确作为 student model 的输入条件，其中：

```text
e in {low, high, max}
```

因此最终模型理论上不是只能自己猜 effort，而是可以由外部显式指定 effort。

更准确地说，最终统一模型可以支持两种使用方式：

```text
用户或系统手动指定：
- low：快、省 token
- high：更稳
- max：复杂任务、长 horizon、愿意付出更多成本
```

也可以由产品层自动选择：

```text
系统自动调度：
- 简单任务默认 low
- 复杂 coding / agent task 升到 high 或 max
- 根据 latency、价格、用户套餐、任务风险动态选择 effort
```

所以 MOPD 不是把 effort 控制能力抹掉，而是把 9 个 expert 的能力压回一个可部署、可条件控制的模型。

### 13.1 MOPD 的训练直觉

论文中，MOPD 对给定 domain `d` 和采样的 effort level `e`，选择对应 teacher：

```text
pi_teacher^(d,e)
```

student 在条件 `e` 下生成 token，teacher 对 student 当前 token 给出 per-token OPD reward：

```text
reward = clipped log-ratio between teacher and student
```

直觉上就是：

```text
当 e = low 时，student 学 low-effort teacher 的策略。
当 e = high 时，student 学 high-effort teacher 的策略。
当 e = max 时，student 学 max-effort teacher 的策略。
```

这不是简单 SFT 蒸馏，而是带 dense per-token reward 的 on-policy 蒸馏，因此可以继续接入他们已有的 RL 框架和 partial rollout 基础设施。

### 13.2 Max 不等于无限思考

即使指定 `max`，也不表示模型必须把预算用满。Reasoning Effort RL 中，max-budget variant 仍然有 cap，用于压制 excessive overthinking。

因此 low/high/max 更像三种不同的策略分布：

```text
low：倾向短思考、少工具调用、低成本。
high：中等预算下提高稳定性和成功率。
max：允许更长推理和更复杂 agentic trajectory，但仍然受预算约束。
```

最终结论：MOPD 的作用是把 9 个 domain/effort experts 的能力合并进一个统一模型，同时保留 effort 作为外部可控条件。它不是让模型完全自行决定付出多少努力，而是让系统既可以手动指定 effort，也可以在产品层根据任务复杂度和成本约束自动选择 effort。

While we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting。

