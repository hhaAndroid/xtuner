# XTuner Agent Rollout OpenTelemetry Tracing

这个目录沉淀 XTuner / CRG agent rollout 链路追踪的说明、Jaeger memory 配置和常用启动脚本。

核心启动脚本：recipe/otle/agentic_rl_test.sh

## 1. OpenTelemetry 是什么

OpenTelemetry，简称 OTel，是一套应用观测标准和 SDK。它本身不是 LLM 专用工具，而是通用的应用观测体系，可以观测 HTTP 服务、数据库、队列、推理服务、agent、工具调用、训练系统等。

OTel 常见三类信号：

- Trace：一次请求或一次任务的完整链路。我们现在主要用它。
- Metric：计数器、耗时分布、吞吐等聚合指标。
- Log：普通日志，也可以和 trace 关联。

这套实现主要用 Trace 来回答这些问题：

- 一个 agent rollout 总耗时花在哪里？
- `llm.chat` 慢是客户端慢、session server 慢、worker 慢，还是 tool 慢？
- 同一个样本在 baseline 和优化版本之间，耗时差异在哪里？

## 2. 核心概念

### Span

Span 是一段被计时的代码块，例如：

- `agent.function_call.forward`
- `llm.chat`
- `llm.client.http_post`
- `xtuner.session_server.forward_worker`
- `tool.python.run`

每个 span 有开始时间、结束时间、耗时、属性和父子关系。

### Trace

Trace 是一组 span 组成的一棵树，表示一次完整执行。比如一次 agent rollout 可以是一条 trace：

```text
agent.function_call.forward
  agent.function_call.policy_turn
    agent.policy.forward
      agent.policy.llm_chat
        llm.chat
          llm.client.http_post
            xtuner.session_server.forward_worker
              xtuner.session_server.stream_read
  agent.function_call.env_turn
    tool.python.run
```

### Trace Context

跨进程链路能连起来，靠的是上下文传播。OTel 标准的 HTTP header 是：

```text
traceparent
tracestate
```

在当前环境里，有些中间服务不会保留自定义 header，所以 lagent 侧还会把 trace context 放进请求 body 的 `_otel_trace_context` 字段，session server 再从 body 里恢复上下文。

### Attribute / Tag

Attribute 是 span 上的键值对。Jaeger UI 里通常叫 tags。我们用它保存分析需要的字段，例如：

```text
input_tokens
output_tokens
first_chunk_ms
first_output_token_ms
first_content_ms
finish_reason
target_url
case.id
run.id
```

### Resource / Service

Resource 描述产生 span 的进程或服务。Jaeger UI 里会显示成 service，例如：

```text
agent-rollout-trace
xtuner-session-server
```

注意：不同 service 的 span 仍然可以在同一条 trace 里。链路是否连起来看 parent-child，不看 service name 是否相同。

### Exporter

Exporter 负责把 SDK 内存里的 span 发出去。当前主要用 OTLP HTTP exporter，把 trace 发到 Jaeger：

```text
http://<host>:14318/v1/traces
```

没有 exporter 时，span 不会进入 Jaeger。使用 BatchSpanProcessor 时，span 会批量导出，不会无限保存在 SDK 内存里。

### Collector / Jaeger

Collector 是接收、处理、转发遥测数据的组件。Jaeger 2.x 本身基于 OpenTelemetry Collector 体系，可以直接接收 OTLP，再把 trace 存到 Jaeger storage 中并提供 UI 查询。

当前使用的是 Jaeger memory storage：

- 优点：简单，容器里无需再跑 Docker。
- 缺点：Jaeger 重启后数据会丢。

## 3. 整体流程

当前推荐链路是 Python 进程直接把 span 发给 Jaeger，不经过额外的中间 HTTP 服务：

```text
XTuner / CRG / lagent Python 进程
  业务代码创建 span
  OpenTelemetry SDK 维护当前 span 上下文
  BatchSpanProcessor 在进程内批量收集已结束的 span
  OTLP HTTP Exporter 发 HTTP POST /v1/traces # export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
        |
        v
Jaeger 进程 14318 端口
  OTLP HTTP receiver 接收 trace
  batch processor 处理 trace
  jaeger_storage_exporter 写入 memory storage
        |
        v
Jaeger memory storage
  保存 trace / span 数据
        |
        v
Jaeger query + UI 16686
  浏览器查询、搜索、展开 trace
```

所以主链路可以简化理解为：

```text
Python OTel exporter -> Jaeger OTLP receiver -> Jaeger storage -> Jaeger UI
```

端口含义：

```text
14318  OTLP HTTP 接收端口，实验进程把 trace 发到这里
14317  OTLP gRPC 接收端口，当前默认不用
16686  Jaeger UI / query 端口，浏览器打开这里
```

对应到环境变量：

```bash
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://10.102.250.69:14318/v1/traces
```

这里的 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` 指向 Jaeger 的 OTLP HTTP receiver。不要填 `16686`，`16686` 只是给浏览器看的 UI。

Jaeger 2.x 本身可以直接接收 OTLP，所以这套方案不需要单独再启动 OpenTelemetry Collector。我们启动的 Jaeger 进程同时承担了三件事：

```text
receiver  接收 Python exporter 发来的 span
storage   保存 span，当前是 memory storage
query/ui  提供浏览器查询和展示
```

另外，本目录里还有一个可选的 `otlp_http_sink.py`。它不是 Jaeger 主链路的一部分，只用于把 OTLP HTTP 请求解码后落盘成 jsonl：

```text
Python OTel exporter -> otlp_http_sink.py -> spans.jsonl
```

这个 sink 不会自动再转发给 Jaeger。如果你使用 Jaeger UI 分析，就不需要启动它。

跨进程链路关系和数据上报是两件不同的事：

- 链路关系靠 `traceparent` / `_otel_trace_context` 在 agent、lagent、session server 之间传播。
- 数据上报靠每个进程里的 exporter 把本进程产生的 span 发到 Jaeger。

因此，同一条 trace 里可能有多个 Python 进程产生的 span。只要 trace context 传播正确，它们最终都会在 Jaeger 里拼成一条完整链路。

## 4. 最简 Python 用例

安装依赖：

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

启动 Jaeger 后，Python 侧配置：

```bash
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:14318/v1/traces
export OTEL_SERVICE_NAME=demo
```

最小代码：

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "demo"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("hello") as span:
    span.set_attribute("case.id", "example-1")
    span.set_attribute("input_tokens", 128)
```

打开：

```text
http://127.0.0.1:16686/
```

选择 service `demo` 即可查询。

## 5. 当前 XTuner / CRG 实现

### XTuner tracing helper

文件：

```text
xtuner/v1/rl/rollout/otel.py
```

作用：

- 初始化 OpenTelemetry SDK。
- 支持 OTLP HTTP / OTLP gRPC / console exporter。
- 支持从 HTTP header 或 body 中恢复 trace context。
- 支持向下游 HTTP 请求注入 trace context。
- 提供 `start_span` / `begin_span` / `end_span` / `set_attrs` / `use_context` 等工具。

### XTuner session server

文件：

```text
xtuner/v1/rl/rollout/session_server.py
```

当前记录的关键 span：

- `xtuner.session_server.on_request`
- `xtuner.session_server.apply_chat_template`
- `xtuner.session_server.trace_store.search_prompt`
- `xtuner.session_server.tokenize_delta`
- `xtuner.session_server.forward_worker`
- `xtuner.session_server.stream_read`
- `xtuner.session_server.parse_stream_response`
- `xtuner.session_server.on_response`
- `xtuner.session_server.trace_store.insert_response`

其中最重要的是：

```text
xtuner.session_server.forward_worker
```

它会记录：

```text
input_tokens
max_tokens
output_tokens
request_bytes
response_bytes
first_chunk_ms
first_output_token_ms
first_content_ms
finish_reason
target_url
worker_base_url
traceparent_context_source
```

解释：

- `first_chunk_ms`：session server 收到 worker 第一个 stream chunk 的耗时。
- `first_output_token_ms`：第一个带 `output_ids` 的 chunk 出现时间。
- `first_content_ms`：第一个普通 `delta.content` 出现时间。
- `output_tokens`：worker 返回的预测 token 数。
- `finish_reason`：例如 `tool_calls` / `stop` / `abort`。

如果 `first_output_token_ms` 很小但 `output_tokens` 很大，通常说明 worker 很快开始 decode，慢在长输出。

### lagent HTTP client

文件：

```text
/mnt/shared-storage-user/huanghaian/code/gateway/lagent/lagent/llms/model.py
```

当前记录的关键 span：

- `llm.client.http_attempt`
- `llm.client.http_post`
- `llm.client.stream_read`

它会在 HTTP 请求里注入 trace context，并默认把 context 也写入 body：

```text
_otel_trace_context
```

这是为了处理部分中间服务不透传自定义 header 的情况。

### CRG math coder agent

文件：

```text
/mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/recipe/math_code_interpreter/infer/agents/math_coder/config.py
/mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/recipe/math_code_interpreter/infer/agents/math_coder/tracing.py
/mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/recipe/math_code_interpreter/infer/agents/math_coder/tools/python_executor.py
```

当前记录的关键 span：

- `agent.function_call.forward`
- `agent.function_call.policy_turn`
- `agent.policy.forward`
- `agent.policy.llm_chat`
- `llm.chat`
- `agent.env.forward`
- `agent.env.execute_tool`
- `tool.python.run`
- `tool.python.http_post`

`tracing.py` 会自动从 OTel baggage 里读取这些字段，并打到每个 agent span 上：

```text
run.id
case.id
sample.group_index
sample.repeat_index
sample.repeat_k
sample.message_uid
sample.data_source
```

### 跨实验对照 ID

文件：

```text
xtuner/v1/rl/agent_loop_manager/sampler.py
xtuner/v1/rl/agent_loop/localhost_agent_loop/agent_in_localhost_loop.py
```

Sampler 会给每个 rollout 样本写入：

```text
RolloutState.extra_fields["otel"]
```

包含：

```text
case.id
run.id
sample.group_index
sample.repeat_index
sample.repeat_k
sample.message_uid
sample.data_source
```

其中：

- `case.id`：跨实验稳定，用来找同一个样本同一个 repeat。
- `run.id`：一次实验唯一，用来区分 baseline / opt。
- `uid` / `session_id`：单次执行唯一，不建议作为跨实验对照 id。

推荐启动前显式设置：

```bash
export XTUNER_OTEL_RUN_ID=baseline
```

优化后：

```bash
export XTUNER_OTEL_RUN_ID=opt_v1
```

Jaeger 里用 tag 搜：

```text
case.id=<某个 case id>
```

如果两次实验数据都还在 Jaeger memory 里，就能看到同一 `case.id` 对应的多条 trace，再用 `run.id` 区分。

## 6. 环境变量

XTuner / CRG agent 推荐配置：

```bash
export AGENT_OTEL_ENABLED=1
export XTUNER_OTEL_ENABLED=1

export OTEL_SERVICE_NAME=agent-rollout-trace
export XTUNER_OTEL_SERVICE_NAME=xtuner-session-server

export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://10.102.250.69:14318/v1/traces

export XTUNER_OTEL_RUN_ID=baseline
```

字段说明：

- `AGENT_OTEL_ENABLED=1`：开启 CRG / lagent agent 侧 tracing。
- `XTUNER_OTEL_ENABLED=1`：开启 XTuner session server tracing。
- `OTEL_SERVICE_NAME`：agent 侧 service name。
- `XTUNER_OTEL_SERVICE_NAME`：XTuner 侧 service name。
- `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`：使用 OTLP HTTP。
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`：Jaeger OTLP HTTP 接收地址。
- `XTUNER_OTEL_RUN_ID`：本次实验 ID，建议手动设置成 `baseline`、`opt_v1` 等。

如果 Jaeger 就在本机：

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:14318/v1/traces
```

如果实验进程从其他机器或容器访问当前机器：

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://<当前机器 IP>:14318/v1/traces
```

## 7. 当前方案限制和推荐演进

当前默认方案是：

```text
Python span -> OTLP HTTP exporter -> Jaeger 14318 -> Jaeger memory storage -> Jaeger UI 16686
```

它的优点是部署很轻，容器里不需要再启动 Docker，也不需要额外数据库。缺点是 Jaeger 的 trace 数据只存在内存里：

- Jaeger 进程退出，trace 会丢。
- 执行 `restart_jaeger_memory.sh`，旧 trace 会被清空。
- 机器或容器重启，trace 会丢。
- memory 达到上限时，旧 trace 可能被淘汰。
- 如果要在 UI 里同时对比 baseline / opt，中间不能清空 Jaeger memory。

所以 memory Jaeger 适合：

- 临时调试。
- 快速定位瓶颈。
- 看一轮正在跑的实验。

它不适合：

- 长期保存实验 trace。
- 多天后回溯分析。
- 稳定对比多个实验版本。

推荐演进方案分两类数据：

```text
短期实时分析：Jaeger memory
长期可视化回溯：Jaeger 可直接查询的持久化 storage
长期脚本分析：人可读 jsonl
```

这里的“Jaeger 格式文件”不建议理解成单个 `.json` 或 `.jsonl` 文件。Jaeger UI 正常读取的是 storage backend，不是直接读取一个 trace 文件。更合理的形式是一个 Jaeger 可查询的持久化存储目录，例如 Badger：

```text
trace_store/
  badger/
    keys...
    values...
```

以后要回看时，启动 Jaeger 并指向这个 Badger 目录：

```text
Jaeger UI -> Jaeger query -> Badger storage directory
```

这会比较接近 TensorBoard 的使用体验：

```text
训练过程写文件 -> 想看时启动服务读取文件
```

最实用的下一步是把 Jaeger storage 从 memory 换成持久化后端：

- Badger：单机文件存储，部署相对简单，适合当前容器环境优先尝试。
- Elasticsearch：成熟，查询能力强，但部署比较重。
- ClickHouse：适合大量 trace，性能好，但部署和维护也更复杂。

同时，人可读 jsonl 仍然有价值。它适合：

- 快速用脚本统计耗时占比。
- 长期归档关键字段。
- 不启动 Jaeger 时做离线分析。
- 和训练日志、样本 id、reward 等业务数据做 join。

因此比较理想的长期形态是双写：

```text
Python OTel SDK
  -> OTLP exporter
  -> OpenTelemetry Collector
       -> Jaeger / Badger storage     # 给 Jaeger UI 直接查询
       -> jsonl / file exporter       # 给人和脚本离线分析
```

Collector 的价值在于 fan-out：Python 进程只需要发一份 OTLP，Collector 负责同时写入多个后端。这样比让 Python 进程自己双写更干净，也更容易加采样、限流、重试、字段清洗。

如果暂时不想引入 Collector，也可以分阶段做：

```text
阶段 1：Python -> Jaeger memory
阶段 2：Python -> Jaeger Badger
阶段 3：Python -> Collector -> Jaeger Badger
                         -> jsonl
```

本目录的 `otlp_http_sink.py` 可以做简单 jsonl 落盘，但它不是 Jaeger 主链路的一部分，也不会自动把数据转发给 Jaeger。当前阶段为了快速定位 rollout 瓶颈，默认仍使用 Jaeger memory。需要跨实验长期对照时，优先考虑 Badger；需要同时保留人可读 jsonl 时，再引入 Collector 双写。

## 8. Jaeger memory 启动和重启

### 下载 Jaeger

```bash
bash recipe/otle/scripts/download_jaeger.sh
```

默认下载到：

```text
/tmp/jaeger
```

可以改：

```bash
export JAEGER_HOME=/path/to/jaeger
bash recipe/otle/scripts/download_jaeger.sh
```

### 启动

```bash
bash recipe/otle/scripts/start_jaeger_memory.sh
```

端口：

```text
16686  Jaeger UI
14318  OTLP HTTP /v1/traces
14317  OTLP gRPC
```

浏览器打开：

```text
http://127.0.0.1:16686/
```

### 查看状态

```bash
bash recipe/otle/scripts/status_jaeger_memory.sh
```

关键检查：

```bash
curl http://127.0.0.1:16686/api/services
```

如果返回：

```json
{"data":[],"total":0,"limit":0,"offset":0,"errors":null}
```

说明 Jaeger 正常，但还没有收到 trace。

### 重启并清空 memory

```bash
bash recipe/otle/scripts/restart_jaeger_memory.sh
```

这会清空 memory storage 中的旧 trace。

注意：如果你想在 Jaeger UI 里同时对比 baseline 和 opt，不要在两次实验之间重启 Jaeger。memory 重启后旧 trace 会消失。

### 停止

```bash
bash recipe/otle/scripts/stop_jaeger_memory.sh
```

脚本会先停 tmux session，再清理可能残留的 Jaeger 进程。

## 9. 可选：OTLP HTTP jsonl sink

如果暂时不想启动 Jaeger，也可以把 OTLP HTTP trace 简单落盘：

```bash
bash recipe/otle/scripts/start_otlp_http_sink.sh
```

默认监听：

```text
0.0.0.0:4318
```

输出：

```text
/tmp/otelcol/spans.jsonl
/tmp/otelcol/received/*.pb
```

对应环境变量：

```bash
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:4318/v1/traces
```

这个工具只适合备份或脚本分析，交互式链路分析还是建议用 Jaeger。

## 10. 常见问题

### 看到 `connection refused`

说明 exporter 指向的 OTLP endpoint 没有服务在监听。例如：

```text
127.0.0.1:4318 connection refused
```

检查：

```bash
bash recipe/otle/scripts/status_jaeger_memory.sh
```

如果 Jaeger 用本目录配置启动，实验应发到：

```text
http://<host>:14318/v1/traces
```

不是默认的 `4318`。

### 为什么 UI 是 16686，但 endpoint 是 14318

这是两个不同端口：

- `16686`：浏览器 UI 和 Jaeger query API。
- `14318`：实验进程上报 trace 的 OTLP HTTP 接收端口。

### 为什么重启后 trace 不见了

当前使用 memory storage。Jaeger 重启会清空所有 trace。如果要长期保存，需要换成 Badger / Elasticsearch / ClickHouse 等持久化后端，或者额外使用 jsonl sink 落盘。

### 为什么 `first_output_token_ms` 很小但 `forward_worker` 很慢

这通常表示 worker 很快开始输出，但总输出很长。例如：

```text
first_output_token_ms = 140ms
output_tokens = 7055
finish_reason = tool_calls
```

这种情况慢点多半在 decode 长输出，或者模型先生成大量 reasoning / tool call 参数后才结束。

### 如何查同一个样本的两次实验

在 Jaeger Search 的 tags 里查：

```text
case.id=<case id>
```

如果 baseline 和 opt 都还在 Jaeger memory 里，会返回多条 trace。看 `run.id` 区分是哪次实验。

如果两次实验之间重启过 Jaeger memory，旧 trace 已经丢失，UI 里无法查到两条。
