# RolloutRouter 与移除 Ray Generate 设计草案

## 1. 背景

当前 RL rollout 链路里存在两套生成入口：

1. `AgentLoop` 调用 `RolloutController.generate.remote(...)`，再由 `RolloutController` 选择 `RolloutWorker`，最后由 `RolloutWorker.generate(...)` 请求后端推理服务。
2. agentic 场景中，业务 agent 更自然的入口是拿到一个 OpenAI-compatible base URL，然后自己发 `/v1/chat/completions` 请求。

第一套链路让 `RolloutController` / `RolloutWorker` 同时承担控制面和样本级生成职责：

- 控制面：启动 worker、health check、pause/abort、offload/onload、metadata、SessionServer 管理。
- 数据面：`RolloutState -> HTTP payload`、请求后端、解析响应、partial rollout、logprob/routed_experts 处理。

这会让后续 agentic loop、sandbox loop、localhost loop 和第三方 router 难以统一。更简单的方向是彻底删除样本级 Ray generate，把 URL 选择从 AgentLoop 中抽离到独立 `RolloutRouter`。

## 2. 目标

本设计的目标：

1. 删除 `RolloutController.generate` 和 `RolloutWorker.generate`，不再保留 Ray 样本级 generate 接口。
2. `RolloutController` / `RolloutWorker` 只保留控制面职责。
3. 新增独立 `RolloutRouter`，负责为每次 `generate_sample` 选择一个 rollout URL。
4. `AgentLoop` 支持 local class 和 Ray actor 两种形态，但都通过 `generate_sample(..., rollout_url=...)` 接收 URL。
5. `SingleTurnAgentLoop` 作为最小验证例子，复用 remove_generate_v1 分支中 agent loop 直接 HTTP 请求后端的核心逻辑。
6. `AgentInSandboxLoop` 和 `AgentInLocalhostLoop` 后续可以统一为“拿 URL 后由业务 runner 请求该 URL”的模式。
7. abort 能力不能丢失。删除 generate 只删除样本级 Ray generate，不删除控制面 pause/abort。
8. 复杂 agentic 场景的目标形态是统一 router URL。非 router 的 `url_pool` 只用于简单场景和 bootstrap 验证。

## 3. 非目标

本轮原型不解决以下问题：

1. 不一次性重构所有历史测试。
2. 不强制所有 AgentLoop 都走 SessionServer。
3. 不要求第三方 router 立即支持所有 XTuner 内部能力。
4. 不在 `RolloutRouter` 中生成或修改 `RolloutState.session_uid`。
5. 不把 URL 选择逻辑塞进 `AgentLoop`。
6. 不在第一版抽象通用 `StickyRouter[T]`。原型阶段可以在 `RolloutRouter` 内部维护简单 sticky map。

## 4. 核心设计

新的主链路：

```text
ProduceStrategy / ProduceContext
  -> rollout_router.acquire(rollout_state)
  -> agent_loop.generate_sample(rollout_state, rollout_url=endpoint.url)
  -> AgentLoop 内部直接请求 rollout_url/v1/chat/completions
  -> backend worker 或 SessionServer 或 third-party router
```

删除旧链路：

```text
AgentLoop
  -> RolloutController.generate.remote(...)
  -> RolloutWorker.generate.remote(...)
  -> backend HTTP
```

## 5. 组件边界

### 5.1 RolloutController

保留职责：

- 启动 rollout workers。
- 暴露 metadata。
- health check / recovery。
- pause / continue generation。
- cleanup after pause。
- offload / onload。
- shutdown。
- 管理每个 worker 的 SessionServer URL。

删除职责：

- 不再提供 `generate(...)`。
- 不再持有样本级 `SessionRouter` 给 generate 使用。
- 不再对 rollout response 应用 tool/reasoning parser。

### 5.2 RolloutWorker

保留职责：

- 启动实际 backend server。
- 启动 per-worker `SessionServer`。
- health check。
- `/abort_request`。
- offload / onload。
- shutdown / restart。

删除职责：

- 不再提供 `generate(...)`。
- 不再理解 `RolloutState`。
- 不再做 `_get_request_payload`、`_safe_post_request`、`_safe_handle_response`。
- 不再维护 worker 内部 partial rollout state。

### 5.3 RolloutRouter

`RolloutRouter` 是独立组件，只负责给样本选择 endpoint。

它不应该：

- 修改 `RolloutState`。
- 生成 `session_uid`。
- 调用 `AgentLoop`。
- 做 HTTP generate。
- 做 abort。

它应该：

- 根据配置选择 endpoint 类型。
- 根据 metadata 获取 active URLs。
- 在 `sticky_session=True` 且 `rollout_state.session_uid is not None` 时保持同 session 到同 URL。
- 在没有 `session_uid` 时做 round-robin，不记录 sticky 映射。

### 5.4 AgentLoop

`AgentLoop` 只表达业务样本如何生成。

它不应该：

- 自己从 controller metadata 中选择 URL。
- 初始化时绑定固定 worker URL。
- 持有 URL 池。

它应该：

- 接收 `rollout_url`。
- 对该 URL 发请求，或把该 URL 注入业务 runner。
- 对 LLM 访问错误做有限重试。
- 在 pause 时停止等待当前请求或 runner，并配合控制面 abort。

`AgentLoop` 不关心 `rollout_url` 背后是否存在 router，也不关心 router 类型。它只依赖
URL 的协议契约：

- `SingleTurnAgentLoop` 需要一个能直接返回训练所需 token/logprob 字段的 OpenAI-compatible URL。
- agentic loop 需要一个能配合 trace/session 采集的 OpenAI-compatible URL。

这些契约由 task 的 `rollout_router_config` 负责配置和保证，不作为 `generate_sample` 的标准入参。

AgentLoop 的重试和 router 的重试边界：

- AgentLoop 负责单次业务调用内的有限 LLM retry，例如连接错误、请求超时、5xx、可恢复响应格式错误。
- router 负责 endpoint 选择、健康状态、failover、sticky session。
- 没有统一 router 时，AgentLoop retry 只能重试同一个 `rollout_url`。
- 有统一 router 时，AgentLoop retry 仍然请求同一个 router URL，由 router 决定是否换后端 endpoint。

## 6. Router 类型与目标 Endpoint 类型

`RolloutRouter` 需要区分两个维度：

1. router 类型：这个 URL 是怎么暴露给 AgentLoop 的。
2. 目标 endpoint 类型：这个 URL 背后最终请求的是 raw worker 还是 SessionServer。

不能只用一个字段表达这两个维度。本文后续使用 `router_type` 表示暴露方式，
使用 `endpoint_type` 表示背后的目标服务。例如：

- third-party router 当前注册的是 SessionServer URL，因此它的 router 类型是 `third_party`，目标 endpoint 类型是 `session_server`。
- XTuner 内部 router 未来既可以转发到 raw worker，也可以转发到 SessionServer。
- 不启动独立 router 时，`url_pool` 可以直接返回 raw worker URL，也可以直接返回 per-worker SessionServer URL。

### 6.1 Router 类型

#### url_pool

不启动额外 router，只在 XTuner 进程内维护 URL 列表和 sticky session map。

AgentLoop 拿到的是某个具体 worker URL 或具体 SessionServer URL。

限制：

- agent 内部 retry 只能重试同一个具体 URL。
- 无法在 agent 内部透明 failover 到其它 worker 或 SessionServer。
- 不适合作为复杂 agentic loop 的默认方案。
- 适合 `SingleTurnAgentLoop` 或其它简单 single-request 场景的最小验证路径。

#### xtuner

XTuner 自己启动一个内部 router，对 AgentLoop 暴露统一 URL。

它可以选择转发到 raw worker，也可以选择转发到 SessionServer。

如果 agent 内部会拿到 `rollout_url` 后自行请求并重试，那么统一 URL 的价值会变高：

- `url_pool` 返回的是具体 worker/session_server URL，agent 内部 retry 只能重试同一个 URL。
- `third_party` 或 `xtuner` 返回统一 URL，router 可以在服务端做健康检查、失败重试和 sticky session。
- 后续接入健康检测时，统一 router 能在每次请求时基于最新健康状态跳过 inactive endpoint；`url_pool` 只能在外层 acquire 时判断一次，无法管理 agent 内部后续请求。

因此：

- 最小 SingleTurn 验证仍可使用 `url_pool + worker`。
- agentic 场景如果要求 agent 内部透明 retry/failover，应该使用 `third_party + session_server`，或实现 `xtuner + session_server`。
- 如果 third-party router 不满足 sticky/retry 语义，就需要实现 `XTunerRolloutRouter`。

#### third_party

复用当前 `add_apiproxy` 的逻辑，注册 URLs 到第三方 routedapiproxy，然后对 AgentLoop 返回第三方 routed URL。

第一版只支持 third-party router 注册 SessionServer URL。

### 6.2 目标 Endpoint 类型

#### worker

返回 `RolloutController.get_rollout_metadata()["server_url_dict"]` 中的原始 backend URL。

适用场景：

- `SingleTurnAgentLoop`。
- 不需要 trace store。
- 直接请求 backend `/v1/chat/completions`。

#### session_server

返回 `worker_session_url_dict` 中的 per-worker SessionServer URL。

适用场景：

- `AgentInSandboxLoop`。
- `AgentInLocalhostLoop`。
- 需要 SessionServer 处理 trace、prefix tokenization、logprobs、routed_experts。

### 6.3 推荐组合


| router 类型     | 目标 endpoint 类型   | 用途                                            |
| ------------- | ---------------- | --------------------------------------------- |
| `url_pool`    | `worker`         | `SingleTurnAgentLoop` 最小验证，不走 SessionServer   |
| `url_pool`    | `session_server` | 简单 agentic 验证；不支持 agent 内部透明 failover        |
| `xtuner`      | `worker`         | XTuner 统一 router 转发到 raw worker                 |
| `xtuner`      | `session_server` | XTuner 统一 router 转发到 SessionServer，agentic 目标形态之一 |
| `third_party` | `session_server` | 当前第三方 routedapiproxy 模式，agentic 目标形态之一        |


第一版不支持：


| router 类型     | 目标 endpoint 类型 | 原因                                                                     |
| ------------- | -------------- | ---------------------------------------------------------------------- |
| `third_party` | `worker`       | 当前第三方 router 注册路径依赖 SessionServer URL，且 agentic trace 需要 SessionServer |


注意：

- 如果第三方 router 不能保证 session sticky，则不能用于强依赖同 session 同 backend 的 agentic trace 场景。
- 第三方 router 注册逻辑应该从 `rl_trainer.py` 移到 `ThirdPartyRolloutRouter`。
- `router_type` 和 `endpoint_type` 是 router/config 层信息。`AgentLoop.generate_sample` 默认只接收 `rollout_url`。

## 7. Sticky Session 规则

`RolloutRouter` 不生成 session id，也不修改 `RolloutState`。

路由规则：

```text
if sticky_session and rollout_state.session_uid is not None:
    session_uid -> endpoint_url
else:
    round-robin endpoint_url
```

不要 fallback 到 `uid` 或 `message_uid`。

原因：

- `uid` 是样本标识，不一定是 session 标识。
- `message_uid` 可能在同一个 group 内共享，误用会退化为 group 级 sticky。
- 隐式生成 session 会让数据语义变得不透明。

同一个 group 内每个 sample 独立 acquire。默认不做 group 级 sticky。

## 8. 配置关系

`rollout_router_config` 和 `agent_loop_config` 不直接互相引用。它们共同挂在 task config 上。

删除 Ray generate 后，运行时不存在“没有 router 也能生成”的路径。因此第一版设计要求
每个 task 显式配置 `rollout_router_config`，不做隐式推断。这样配置虽然多一行，但能避免
`AgentLoopConfig` 暗中决定 URL 类型，也能让 review 时直接看出该 task 是走 raw worker、
SessionServer 还是 third-party router。

示意：

```python
class AgentLoopTaskConfig(BaseModel):
    task_name: str
    agent_loop_config: AgentLoopConfig
    rollout_router_config: RolloutRouterConfig
    sampler_config: SamplerConfig
    produce_strategy_config: ProduceStrategyConfig
    judger_config: JudgerConfig | None = None
    weight: float = 1.0
```

构建时：

```python
agent_loop = task_cfg.agent_loop_config.build(...)
rollout_router = task_cfg.rollout_router_config.build(rollout_controller)
```

推荐配置约定：

- `SingleTurnAgentLoopConfig` 对应 `RolloutRouterConfig(router_type="url_pool", endpoint_type="worker")`。
- `AgentInSandboxLoopConfig` 对应 `RolloutRouterConfig(router_type="url_pool", endpoint_type="session_server")`。
- `AgentInLocalhostLoopConfig` 对应 `RolloutRouterConfig(router_type="url_pool", endpoint_type="session_server")`。

## 9. 伪代码

### 9.1 Endpoint

```python
class RolloutEndpoint(BaseModel):
    url: str
    router_type: Literal["url_pool", "xtuner", "third_party"]
    endpoint_type: Literal["worker", "session_server"]
    rank: int | None = None
    model_name: str | None = None
```

### 9.2 Router Config

```python
class RolloutRouterConfig(BaseModel):
    router_type: Literal["url_pool", "xtuner", "third_party"] = "url_pool"
    endpoint_type: Literal["worker", "session_server"] = "worker"
    sticky_session: bool = True

    def build(self, rollout_controller) -> RolloutRouter:
        if self.router_type == "third_party":
            if self.endpoint_type != "session_server":
                raise ValueError("third_party router only supports endpoint_type='session_server' for now.")
            return ThirdPartyRolloutRouter(
                rollout_controller=rollout_controller,
                sticky_session=self.sticky_session,
            )
        if self.router_type == "xtuner":
            return XTunerRolloutRouter(
                rollout_controller=rollout_controller,
                endpoint_type=self.endpoint_type,
                sticky_session=self.sticky_session,
            )
        return UrlPoolRolloutRouter(
            rollout_controller=rollout_controller,
            endpoint_type=self.endpoint_type,
            sticky_session=self.sticky_session,
        )
```

### 9.3 URL Pool Router

```python
class UrlPoolRolloutRouter:
    def __init__(self, rollout_controller, endpoint_type, sticky_session=True):
        self.rollout_controller = rollout_controller
        self.endpoint_type = endpoint_type
        self.sticky_session = sticky_session
        self._session_to_endpoint = OrderedDict()
        self._rr_index = 0
        self._lock = asyncio.Lock()

    async def acquire(self, rollout_state: RolloutState) -> RolloutEndpoint:
        session_key = rollout_state.session_uid

        async with self._lock:
            endpoints = await self._load_active_endpoints()

            if self.sticky_session and session_key is not None:
                endpoint = self._session_to_endpoint.get(session_key)
                if endpoint is not None and endpoint in endpoints:
                    return endpoint

                endpoint = self._next(endpoints)
                self._session_to_endpoint[session_key] = endpoint
                return endpoint

            return self._next(endpoints)

    async def _load_active_endpoints(self) -> list[RolloutEndpoint]:
        metadata = await self.rollout_controller.get_rollout_metadata.remote()

        if self.endpoint_type == "worker":
            url_dict = metadata["server_url_dict"]
            status = metadata["worker_server_urls_status"]
        elif self.endpoint_type == "session_server":
            url_dict = metadata["worker_session_url_dict"]
            status = metadata["worker_session_urls_status"]
        else:
            raise ValueError(self.endpoint_type)

        endpoints = []
        for rank, url in sorted(url_dict.items()):
            if url and status.get(url, True):
                endpoints.append(
                    RolloutEndpoint(
                        url=url,
                        router_type="url_pool",
                        endpoint_type=self.endpoint_type,
                        rank=int(rank),
                        model_name=metadata["rollout_config"].model_name,
                    )
                )

        if not endpoints:
            raise RuntimeError("No active rollout endpoint available.")

        return endpoints

    def _next(self, endpoints: list[RolloutEndpoint]) -> RolloutEndpoint:
        endpoint = endpoints[self._rr_index % len(endpoints)]
        self._rr_index += 1
        return endpoint
```

### 9.4 Third Party Router

```python
class ThirdPartyRolloutRouter:
    def __init__(self, rollout_controller, sticky_session=True):
        self.rollout_controller = rollout_controller
        self.sticky_session = sticky_session
        self.started = False
        self.model_name = None
        self.routed_url = "http://.../v1"

    async def start(self):
        metadata = await self.rollout_controller.get_rollout_metadata.remote()
        model_name = metadata["rollout_config"].model_name
        self.model_name = model_name

        delete_from_routedapiproxy(model_name)
        for rank, url in sorted(metadata["worker_session_url_dict"].items()):
            if metadata["worker_session_urls_status"].get(url, False):
                register_to_routedapiproxy(model_name, url)

        self.started = True

    async def acquire(self, rollout_state: RolloutState) -> RolloutEndpoint:
        if not self.started:
            await self.start()
        return RolloutEndpoint(
            url=self.routed_url,
            router_type="third_party",
            endpoint_type="session_server",
            model_name=self.model_name,
        )

    async def shutdown(self):
        if self.model_name is not None:
            delete_from_routedapiproxy(self.model_name)
```

### 9.5 XTuner Router

`XTunerRolloutRouter` 是 XTuner 自己提供统一 router URL 的实现。它不是简单 URL 选择器，
而是会启动一个 HTTP server，对 AgentLoop 暴露稳定 base URL，并在服务端完成 sticky routing、
健康检查和失败处理。

如果第三方 routedapiproxy 能满足 sticky/retry 语义，可以先用 `third_party + session_server`
验证统一 URL 模式；否则 agentic 场景需要实现 `XTunerRolloutRouter`。

最小职责：

```python
class XTunerRolloutRouter:
    def __init__(self, rollout_controller, endpoint_type, sticky_session=True):
        self.rollout_controller = rollout_controller
        self.endpoint_type = endpoint_type
        self.sticky_session = sticky_session
        self.router_url = None
        self._server = None

    async def start(self):
        if self.router_url is not None:
            return

        endpoint_selector = UrlPoolRolloutRouter(
            rollout_controller=self.rollout_controller,
            endpoint_type=self.endpoint_type,
            sticky_session=self.sticky_session,
        )
        self._server = XTunerRouterServer(endpoint_selector=endpoint_selector)
        self.router_url = await self._server.start()

    async def acquire(self, rollout_state: RolloutState) -> RolloutEndpoint:
        if self.router_url is None:
            await self.start()

        metadata = await self.rollout_controller.get_rollout_metadata.remote()
        return RolloutEndpoint(
            url=self.router_url,
            router_type="xtuner",
            endpoint_type=self.endpoint_type,
            model_name=metadata["rollout_config"].model_name,
        )

    async def shutdown(self):
        if self._server is not None:
            await self._server.stop()
        self.router_url = None
```

`XTunerRouterServer` 至少需要处理：

- `/v1/chat/completions` 转发。
- streaming SSE 透传。
- request headers/body/query string 透传。
- timeout 和连接池。
- 基于请求 `session_id` 的 sticky routing。
- backend health / inactive URL 跳过。
- endpoint 失败计数、熔断和恢复。
- 错误响应格式。

### 9.6 ProduceContext

```python
class ProduceContext:
    agent_loop: AgentLoopSpec
    rollout_router: RolloutRouter

    async def generate_sample(self, state: RolloutState) -> RolloutState:
        endpoint = await self.rollout_router.acquire(state)

        if isinstance(self.agent_loop, ray.actor.ActorHandle):
            return await self.agent_loop.generate_sample.remote(state, rollout_url=endpoint.url)
        return await self.agent_loop.generate_sample(state, rollout_url=endpoint.url)

    async def generate_group(self, states: list[RolloutState], **kwargs) -> list[RolloutState]:
        tasks = []
        for state in states:
            tasks.append(create_task(self.generate_sample(state)))
        return await asyncio.gather(*tasks)
```

### 9.7 AgentLoop Actor

```python
class AgentLoopActor:
    async def generate_sample(self, rollout_state, **kwargs):
        return await self.agent_loop.generate_sample(rollout_state, **kwargs)

    async def generate_group(self, rollout_states, **kwargs):
        return await self.agent_loop.generate_group(rollout_states, **kwargs)
```

### 9.8 SingleTurnAgentLoop

`SingleTurnAgentLoop` 作为最小验证，把 remove_generate_v1 中直接 HTTP 请求后端的逻辑放到这里。

```python
class SingleTurnAgentLoop(AgentLoop):
    async def generate_sample(
        self,
        rollout_state: RolloutState,
        *,
        rollout_url: str,
        **kwargs,
    ) -> RolloutState:
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        payload = self._get_request_payload(rollout_state)
        for attempt in range(self.max_retry_per_sample + 1):
            http_result = await self._safe_post_request(
                url=f"{rollout_url.rstrip('/')}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.rollout_config.api_key}",
                },
                payload=payload,
            )
            if not self._should_retry(http_result, attempt):
                break

        rollout_state = await self._handle_http_result(rollout_state, http_result, **kwargs)
        if rollout_state.status != Status.COMPLETED:
            return rollout_state

        if self.judger is not None and not self.enable_batch_judge:
            rollout_state = await self.judger.judge(rollout_state)

        return rollout_state
```

## 10. Abort 设计

删除 Ray generate 后，abort 分为三层。

### 10.1 Producer 停止调度

`AgentLoopManager.pause_produce()` 设置 `_update_event`，`ProduceStrategy` 看到后停止 schedule 新 rollout task。

### 10.2 AgentLoop 停止等待当前请求

`SingleTurnAgentLoop.pause()` 保留 `_pause_event`：

- `_safe_post_request` 同时等待 HTTP 请求和 pause event。
- pause 先到时，尽量短时间等待 HTTP 返回；超时则取消 client-side task。
- 返回样本标记为 `Status.ABORTED` / `finish_reason="abort"`。
- pause/abort 优先级高于 retry；一旦 pause event 触发，不再发起下一次重试。

### 10.3 Rollout backend 中断生成

`RolloutController.pause_generation()` 保留，并广播：

```python
worker.pause_generation.remote()
```

`RolloutWorker.pause_generation()` 继续请求：

```text
POST {worker.server_url}/abort_request {"abort_all": true}
```

这部分不依赖 `RolloutWorker.generate`，因此删除 generate 不影响 abort。

## 11. 迁移步骤

### 步骤 1：新增 RolloutRouter

新增文件建议：

```text
xtuner/v1/rl/rollout/router.py
```

实现：

- `RolloutEndpoint`
- `RolloutRouterConfig`
- `UrlPoolRolloutRouter`
- `ThirdPartyRolloutRouter`
- `XTunerRolloutRouter`

第一阶段至少实现：

- `url_pool + worker`：用于 `SingleTurnAgentLoop` 最小验证。
- `third_party + session_server`：用于验证统一 URL 的 agentic 路径。

如果第三方 router 不满足 sticky/retry 需求，再实现 `xtuner + session_server`。

### 步骤 2：AgentLoopTaskConfig 接入 rollout_router_config

在 `AgentLoopTaskConfig` 中新增：

```python
rollout_router_config: RolloutRouterConfig
```

`AgentLoopManagerConfig.build()` 中构建 router，并放进 `_TaskRunner`。

### 步骤 3：ProduceContext 注入 rollout_url

`_TaskRunner` 增加 `rollout_router` 字段。

`_build_produce_context(...)` 把 router 放入 `ProduceContext`。

`ProduceContext.generate_group()` 改成对 group 内每个 sample 单独：

1. `endpoint = await rollout_router.acquire(state)`
2. `agent_loop.generate_sample(state, rollout_url=endpoint.url, ...)`

### 步骤 4：SingleTurnAgentLoop 改成 HTTP generate

把 remove_generate_v1 中迁移到 `SingleTurnAgentLoop` 的逻辑作为参考，保留：

- deterministic seed。
- request payload 构造。
- `/v1/chat/completions` 请求。
- response 解析。
- logprobs / routed_experts。
- partial rollout。
- retry。
- pause event。

去掉：

- 初始化时绑定 URL。
- agent loop 内部 URL router。

### 步骤 5：删除 Ray generate

删除：

- `RolloutController.generate`
- `RolloutWorker.generate`
- `RolloutWorker._get_request_payload`
- `RolloutWorker._safe_post_request`
- `RolloutWorker._safe_handle_response`
- controller 内仅服务 generate 的 parser 逻辑。
- controller 内仅服务 generate 的 `SessionRouter` 成员。
- `set_enable_partial_rollout` 如果只服务 worker generate，也同步删除或迁移。

保留：

- `pause_generation`
- `continue_generation`
- `cleanup_after_pause`
- `offload`
- `onload`
- `shutdown`
- `get_rollout_metadata`
- SessionServer 启停。

### 步骤 6：add_apiproxy 迁移

把 `rl_trainer.py` 中的 `add_apiproxy` 搬进 `ThirdPartyRolloutRouter`。

Trainer 不直接注册第三方 router，只负责 build config。

### 步骤 7：AgentInSandboxLoop / AgentInLocalhostLoop 接入 rollout_url

最小改动：

- `generate_sample(..., rollout_url=...)`
- `_run_item(item, rollout_url=...)`
- sandbox 通过 env 注入，例如 `XTUNER_ROLLOUT_BASE_URL`。
- localhost 通过 context 或 item field 注入。

这一步可以在 SingleTurn 验证后做。

## 12. 测试计划

### 12.1 Router 单测

覆盖：

- `router_type="url_pool", endpoint_type="worker"` 读取 `server_url_dict`。
- `router_type="url_pool", endpoint_type="session_server"` 读取 `worker_session_url_dict`。
- inactive URL 不返回。
- `sticky_session=True` 且 `session_uid` 存在时同 session 返回同 URL。
- `session_uid is None` 时 round-robin，且不修改 state。
- 无 active endpoint 时抛错。

### 12.2 ProduceContext 单测

覆盖：

- local AgentLoop 透传 `rollout_url`。
- Ray AgentLoopActor 透传 `rollout_url`。
- group 内每个 sample 独立 acquire。

### 12.3 SingleTurn 集成验证

覆盖：

- 不走 `RolloutController.generate`。
- `RolloutRouter(router_type="url_pool", endpoint_type="worker")` 返回 raw worker URL。
- `SingleTurnAgentLoop` 直接请求 `/v1/chat/completions`。
- `response_ids`、`logprobs`、`finish_reason` 正常写入。

### 12.4 Abort 验证

覆盖：

- `pause_produce()` 后不再 schedule 新 sample。
- 已发 HTTP 请求能被本地 pause event 退出。
- `RolloutController.pause_generation()` 仍会广播 worker `/abort_request`。
- aborted sample 状态为 `Status.ABORTED`。

## 13. 风险与待确认

### 13.1 SessionServer 与 SingleTurn

`SingleTurnAgentLoop` 默认应该走 `endpoint_type="worker"`。因为当前 `SessionServer` 在非 trace 请求下可能会清理 token/logprob 字段，不适合作为 SingleTurn 的默认入口。

后续如果希望 SingleTurn 也能走 SessionServer，需要确认 SessionServer 在非 trace 模式下是否保留：

- `return_token_ids=True`
- `return_logprob=True`
- `return_routed_experts=True`

### 13.2 第三方 router sticky 能力

如果第三方 router 不保证同 `session_id` 到同 backend，则不能用于强 sticky session 的 agentic 场景。

需要明确第三方 router 的能力：

- 是否读取 `session_id`。
- 是否保证 sticky。
- worker 失效时如何迁移或失败。

### 13.3 Agent 内部重试

`url_pool` 模式把具体 URL 交给 AgentLoop。agent 内部如果直接拿这个 URL 发请求，那么它的重试只能打到同一个 URL，
无法透明切换到其它 worker 或 SessionServer。

这对不同场景影响不同：

- `SingleTurnAgentLoop`：外层 `generate_sample` 可以失败后重新 acquire URL，因此不一定需要统一 router。
- agentic loop：一次 `generate_sample` 内部可能包含多轮请求和业务重试。若 agent 内部只知道具体 URL，则无法由 XTuner 接管重试路由。
- 强 session 场景：即使有统一 router，也不能随意跨 worker failover。只有在 router 能保证 session sticky、并能定义失败迁移语义时才安全。
- AgentLoop 应该支持有限 retry，但 retry 的目标 URL 不变。是否切换后端 endpoint 是 router 的职责。

因此 agentic 场景推荐：

1. 优先使用能保证 sticky session 和健康重试的 third-party router。
2. 如果 third-party router 不能满足语义，再实现 `XTunerRolloutRouter`。
3. 在没有统一 router 时，agent 内部 retry 只能视为“同 endpoint retry”，跨 endpoint failover 由外层样本重试处理。

### 13.4 健康检测与恢复

复杂 agentic 场景中，健康检测更适合由统一 router 管理：

- router 能在每次请求到来时读取最新 endpoint 状态。
- router 能集中维护失败计数、熔断、恢复和日志。
- router 能避免把请求继续发往已 inactive 的 worker/session_server。
- 外层 `url_pool.acquire()` 只能在 sample 开始前选择一次 URL，agent 内部多轮请求期间无法动态切换。

因此，`url_pool` 只能依赖 `RolloutController.get_rollout_metadata()` 中的状态快照做静态过滤；
`third_party` 或 `XTunerRolloutRouter` 才是后续健康检测、故障隔离和统一重试的主路径。

### 13.5 旧测试迁移成本

当前很多测试直接调用 `rollout_controller.generate.remote(...)`。删除 generate 后这些测试需要改写为：

- controller/worker 控制面测试。
- router 测试。
- SingleTurn HTTP generate 测试。

### 13.6 partial rollout 归属

删除 worker generate 后，partial rollout 逻辑应该归属 AgentLoop，而不是 Worker。

第一版只在 `SingleTurnAgentLoop` 中迁移验证；agentic loop 是否需要 partial rollout 后续单独设计。

## 14. 最小可验收结果

第一阶段完成后应满足：

1. 代码中没有 `RolloutController.generate` 和 `RolloutWorker.generate`。
2. `SingleTurnAgentLoop` 不通过 Ray generate，直接请求 `rollout_url/v1/chat/completions`。
3. `RolloutRouter(router_type="url_pool", endpoint_type="worker")` 能给 SingleTurn 分配 raw worker URL。
4. `RolloutRouter` 不修改 `RolloutState`。
5. 有 `session_uid` 且 sticky enabled 时，同 session 返回同 URL。
6. 没有 `session_uid` 时 round-robin。
7. abort 仍然通过 `RolloutController.pause_generation -> RolloutWorker.pause_generation -> /abort_request` 生效。
