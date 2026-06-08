from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController

from .agent_loop import AgentLoop, AgentLoopConfig
from .utils import PartialRolloutHandler


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    """Configuration for the built-in single-turn agent loop.

    ``SingleTurnAgentLoopConfig`` runs one model generation for each input
    ``RolloutState`` and optionally sends the completed output to a judger
    selected via ``rollout_state.data_source``. It is the default choice for
    math, QA, and other single-response RL tasks.

    Args:
        sample_params (SampleParams): Sampling parameters used by the rollout
            backend, such as temperature and maximum generation length.
        hf_checkpoint (str): Hugging Face checkpoint path used to identify the
            policy checkpoint for the agent loop.
        cpu_resources (CPUResourcesConfig | None): PG-external CPU resources
            used to run this agent loop as Ray actors. ``None`` runs the loop
            in local mode. Defaults to None.

    **Examples:**

    Example configuration for a single-turn task::

        config = SingleTurnAgentLoopConfig(
            sample_params=SampleParams(max_tokens=1024, temperature=1.0),
            hf_checkpoint="Qwen/Qwen3-8B",
        )
    """

    def build_local(
        self, rollout_controller, judgers: dict[str, Judger] | None = None, logger=None
    ) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judgers=judgers,
            logger=logger,
        )


class SingleTurnAgentLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judgers: dict[str, Judger] | None = None,
        logger=None,
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judgers, logger)
        self.max_tokens = self.sample_params.max_tokens
        self.partial_rollout_handler = PartialRolloutHandler(max_tokens=self.max_tokens)

    async def generate_sample(
        self,
        rollout_state: RolloutState,
        **kwargs,
    ) -> RolloutState:
        enable_partial_rollout = kwargs.get("enable_partial_rollout", False)

        # rollout state 预处理, enable_partial_rollout = True 会在这里拼接 token 和修正 max_token。
        # postprocess（含 routed_experts 同步 ray.get / ray.put）已下沉到 RolloutWorker.generate
        # 内执行，分散到每个 GPU worker，避免在中心化的 agent_loop 进程里串行成为瓶颈。
        rollout_state = self.partial_rollout_handler.preprocess(rollout_state, enable_partial_rollout)
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        # 推理引擎 generate；返回时 rollout_state 已经合并了 partial-rollout 历史。
        rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
        # 非 COMPLETED 状态（如被截断、放弃等）直接早退，不触发打分
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        judger = self._resolve_judger(rollout_state)
        # batch judger 由 producer 在 group 完成后统一打分，这里跳过
        if judger is not None and not judger.is_batch_judger:
            judged = await judger.judge([rollout_state])
            rollout_state = judged[0]
        return rollout_state

    def _resolve_judger(self, rollout_state: RolloutState) -> Judger | None:
        if not self.judgers:
            return None
        data_source = rollout_state.data_source
        if data_source is None:
            return None
        return self.judgers.get(data_source)
