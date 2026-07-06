from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.utils import clear_rollout_response_for_rerun
from xtuner.v1.utils import get_logger


class PartialRolloutHandler:
    """Preprocess rollout state for partial-rollout continuation.

    Postprocessing (history merge + routed-experts stitching) lives in the
    rollout worker process — see
    :func:`xtuner.v1.rl.utils.partial_rollout_postprocess`. Running it inside
    each ``RolloutWorker.generate`` distributes the heavy ``ray.get`` /
    ``ray.put`` cost across all GPU workers and lets the merge reuse the
    cur tensor that the rollout engine just produced locally, without
    bottlenecking a single agent_loop process.
    """

    def __init__(self, max_tokens: int) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.max_tokens = max_tokens

    def preprocess(self, rollout_state: RolloutState, enable_partial_rollout: bool = False) -> RolloutState:
        # Per-request output budget: the total token budget (``max_tokens``) minus the
        # initial prompt length, so longer prompts get a correspondingly smaller
        # generation allowance instead of a fixed value shared across all requests.
        prompt_ids = list(rollout_state.prompt_ids or [])
        prompt_len = len(prompt_ids)
        request_max_tokens = self.max_tokens - prompt_len

        if rollout_state.status == Status.EXPIRED or (
            not enable_partial_rollout and rollout_state.status == Status.ABORTED
        ):
            rollout_state = clear_rollout_response_for_rerun(rollout_state)
            rollout_state.sample_params = rollout_state.sample_params.model_copy(
                update={"max_tokens": request_max_tokens}
            )
            rollout_state.response = ""
            rollout_state.status = Status.INIT

        if not rollout_state.response_ids or rollout_state.status == Status.COMPLETED:
            # Fresh (or already completed) request: apply the per-request output budget.
            rollout_state.sample_params = rollout_state.sample_params.model_copy(
                update={"max_tokens": request_max_tokens}
            )
            return rollout_state

        # Set up token and length variable
        response_ids = rollout_state.response_ids
        response_len = len(response_ids)

        rollout_state.tokens = prompt_ids + response_ids  # concatenate for partial rollout continuation
        # Remaining generation budget = per-request output budget minus tokens already produced.
        remaining_tokens = request_max_tokens - response_len
        rollout_state.sample_params = rollout_state.sample_params.model_copy(update={"max_tokens": remaining_tokens})

        self.logger.debug(
            f"[PartialRolloutHandler] Sample {rollout_state.uid} continue rollout | Remaining tokens allowed: {remaining_tokens} | Status: {rollout_state.status} | Prompt len: {prompt_len} | Response len: {response_len} | Staleness: {rollout_state.seq_staleness} | Total tokens: {len(rollout_state.tokens)}"
        )
        # The rollout worker reads ``history_response_dict`` after generation
        # and merges it back into rollout_state via partial_rollout_postprocess.
        rollout_state.extra_fields["history_response_dict"] = {
            "response_ids": rollout_state.tokens[prompt_len:] if rollout_state.tokens else [],
            "response": rollout_state.response or "",
            "logprobs": rollout_state.logprobs or [],
            "response_mask": rollout_state.response_mask or [],
            "routed_experts": rollout_state.routed_experts,
        }
        return rollout_state
