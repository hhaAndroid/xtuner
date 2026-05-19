import atexit
import signal
import subprocess
import time
from typing import TYPE_CHECKING, Optional, cast

import ray
import torch
from ray import ObjectRef

from xtuner.v1.utils.logger import get_logger

from .misc import find_free_ports


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RolloutState

    from .ray_accelerator_worker import AcceleratorType


logger = get_logger()
_partial_rollout_logger = get_logger("partial_rollout_postprocess")


@ray.remote
def find_master_addr_and_port(
    nums: int = 1, start_port: Optional[int] = None, end_port: Optional[int] = None
) -> tuple[str, int] | tuple[str, list[int]]:
    """Finds an available master address and a specified number of ports.

    This remote function gets the node's IP address and binds to one or more
    available ports, which can be used for distributed communication.

    Args:
        nums (int): The number of ports to find. Defaults to 1.
        start_port (Optional[int]): The starting port to search from.
            If None, random available ports will be used. Defaults to None.
        end_port (Optional[int]): The ending port to search to (exclusive).
            If start_port is None, this parameter is ignored. Defaults to None.

    Returns:
        A tuple containing the address and a single port if `nums` is 1,
        or a list of ports if `nums` is greater than 1.
    """
    addr = ray.util.get_node_ip_address()
    ports = find_free_ports(nums=nums, host="", start_port=start_port, end_port=end_port)

    if len(ports) == 1:
        return addr, ports[0]
    else:
        return addr, ports


@ray.remote
def get_accelerator_ids(accelerator: str) -> list:
    """Get the IDs of the available accelerators (GPUs, NPUs, etc.) in the Ray
    cluster."""
    return ray.get_runtime_context().get_accelerator_ids()[accelerator]


def get_ray_accelerator() -> "AcceleratorType":
    from xtuner.v1.utils.device import get_device

    """Get the type of accelerator available in the Ray environment.

    This function checks for the availability of CUDA and NPU devices and
    returns the corresponding accelerator type.

    Returns:
        AcceleratorType: The type of accelerator ("GPU" or "NPU").

    Raises:
        NotImplementedError: If neither CUDA nor NPU is available.
    """
    accelerator = None
    if get_device() == "cuda":
        accelerator = "GPU"
        return "GPU"
    else:
        try:
            import torch_npu  # noqa: F401

            accelerator = "NPU"
        except ImportError:
            pass

    if accelerator is None:
        raise NotImplementedError(
            "Supports only CUDA or NPU. If your device is CUDA or NPU, "
            "please make sure that your environmental settings are "
            "configured correctly."
        )

    return cast("AcceleratorType", accelerator)


def free_object_refs(refs: list[ObjectRef]) -> None:
    valid_refs: list[ObjectRef] = []
    seen: set[bytes] = set()
    for ref in refs:
        if not isinstance(ref, ObjectRef):
            continue
        try:
            binary_id = ref.binary()
        except Exception:
            valid_refs.append(ref)
            continue
        if binary_id in seen:
            continue
        seen.add(binary_id)
        valid_refs.append(ref)
    if not valid_refs:
        return

    try:
        ray._private.internal_api.free(valid_refs, local_only=False)
    except Exception:
        ray.internal.free(valid_refs, local_only=False)


def free_rollout_state_refs(rollout_state: "RolloutState") -> None:
    """Free trajectory-level plasma ObjectRefs on a discarded RolloutState.

    This only releases ``routed_experts`` (one ObjectRef per trajectory,
    written by the rollout engine). It deliberately does NOT touch
    ``mm_info["pixel_values"]`` — that ref is **prompt-level**: every
    deepcopy / pop_pending_keep / sibling trajectory of the same prompt
    shares the same plasma ID, and the same ID also lives on
    ``aggregation.original_prompt`` and on every COMPLETED trajectory
    already written to the replay buffer. Releasing it from any single
    trajectory's drop path would invalidate refs still in use by:

    * concurrent in-flight rollouts on other rollout workers,
    * sibling trajectories already accepted into the replay buffer and
      en route to the trainer.

    pixel_values cleanup is owned by ``RLController.fit`` after every
    train step (it walks ``packed_data_batches`` and frees once per
    referenced ID). Trajectories that never reach the trainer (STOPPED /
    FILTERED / orphan / FAILED) leak their share of ``pixel_values`` until
    the prompt has at least one trajectory that does reach training, at
    which point the controller's pass cleans the shared ID.
    """
    routed_experts = getattr(rollout_state, "routed_experts", None)
    if isinstance(routed_experts, ObjectRef):
        rollout_state.routed_experts = None
        free_object_refs([routed_experts])


def free_rollout_state_list_refs(rollout_states: "list[RolloutState]") -> None:
    """Batch version of :func:`free_rollout_state_refs`.

    Same trajectory-level-only semantics: pixel_values is intentionally
    left alone (see :func:`free_rollout_state_refs`).
    """
    refs: list[ObjectRef] = []
    for rollout_state in rollout_states:
        routed_experts = getattr(rollout_state, "routed_experts", None)
        if isinstance(routed_experts, ObjectRef):
            refs.append(routed_experts)
            rollout_state.routed_experts = None
    if refs:
        free_object_refs(refs)


def clear_rollout_response_for_rerun(rollout_state: "RolloutState") -> "RolloutState":
    routed_experts = getattr(rollout_state, "routed_experts", None)
    if isinstance(routed_experts, ObjectRef):
        free_object_refs([routed_experts])
    rollout_state.tokens = getattr(rollout_state, "prompt_ids", None)
    rollout_state.response = None
    rollout_state.response_ids = []
    rollout_state.logprobs = []
    rollout_state.routed_experts = None
    rollout_state.finish_reason = None
    rollout_state.response_mask = []
    rollout_state.response_model_steps = []
    rollout_state.reward = None
    rollout_state.error_msg = None
    return rollout_state


def _resolve_routed_experts_tensor(
    routed_experts: torch.Tensor | list[int] | ObjectRef,
) -> torch.Tensor:
    """Resolve ``routed_experts`` into a tensor without going through Python lists.

    The rollout engine writes a ``torch.Tensor`` into the object store, so
    callers can stay tensor-native end-to-end and avoid the legacy
    ``tolist()`` round-trip that dominates partial-rollout postprocess
    cost.
    """
    if isinstance(routed_experts, ObjectRef):
        routed_experts = ray.get(routed_experts)
    if isinstance(routed_experts, torch.Tensor):
        return routed_experts
    return torch.as_tensor(routed_experts)


def partial_rollout_postprocess(rollout_state: "RolloutState") -> "RolloutState":
    """Merge partial-rollout history into the freshly generated trajectory.

    Pop the ``history_response_dict`` set by
    :meth:`PartialRolloutHandler.preprocess`, concatenate the response /
    logprob / mask lists, and stitch ``routed_experts`` together using
    ``torch.cat``. This function is the heavy step in the partial-rollout
    pipeline: it does ``ray.get`` and ``ray.put`` on per-token routing
    tensors, both of which block the calling thread. It must run in the
    rollout worker that just produced ``rollout_state.routed_experts`` so
    the merge is distributed across all GPU workers (one ``ray.get`` for
    history + one ``ray.put`` for the merged tensor — never round-tripping
    cur through plasma). Run it via ``asyncio.to_thread`` from inside an
    event loop to avoid blocking concurrent requests in the same actor.

    The merged tensor is stored as a ``numpy.ndarray`` rather than a
    ``torch.Tensor``: Ray's CPU-tensor serializer round-trips through
    ``torch.save`` / ``torch.load`` and trips the ``UntypedStorage.dtype``
    attribute removed in newer PyTorch (legacy-load path). NumPy goes
    through Arrow zero-copy and the trainer's
    ``torch.as_tensor(arr, dtype=torch.long)`` is also zero-copy when the
    dtype already matches.

    Args:
        rollout_state (RolloutState): Trajectory returned by the rollout
            engine, possibly carrying a ``history_response_dict`` in
            ``extra_fields``.

    Returns:
        RolloutState: The same instance, with history merged in.
    """
    history_dict = rollout_state.extra_fields.pop("history_response_dict", None)
    if not history_dict:
        return rollout_state

    rollout_state.response_ids = history_dict.get("response_ids", []) + (rollout_state.response_ids or [])
    rollout_state.response = history_dict.get("response", "") + (rollout_state.response or "")
    rollout_state.logprobs = history_dict.get("logprobs", []) + (rollout_state.logprobs or [])
    rollout_state.response_mask = history_dict.get("response_mask", []) + (rollout_state.response_mask or [])
    history_routed_experts_ref = history_dict.get("routed_experts")
    cur_routed_experts_ref = rollout_state.routed_experts
    if history_routed_experts_ref is not None and cur_routed_experts_ref is not None:
        start_time = time.time()
        history_t = _resolve_routed_experts_tensor(history_routed_experts_ref)
        cur_t = _resolve_routed_experts_tensor(cur_routed_experts_ref)
        history_len = history_t.shape[0]
        cur_len = cur_t.shape[0]
        assert history_len - 1 <= cur_len, (
            f"Existing routed_experts len: {history_len}, current routed_experts len: {cur_len}"
        )
        # Slice cur to drop the prefix overlap with history; concat with
        # torch so we never round-trip through Python lists (the legacy
        # ``tolist`` + list concat path was the dominant cost — ~5s/call).
        # Then hand off to plasma as a numpy array so we avoid Ray's
        # torch-tensor pickle path (which can hit ``_legacy_load`` and the
        # ``UntypedStorage.dtype`` AttributeError on newer PyTorch).
        concat = torch.cat([history_t, cur_t[history_len:]], dim=0).contiguous()
        rollout_state.routed_experts = ray.put(concat.numpy())
        # NOTE on ref ownership: do NOT free history_ref / cur_ref here.
        # ``history_ref`` was lifted from ``history_response_dict`` but the
        # same ObjectRef value still lives on the trajectory in the
        # producer-side aggregator (pending_keep / completed copies); freeing
        # it here causes "object was manually freed" errors when the
        # aggregator pops the same trajectory in the next resume round.
        # ``cur_ref`` is the just-put ref from the rollout engine in this
        # process, but ray serialises ``rollout_state`` back to the producer
        # using the OLD ``cur_ref`` value before our reassignment took effect
        # in some code paths; freeing it racily invalidates that snapshot.
        # Old refs are reclaimed via the terminal-state cleanups
        # (aggregator.drop / replay_buffer.put on EXPIRED|FILTERED /
        # _dispatch_trajectory free_rollout_state_refs).
        elapsed = time.time() - start_time
        _partial_rollout_logger.info(
            f"routed_experts concatenation time: {elapsed:.4f}s "
            f"(history_len={history_len}, cur_len={cur_len}, concat_len={concat.shape[0]})"
        )
    elif history_routed_experts_ref is None and cur_routed_experts_ref is not None:
        rollout_state.routed_experts = cur_routed_experts_ref
    elif history_routed_experts_ref is not None and cur_routed_experts_ref is None:
        rollout_state.routed_experts = history_routed_experts_ref

    return rollout_state


def close_ray():
    """Clean up the ray resource."""
    # 1. Shutdown ray if initialized
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown successfully")
    except Exception as e:
        logger.warning(f"Error during ray.shutdown(): {e}")

    # 2. Stop ray launched by CLI
    try:
        result = subprocess.run(["ray", "stop", "--force"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning(f"Ray stop failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error stopping ray cluster: {e}")


def register_cleanup():
    """Register cleanup handlers for Ray on exit and signals."""
    _cleaned = False

    def cleanup_once():
        nonlocal _cleaned
        if not _cleaned:
            _cleaned = True
            close_ray()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_once()
        import sys

        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(cleanup_once)


def bind_train_rollout(
    train_workers,
    rollout_controller,
) -> None:
    """Bind the training and rollout workers for updating weights.

    This function retrieves rollout information from the rollout controller
    and distributes it to the training workers, enabling them to update the
    rollout models' weights.

    Args:
        train_workers: A list of training worker actors.
        rollout_controller: The rollout controller actor.
    """
    info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())  # type: ignore[attr-defined]
    ray.get([worker.update_rollout_info.remote(**info_dict) for worker in train_workers])  # type: ignore[attr-defined]
    return
