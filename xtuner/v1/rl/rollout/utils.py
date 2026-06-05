import asyncio
import os
import threading
import time
from collections import OrderedDict
from itertools import cycle
from typing import TYPE_CHECKING, Any, Optional

import httpx
import ray

from xtuner.v1.rl.utils import asyncio_run
from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from .controller import RolloutControllerProxy, WorkerInfo
    from .worker import RolloutConfig, RolloutWorker

ROLLOUT_RAY_GET_TIMEOUT = int(os.getenv("XTUNER_ROLLOUT_RAY_GET_TIMEOUT", str(5 * 3600)))  # default 5 hours
logger = get_logger()


class SessionRouter:
    def __init__(
        self,
        worker_infos: dict[int, "WorkerInfo"],  # worker: worker_status
        worker_infos_lock: Optional[threading.RLock] = None,
        max_sessions: int = 10000,
        max_idle_seconds: Optional[float] = 3600.0,
    ):
        self._worker_infos = worker_infos
        self._worker_infos_lock = worker_infos_lock
        self._max_sessions = max_sessions
        self._max_idle = max_idle_seconds

        # OrderedDict: key=session_id -> value=(worker_rank, last_used_ts)
        self._map: OrderedDict[int, tuple[int, float]] = OrderedDict()

        self._worker_cycler = cycle(worker_infos.keys())
        self._lock = asyncio.Lock()
        self.logger = get_logger()

    def _now(self) -> float:
        return time.time()

    def _evict_expired(self):
        if self._max_idle is None:
            return
        now = self._now()

        to_delete = []
        for sid, (_, last_used) in self._map.items():
            if now - last_used > self._max_idle:
                to_delete.append(sid)
            else:
                break
        for sid in to_delete:
            self._map.pop(sid, None)

    def _evict_lru_to_capacity(self):
        while len(self._map) > self._max_sessions:
            self._map.popitem(last=False)

    def _choose_next_active_worker(self) -> tuple[int, Any]:
        n = len(self._worker_infos)
        for _ in range(n):
            rank = next(self._worker_cycler)
            if self._worker_infos_lock is None:
                info = self._worker_infos[rank]
                if info and info.is_active:
                    return rank, info.actor
            else:
                with self._worker_infos_lock:
                    info = self._worker_infos[rank]
                    if info and info.is_active:
                        return rank, info.actor
        return -1, None

    async def get_worker(self, session_id: int) -> Optional[Any]:
        async with self._lock:
            self._evict_expired()

            if session_id in self._map:
                worker_rank, _ = self._map.pop(session_id)
                if self._worker_infos_lock is None:
                    info = self._worker_infos.get(worker_rank)
                else:
                    with self._worker_infos_lock:
                        info = self._worker_infos.get(worker_rank)
                if info and info.is_active:
                    self._map[session_id] = (worker_rank, self._now())
                    return info.actor

            rank, worker = self._choose_next_active_worker()
            if rank == -1:
                return None
            self._map[session_id] = (rank, self._now())
            self._evict_lru_to_capacity()
            return worker


class RolloutHealthChecker:
    def __init__(
        self,
        config: "RolloutConfig",
        workers_info: dict[int, "WorkerInfo"],
        worker_infos_lock: Optional[threading.RLock] = None,
    ):
        self._workers_info = workers_info
        self._worker_infos_lock = worker_infos_lock
        self._check_interval = config.health_check_interval_seconds
        self._check_failure_threshold = config.health_check_failure_threshold
        self._stop_event: Optional[threading.Event] = None
        self._pause_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 启动时设置为暂停状态，开始generation后再调用restart方法恢复
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("RolloutHealthChecker started.")

    def stop(self) -> None:
        if not self._thread:
            return

        assert self._stop_event is not None
        self._stop_event.set()
        if self._pause_event:
            self._pause_event.clear()
        self._thread.join(timeout=5)
        self._thread = None
        self._stop_event = None
        logger.info("RolloutHealthChecker stopped.")

    def pause(self) -> None:
        if self._pause_event is None:
            return
        self._pause_event.set()
        logger.info("RolloutHealthChecker paused.")

    def resume(self) -> None:
        if self._pause_event is None:
            return
        self._pause_event.clear()
        logger.info("RolloutHealthChecker restarted.")

    def run_once(self) -> None:
        logger.debug("RolloutHealthChecker running health checks for all workers.")
        if self._worker_infos_lock is None:
            workers_snapshot = {
                rank: (info.actor, info.url, info.is_active) for rank, info in self._workers_info.items()
            }
        else:
            with self._worker_infos_lock:
                workers_snapshot = {
                    rank: (info.actor, info.url, info.is_active) for rank, info in self._workers_info.items()
                }

        tasks = [
            check_worker_health(
                actor,
                rank,
                url,
                is_active,
                self._check_failure_threshold,
            )
            for rank, (actor, url, is_active) in workers_snapshot.items()
        ]

        async def _run_checks() -> list[bool]:
            return await asyncio.gather(*tasks)

        check_results = asyncio_run(_run_checks())
        inactive_workers = []
        for rank, is_healthy in zip(workers_snapshot.keys(), check_results):
            if not is_healthy:
                logger.warning(f"Worker {rank} failed health check. Marking as inactive.")
                if self._worker_infos_lock is None:
                    self._workers_info[rank].is_active = False
                    inactive_worker = self._workers_info[rank].actor
                else:
                    with self._worker_infos_lock:
                        self._workers_info[rank].is_active = False
                        inactive_worker = self._workers_info[rank].actor
                if inactive_worker is None:
                    logger.error(f"[RolloutHealthChecker] Worker {rank} has no actor reference. Skipping shutdown.")
                    continue
                inactive_workers.append((rank, inactive_worker))
            else:
                logger.debug(f"[RolloutHealthChecker] Worker {rank} passed health check.")

        for rank, inactive_worker in inactive_workers:
            try:
                ray.get(inactive_worker.offload.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
                ray.get(inactive_worker.shutdown.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(f"Exception while shutting down worker {rank}: {e}")

    def _run_loop(self) -> None:
        assert self._stop_event is not None and self._pause_event is not None
        logger.info("RolloutHealthChecker loop started.")

        while not self._stop_event.is_set():
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self.run_once()

            if self._stop_event.wait(self._check_interval):
                break


async def send_abort_request(client: httpx.AsyncClient, url: str, timeout: float = 60.0) -> tuple[str, bool]:
    worker_url = f"{url}/abort_request"
    try:
        response = await client.post(worker_url, json={"abort_all": True}, timeout=timeout)
        response.raise_for_status()
        logger.debug(f"Successfully sent abort request to {url}")
        return url, True
    except Exception as e:
        logger.error(f"Failed to send abort request to {url}: {e}")
        return url, False


async def pause_generation(
    rollout_ctl: "RolloutControllerProxy",
    pause_time_out: float = 60.0,
    abort_retries: int = 3,
    abort_retry_interval_s: float = 30.0,
) -> None:
    """Pause rollout generation and abort all in-flight requests.

    Inference engines have been observed to occasionally drop a single
    abort_request call (request lost in transit, server busy on a long
    running batch, etc.), so the abort is sent ``abort_retries`` times
    with ``abort_retry_interval_s`` seconds between attempts. Every attempt
    fires the abort to every server in parallel; per-attempt success /
    failure is logged so a partial recovery can be diagnosed from logs.
    The total pause cost is therefore approximately
    ``(abort_retries - 1) * abort_retry_interval_s + last_request_latency``.

    Args:
        rollout_ctl: Controller actor proxy whose remote ``pause_generation``
            sets the worker-side ``receive_abort_request`` event.
        pause_time_out: HTTP timeout for each individual abort request.
        abort_retries: Total number of abort rounds (>=1). Default 3.
        abort_retry_interval_s: Delay between rounds. Default 30s.
    """
    await rollout_ctl.pause_generation.remote()  # type: ignore[attr-defined]
    rollout_ctl_metadata = await rollout_ctl.get_rollout_metadata.remote()  # type: ignore[attr-defined]
    infer_server_url = list(rollout_ctl_metadata["server_url_dict"].values())
    if abort_retries < 1:
        raise ValueError(f"abort_retries must be >= 1, got {abort_retries}.")

    async with httpx.AsyncClient() as client:
        for attempt in range(1, abort_retries + 1):
            tasks = [send_abort_request(client, url, timeout=pause_time_out) for url in infer_server_url]
            results = await asyncio.gather(*tasks)
            failed_workers = [url for url, success in results if not success]
            succeeded_count = len(infer_server_url) - len(failed_workers)

            if failed_workers:
                logger.warning(
                    f"Abort attempt {attempt}/{abort_retries} completed. "
                    f"Succeeded: {succeeded_count}, Failed: {len(failed_workers)}. "
                    f"Failed workers: {failed_workers}"
                )
            else:
                logger.info(
                    f"Abort attempt {attempt}/{abort_retries}: "
                    f"all {succeeded_count} abort requests sent successfully."
                )

            if attempt < abort_retries:
                await asyncio.sleep(abort_retry_interval_s)


async def continue_generation(rollout_ctl: "RolloutControllerProxy") -> None:
    return await rollout_ctl.continue_generation.remote()  # type: ignore[attr-defined]


async def check_worker_health(
    worker: "RolloutWorker", rank: int, url: str, is_active: bool, failure_threshold: int = 3
) -> bool:
    return True
    if worker is None or not is_active:
        logger.warning("Worker has no actor reference or is marked inactive.")
        return False
    failing_count = 0
    while failing_count < failure_threshold:
        try:
            health_status = await worker.check_health.remote()  # type: ignore[attr-defined]
            if health_status:
                return True
            failing_count += 1
            logger.warning(f"Health check failed for worker {rank} at {url}. Failure count: {failing_count}")
        except Exception as e:
            failing_count += 1
            logger.error(
                f"Exception during health check for worker {rank} at {url}: {e}. Failure count: {failing_count}"
            )
    return False
