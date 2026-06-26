from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import ray
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.utils.misc import check_chat_completions, delete_from_routedapiproxy, register_to_routedapiproxy
from xtuner.v1.utils import get_logger


RolloutRouterType = Literal["url_pool", "third_party", "xtuner"]
RolloutEndpointType = Literal["worker", "session_server"]


@dataclass(frozen=True)
class RolloutEndpoint:
    url: str
    endpoint_type: RolloutEndpointType


class RolloutRouter:
    endpoint_type: RolloutEndpointType

    async def acquire(self, rollout_state: RolloutState) -> RolloutEndpoint:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class RolloutRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    router_type: RolloutRouterType = "url_pool"
    endpoint_type: RolloutEndpointType = "worker"
    sticky_session: bool = True
    third_party_routed_url: str = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1"
    third_party_check_max_attempts: int = Field(default=5, ge=1)
    third_party_check_interval: float = Field(default=3.0, ge=0.0)

    def build(self, rollout_controller) -> RolloutRouter:
        if self.router_type == "url_pool":
            return UrlPoolRolloutRouter(
                rollout_controller=rollout_controller,
                endpoint_type=self.endpoint_type,
                sticky_session=self.sticky_session,
            )
        if self.router_type == "third_party":
            return ThirdPartyRolloutRouter(
                rollout_controller=rollout_controller,
                endpoint_type=self.endpoint_type,
                sticky_session=self.sticky_session,
                routed_url=self.third_party_routed_url,
                check_max_attempts=self.third_party_check_max_attempts,
                check_interval=self.third_party_check_interval,
            )
        if self.router_type == "xtuner":
            return XTunerRolloutRouter(
                rollout_controller=rollout_controller,
                endpoint_type=self.endpoint_type,
                sticky_session=self.sticky_session,
            )
        raise ValueError(f"Unsupported rollout router type: {self.router_type}")


def _get_rollout_metadata(rollout_controller) -> dict[str, Any]:
    get_metadata = rollout_controller.get_rollout_metadata
    if hasattr(get_metadata, "remote"):
        return ray.get(get_metadata.remote())  # type: ignore[attr-defined]
    return get_metadata()


def _sorted_url_items(url_dict: dict[Any, str]) -> list[tuple[int, str]]:
    return [(int(rank), url) for rank, url in sorted(url_dict.items(), key=lambda item: int(item[0]))]


class UrlPoolRolloutRouter(RolloutRouter):
    def __init__(
        self,
        rollout_controller,
        endpoint_type: RolloutEndpointType,
        sticky_session: bool = True,
        max_sessions: int = 10000,
    ) -> None:
        self.rollout_controller = rollout_controller
        self.endpoint_type = endpoint_type
        self.sticky_session = sticky_session
        self._max_sessions = max_sessions
        self._session_to_endpoint: OrderedDict[int, RolloutEndpoint] = OrderedDict()
        self._rr_index = 0
        self._lock = asyncio.Lock()

    def _load_active_endpoints(self) -> list[RolloutEndpoint]:
        metadata = _get_rollout_metadata(self.rollout_controller)
        if self.endpoint_type == "worker":
            url_dict = metadata["server_url_dict"]
            status_dict = metadata.get("worker_server_urls_status") or {}
        else:
            url_dict = metadata["worker_session_url_dict"]
            status_dict = metadata.get("worker_session_urls_status") or {}

        endpoints = []
        for _rank, url in _sorted_url_items(url_dict):
            if status_dict.get(url, True):
                endpoints.append(RolloutEndpoint(url=url, endpoint_type=self.endpoint_type))
        return endpoints

    async def acquire(self, rollout_state: RolloutState) -> RolloutEndpoint:
        session_uid = rollout_state.session_uid
        async with self._lock:
            endpoints = self._load_active_endpoints()
            if not endpoints:
                raise RuntimeError(f"No active rollout endpoint available for endpoint_type={self.endpoint_type}.")

            active_urls = {endpoint.url for endpoint in endpoints}
            if self.sticky_session and session_uid is not None and session_uid in self._session_to_endpoint:
                endpoint = self._session_to_endpoint.pop(session_uid)
                if endpoint.url in active_urls:
                    self._session_to_endpoint[session_uid] = endpoint
                    return endpoint

            endpoint = endpoints[self._rr_index % len(endpoints)]
            self._rr_index += 1
            if self.sticky_session and session_uid is not None:
                self._session_to_endpoint[session_uid] = endpoint
                while len(self._session_to_endpoint) > self._max_sessions:
                    self._session_to_endpoint.popitem(last=False)
            return endpoint


class ThirdPartyRolloutRouter(RolloutRouter):
    def __init__(
        self,
        rollout_controller,
        endpoint_type: RolloutEndpointType,
        sticky_session: bool,
        routed_url: str,
        check_max_attempts: int,
        check_interval: float,
    ) -> None:
        if endpoint_type != "session_server":
            raise ValueError("third_party rollout router only supports endpoint_type='session_server'.")
        self.rollout_controller = rollout_controller
        self.endpoint_type = endpoint_type
        self.sticky_session = sticky_session
        self.routed_url = routed_url
        self.check_max_attempts = check_max_attempts
        self.check_interval = check_interval
        self._started = False
        self._lock = asyncio.Lock()
        self.logger = get_logger()

    def _check_chat_completions_with_retry(self, base_url: str, model_name: str) -> bool:
        for attempt in range(1, self.check_max_attempts + 1):
            if check_chat_completions(base_url, model_name):
                return True
            if attempt < self.check_max_attempts:
                self.logger.warning(
                    f"check chat completions failed for {base_url}, "
                    f"retrying {attempt}/{self.check_max_attempts - 1} after {self.check_interval}s"
                )
                time.sleep(self.check_interval)
        return False

    def _start_once(self) -> None:
        metadata = _get_rollout_metadata(self.rollout_controller)
        model_name = metadata["rollout_config"].model_name
        delete_from_routedapiproxy(model_name)
        self.logger.info(f"deleted {model_name} from routedapiproxy")

        url_dict = metadata["worker_session_url_dict"]
        status_dict = metadata.get("worker_session_urls_status") or {}
        for _rank, url in _sorted_url_items(url_dict):
            if not status_dict.get(url, False):
                continue
            register_to_routedapiproxy(model_name, url)
            if not self._check_chat_completions_with_retry(url, model_name):
                raise RuntimeError(f"check chat completions failed for {url}")

        if not self._check_chat_completions_with_retry(self.routed_url, model_name):
            raise RuntimeError(f"check chat completions failed for routed URL {self.routed_url}")
        self._model_name = model_name
        self._started = True
        self.logger.info("registered rollout session servers to routedapiproxy")

    async def acquire(self, rollout_state: RolloutState) -> RolloutEndpoint:
        async with self._lock:
            if not self._started:
                self._start_once()
        return RolloutEndpoint(url=self.routed_url, endpoint_type=self.endpoint_type)

    async def close(self) -> None:
        model_name = getattr(self, "_model_name", None)
        if model_name is not None:
            delete_from_routedapiproxy(model_name)


class XTunerRolloutRouter(RolloutRouter):
    def __init__(
        self,
        rollout_controller,
        endpoint_type: RolloutEndpointType,
        sticky_session: bool = True,
    ) -> None:
        self.rollout_controller = rollout_controller
        self.endpoint_type = endpoint_type
        self.sticky_session = sticky_session

    async def acquire(self, rollout_state: RolloutState) -> RolloutEndpoint:
        raise NotImplementedError("XTunerRolloutRouter HTTP server is not implemented in this refactor.")
