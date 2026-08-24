import ray
import torch
import torch.distributed as dist
import os
from ray.util.placement_group import (
    VALID_PLACEMENT_GROUP_STRATEGIES,
    PlacementGroup,
    placement_group,
    placement_group_table,
)
import socket
from xtuner.v1.ray.base import AutoAcceleratorWorkers
from xtuner.v1.ray.base import SingleAcceleratorWorker
from ray.actor import ActorClass, ActorProxy

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from datetime import timedelta
from typing import Any
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_custom_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


ray.init(num_gpus=8)

bundles = [
            {
                "CPU": 1,
                'GPU': 1,
            }
    ] * 4


rollout_pg = placement_group(bundles=bundles, strategy='PACK', name='rollout_pg')
ray.get(rollout_pg.ready(), timeout=10)

train_pg = placement_group(bundles=bundles, strategy='PACK', name='train_pg')
ray.get(train_pg.ready(), timeout=10)


class TrainWorker(SingleAcceleratorWorker):
    def __init__(
        self,
        config: dict,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        device_type=None,
    ):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(ray.get_runtime_context().get_accelerator_ids()['GPU'][0])
        backend = "nccl"
        dist.init_process_group(
            backend=backend,
            init_method="env://",  # 这告诉 PyTorch 从环境变量读取配置
        )
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        self.rollout_device_mesh = DeviceMesh(
                    "cpu", mesh=[[0],[1],[2],[3]], mesh_dim_names=("engine_instance", "engine_parallel")
                )
        self.new_group=None
    
    def test_all_reduce(self):
        tensor = torch.tensor([1.0], device=torch.accelerator.current_accelerator())
        dist.all_reduce(tensor)
        print('All-reduce result:', tensor.item())
        return tensor
    
    def init_new_group(self, rollout_workers):
        _is_src_rank = dist.get_rank() == 0
        if _is_src_rank:
            self.rollout_workers = rollout_workers

            print("[TrainWorker]Initializing new process group from rank 0")
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            refs = [
                    engine.init_new_group.remote(
                        master_address,
                        master_port,
                        i + 1,
                        5,
                        "new_group",
                        backend="nccl",
                    )
                    for i, engine in enumerate(rollout_workers)
            ]
        
            self.new_group = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=5,
                rank=0,
                group_name='new_group',
            )
            ray.get(refs)
            print(f"[TrainWorker]New process group initialized at rank 0 {self.new_group}")

    def test_broadcast(self):
        _is_src_rank = dist.get_rank() == 0
        if _is_src_rank:

            tensor = torch.randn((4,5), dtype=torch.bfloat16, device=torch.accelerator.current_accelerator())

            state_dict = {'aa':tensor}
            dtypes=[param.dtype for _, param in state_dict.items()]
            shapes=[param.shape for _, param in state_dict.items()]
            dtypes=[str(dtype).replace("torch.", "") for dtype in dtypes]

            ref = [engine.test_broadcast.remote(dtypes, shapes) for engine in self.rollout_workers]
            
            print(f"[TrainWorker]Broadcasting from rank 0 {shapes}, {tensor}")
            dist.broadcast(tensor, src=0, group=self.new_group)
            print('[TrainWorker] Broadcast result:', tensor)

            ray.get(ref)

            return tensor
        
    def ready(self) -> bool:
        return True


class RolloutWorker(SingleAcceleratorWorker):
    def __init__(
        self,
        config: dict,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        device_type=None,
    ):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(ray.get_runtime_context().get_accelerator_ids()['GPU'][0])
        backend = "nccl"
        dist.init_process_group(
            backend=backend,
            init_method="env://",  # 这告诉 PyTorch 从环境变量读取配置
        )
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))

        self.new_group=None
    
    def test_all_reduce(self):
        tensor = torch.tensor([1.0], device=torch.accelerator.current_accelerator())
        dist.all_reduce(tensor)
        print('All-reduce result:', tensor.item())
        return tensor
    
    def init_new_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        print("[RolloutWorker]Initializing new process group from rank", dist.get_rank())
        self.new_group = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank_offset,
                group_name=group_name,
            )
        print(f"[RolloutWorker]New process group initialized at rank {self.new_group}", dist.get_rank())

    def test_broadcast(self, dtypes, shapes):
        for dtype, shape in zip(dtypes, shapes):
            target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
            weight = torch.empty(shape, dtype=target_dtype, device=torch.accelerator.current_accelerator())
            print(f"[RolloutWorker]Receiving broadcast at rank ", dist.get_rank())
            dist.broadcast(weight, src=0, group=self.new_group)
            print('[RolloutWorker] Broadcast result:', weight)

    def ready(self) -> bool:
        return True


RayTrainWorker = ray.remote(
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                    }
                },
            )(TrainWorker)

train_workers, _ = AutoAcceleratorWorkers.from_placement_group(RayTrainWorker, {"a":1}, train_pg)
ray.wait([worker.ready.remote() for worker in train_workers])
# ray.get([w.test_all_reduce.remote() for w in train_workers])

RayRolloutWorker = ray.remote(
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                    }
                },
            )(RolloutWorker)

rollout_workers, _ = AutoAcceleratorWorkers.from_placement_group(RayRolloutWorker, {"a":1}, rollout_pg)
ray.wait([worker.ready.remote() for worker in rollout_workers])
# ray.get([w.test_all_reduce.remote() for w in rollout_workers])

ray.get([w.init_new_group.remote(rollout_workers) for w in train_workers])
ray.get([w.test_broadcast.remote() for w in train_workers])
