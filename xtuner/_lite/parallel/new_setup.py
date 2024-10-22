import torch.distributed as dist
from mmengine.dist import infer_launcher, init_dist
from torch.distributed.device_mesh import init_device_mesh
import torch
import time
from contextlib import contextmanager
from xtuner._lite import get_logger

logger = get_logger()

_SP_MESH = None
_DP_MESH = None
_TP_MESH = None
_FSDP_MESH = None
_WORLD_MESH = None


def get_device():
    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        try:
            import torch_npu  # noqa: F401
            device = 'npu'
        except ImportError:
            pass

    if device is None:
        raise NotImplementedError(
            'Supports only CUDA or NPU. If your device is CUDA or NPU, '
            'please make sure that your environmental settings are '
            'configured correctly.')

    return device


def setup_parallel(sp_size=1, tp_size=1):

    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    device = get_device()

    world_size = dist.get_world_size()
    assert world_size % sp_size == 0
    assert world_size % sp_size % tp_size == 0
    assert tp_size <= 8

    dp_size = world_size // sp_size // tp_size
    data_mesh = init_device_mesh(
        device, (dp_size, sp_size, tp_size), mesh_dim_names=('dp', 'sp', 'tp'))

    model_mesh = init_device_mesh(
        device, (dp_size * sp_size, tp_size), mesh_dim_names=('fsdp', 'tp'))

    world_mesh = init_device_mesh(
        device, (world_size, ), mesh_dim_names=('world', ))

    global _DP_MESH, _DP_GROUP, _DP_WORLD_SIZE
    _DP_MESH = data_mesh['dp']
    _DP_GROUP = data_mesh['dp'].get_group()
    _DP_WORLD_SIZE = data_mesh['dp'].size()

    global _SP_MESH, _SP_GROUP, _SP_WORLD_SIZE
    _SP_MESH = data_mesh['sp']
    _SP_GROUP = data_mesh['sp'].get_group()
    _SP_WORLD_SIZE = data_mesh['sp'].size()

    global _TP_MESH, _TP_GROUP, _TP_WORLD_SIZE
    _TP_MESH = model_mesh['tp']
    _TP_GROUP = model_mesh['tp'].get_group()
    _TP_WORLD_SIZE = model_mesh['tp'].size()

    global _WORLD_MESH, _FSDP_MESH
    _WORLD_MESH = world_mesh['world']
    _FSDP_MESH = model_mesh['fsdp']


def get_world_mesh():
    return _WORLD_MESH


def get_dp_mesh():
    return _DP_MESH


def get_fsdp_mesh():
    return _FSDP_MESH


def get_sp_mesh():
    return _SP_MESH


def get_tp_mesh():
    return _TP_MESH


def get_torch_device_module():

    device = get_device()
    if device == 'cuda':
        return torch.cuda
    elif device == 'npu':
        return torch.npu
    else:
        raise NotImplementedError


@contextmanager
def profile_time_and_memory(desc):

    torch_device = get_torch_device_module()
    start_t = time.time()
    torch_device.reset_peak_memory_stats()

    yield

    max_memory = torch_device.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.success(f'{desc} Elapsed time {cost_time:.2f} seconds, '
                f'peak gpu memory {max_memory/1024**3:.1f}G')
