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
_SP_ULYESS_MESH = None
_SP_RING_MESH = None
_SP_ULYESS_GROUP = None
_SP_RING_GROUP = None
_SP_WORLD_SIZE = None
_SP_ULYESS_WORLD_SIZE = None
_SP_RING_WORLD_SIZE = None


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


def setup_parallel(sp_size=1, tp_size=1, sp_ring_degree=1):
    assert tp_size == 1

    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    device = get_device()

    world_size = dist.get_world_size()
    assert world_size % sp_size == 0
    assert world_size % sp_size % tp_size == 0
    assert tp_size <= 8

    if tp_size > 1:  # TODO: 临时方案
        pass
        # 如果不注释，global 会出问题，虽然这个分支永远进不去，非常奇怪
        # dp_size = world_size // sp_size // tp_size
        # data_mesh = init_device_mesh(
        #     device, (dp_size, sp_size, tp_size), mesh_dim_names=('dp', 'sp', 'tp'))
        #
        # model_mesh = init_device_mesh(
        #     device, (dp_size * sp_size, tp_size), mesh_dim_names=('fsdp', 'tp'))
        #
        # world_mesh = init_device_mesh(
        #     device, (world_size,), mesh_dim_names=('world',))
        #
        # global _DP_MESH, _DP_GROUP, _DP_WORLD_SIZE
        # _DP_MESH = data_mesh['dp']
        # _DP_GROUP = data_mesh['dp'].get_group()
        # _DP_WORLD_SIZE = data_mesh['dp'].size()
        #
        # global _SP_MESH, _SP_GROUP, _SP_WORLD_SIZE
        # _SP_MESH = data_mesh['sp']
        # _SP_GROUP = data_mesh['sp'].get_group()
        # _SP_WORLD_SIZE = data_mesh['sp'].size()
        #
        # global _TP_MESH, _TP_GROUP, _TP_WORLD_SIZE
        # _TP_MESH = model_mesh['tp']
        # _TP_GROUP = model_mesh['tp'].get_group()
        # _TP_WORLD_SIZE = model_mesh['tp'].size()
        #
        # global _WORLD_MESH, _FSDP_MESH
        # _WORLD_MESH = world_mesh['world']
        # _FSDP_MESH = model_mesh['fsdp']
    else:
        dp_size = world_size // sp_size

        assert sp_size % sp_ring_degree == 0
        sp_ulysses_degree = sp_size // sp_ring_degree

        global_device_mesh = init_device_mesh(
            'cuda', (dp_size, sp_size), mesh_dim_names=('dp', 'sp'))
        # TODO HHA: 非常关键，顺序不能错，不能是 (dp_size, sp_ulysses_degree， sp_ring_degree)
        device_mesh = init_device_mesh(
            'cuda', (dp_size, sp_ring_degree, sp_ulysses_degree), mesh_dim_names=('dp', 'sp_ring', 'sp_ulysses'))
        global _DP_MESH, _SP_ULYESS_MESH, _SP_RING_MESH, _SP_MESH
        _DP_MESH = device_mesh['dp']
        _SP_ULYESS_MESH = device_mesh['sp_ulysses']
        _SP_RING_MESH = device_mesh['sp_ring']
        _SP_MESH = global_device_mesh['sp']

        global _DP_GROUP, _SP_ULYESS_GROUP, _SP_RING_GROUP, _SP_GROUP
        _DP_GROUP = device_mesh.get_group('dp')
        _SP_ULYESS_GROUP = device_mesh.get_group('sp_ulysses')
        _SP_RING_GROUP = device_mesh.get_group('sp_ring')
        _SP_GROUP = global_device_mesh.get_group('sp')

        model_mesh = init_device_mesh(
            device, (dp_size * sp_size, tp_size), mesh_dim_names=('fsdp', 'tp'))

        world_mesh = init_device_mesh(
            device, (world_size,), mesh_dim_names=('world',))

        global _TP_MESH, _TP_GROUP, _TP_WORLD_SIZE
        _TP_MESH = model_mesh['tp']
        _TP_GROUP = model_mesh['tp'].get_group()
        _TP_WORLD_SIZE = model_mesh['tp'].size()

        global _WORLD_MESH, _FSDP_MESH
        _WORLD_MESH = world_mesh['world']
        _FSDP_MESH = model_mesh['fsdp']


def get_sp_world_size():
    global _SP_WORLD_SIZE
    if _SP_WORLD_SIZE is not None:
        return _SP_WORLD_SIZE
    if not dist.is_initialized() or (_SP_GROUP is None):
        _SP_WORLD_SIZE = 1
    else:
        _SP_WORLD_SIZE = dist.get_world_size(_SP_GROUP)
    return _SP_WORLD_SIZE


def get_ulysess_mesh():
    return _SP_ULYESS_MESH


def get_ring_mesh():
    return _SP_RING_MESH


def get_ulysess_group():
    return _SP_ULYESS_GROUP


def get_ring_group():
    return _SP_RING_GROUP


def get_ulysess_world_size():
    global _SP_ULYESS_WORLD_SIZE
    if _SP_ULYESS_WORLD_SIZE is not None:
        return _SP_ULYESS_WORLD_SIZE
    if not dist.is_initialized() or (_SP_ULYESS_GROUP is None):
        _SP_ULYESS_WORLD_SIZE = 1
    else:
        _SP_ULYESS_WORLD_SIZE = dist.get_world_size(_SP_ULYESS_GROUP)
    return _SP_ULYESS_WORLD_SIZE


def get_ring_world_size():
    global _SP_RING_WORLD_SIZE
    if _SP_RING_WORLD_SIZE is not None:
        return _SP_RING_WORLD_SIZE
    if not dist.is_initialized() or (_SP_RING_GROUP is None):
        _SP_RING_WORLD_SIZE = 1
    else:
        _SP_RING_WORLD_SIZE = dist.get_world_size(_SP_RING_GROUP)
    return _SP_RING_WORLD_SIZE


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


def get_sp_group():
    return _SP_GROUP


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
