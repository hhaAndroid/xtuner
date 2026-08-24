import os
import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
logger.setLevel('INFO')
import ray
import numpy as np


class TensorSaver:
    def __init__(self, workdir: str, load_dir=None, enable_replace: int = 0):
        ray_devices = ray.get_gpu_ids()
        logger.info(f'TensorSaver, device={torch.cuda.current_device()} ray device={ray_devices} workdir={workdir}, load_dir={load_dir}, ')
        if workdir:
            os.makedirs(workdir, exist_ok=True)
        self.workdir = workdir
        self._step = -1
        self._enable = False
        self._load_dir = load_dir
        self._enable_replace = enable_replace  == 1
        self.is_rank0 = False
    
    def set_rank0(self):
        logger.info(f'Set to rank0')
        self.is_rank0 = True

    def activate(self):
        if not self.workdir:
            return
        
        self._enable = True
    
    def deactivate(self):
        if not self.workdir:
            return

        self._enable = False

    def load_ref_tensor(self, name):
        file = os.path.join(self._load_dir, f'step0.{name}.pt')
        assert os.path.exists(file), file
        return torch.load(file, weights_only=False) 
    
    def replace_tensor(self, name, src_tensor):
        if not self._enable or not self._enable_replace:
            return src_tensor
        torch.cuda.synchronize()
        if self._load_dir and os.path.exists(self._load_dir):
            load_tensor = self.load_ref_tensor(name)
            load_tensor = load_tensor.to(src_tensor)
            new_load_tensor = load_tensor
            if src_tensor.ndim == 3:
                new_load_tensor = load_tensor[:, :src_tensor.shape[1]]
            elif src_tensor.ndim == 2:
                if load_tensor.ndim == 3:
                    load_tensor = load_tensor.squeeze(0)
                new_load_tensor = load_tensor[:src_tensor.shape[0]]
            if new_load_tensor.shape != src_tensor.shape:
                logger.error(f'{src_tensor.shape}, {new_load_tensor.shape}, {load_tensor.shape}')
            src_tensor.copy_(new_load_tensor)
            return src_tensor.contiguous()
        else:
            logger.info(f'load dir not exists{self._load_dir}')

    def step(self):
        if not self._enable:
            return
        if not self.is_rank0:
            return

        self._step += 1
        
    def save(self, name, tensor):
        if not self._enable:
            return
        if not self.is_rank0:
            return
        file = os.path.join(self.workdir, f'step{self._step}.{name}.pt')
        torch.save(tensor.clone().cpu(), file)
        logger.info(f'save tensor={tensor.shape} {tensor.dtype} to {file=}')
