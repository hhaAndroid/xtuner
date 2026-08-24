from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.model import Qwen3Dense8BConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
import json
import os

meta_data_path = os.environ['META_DATA_PATH']
model_path = os.environ['MODEL_PATH']
work_dir = os.environ['WORK_DIR']
global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', 8))

sample_max_length = 32768
pack_max_length = 32768
total_step = 100
lr = 8e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.1
recompute_ratio = 1.0
loss_reduction = "square"

model_cfg = Qwen3Dense8BConfig(compile_cfg=False)

ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0)
                                          ),
                 "tokenize_fn": OpenaiTokenizeFunctionConfig(
                                chat_template='qwen3',
                                max_length=sample_max_length)
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    num_workers=8,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio, 
                      checkpoint_preserve_rng_state=False)

resume_cfg = ResumeConfig(auto_resume=False)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker='tensorboard',
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_step=total_step,
    work_dir=work_dir
)
