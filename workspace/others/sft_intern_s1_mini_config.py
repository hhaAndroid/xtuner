from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig
from xtuner.v1.model import InternS1MiniConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os

ceph_config = os.environ.get('CEPH_CONFIG_PATH')
meta_data_path = os.environ['META_DATA_PATH']
work_dir = os.environ['WORK_DIR']
tokenizer_cache_dir = os.environ['TOKENIZER_CACHE_DIR']
global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', 8))

model_path = "/mnt/shared-storage-user/intern7shared/wangweiyun/OpenGVLab-rc1-hf/InternVL3_5-8B-CPT-HF"

# 训练超参数
sample_max_length = 32768
pack_max_length = 32768
min_num_frames = 8
max_num_frame = 36
total_step = 8000
hf_interval = 1000000
checkpoint_interval = 2000000
checkpoint_maxkeep = 10
lr = 8e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.1
recompute_ratio = 1.0
drop_path_rate = 0.0
loss_reduction = "square"

# model config
model_cfg = InternS1MiniConfig()

# dataset config
if ceph_config is not None:
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
else:
    oss_loader_cfg = None

ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          cache_tag='cache_tags_v1',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg,
                                                           max_length=sample_max_length,
                                                           max_dynamic_patch=_data.get('max_dynamic_patch',
                                                                                       None),
                                                           min_dynamic_patch=_data.get('min_dynamic_patch',
                                                                                       None),
                                                           min_num_frames=_data.get('min_num_frames', min_num_frames),
                                                           max_num_frames=_data.get('max_num_frames', max_num_frame),
                                                           data_augment=_data.get('data_augment', False),
                                                           system_message=_data.get('system_message', None),
                                                           hash=_data.get('hash', None),
                                                           oss_loader_cfg=oss_loader_cfg,
                                                           debug=True,
                                                           oss_time_log_thr=10
                                                           )
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=False,
    collator="intern_s1_vl_sft_collator",
    num_workers=8,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio, torch_compile=True,
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
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    work_dir=work_dir,
    # profile_step=10
)
