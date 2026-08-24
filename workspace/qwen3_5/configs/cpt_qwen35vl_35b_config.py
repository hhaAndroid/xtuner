from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig, PretrainTokenizeFunctionConfig
from xtuner.v1.model import Qwen3VLMoE235BA22Config
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config

from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import QWEN3VL_COMPILE_CFG
QWEN3VL_COMPILE_CFG.pop("xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward")

ceph_config = os.environ.get('CEPH_CONFIG_PATH')
meta_data_path = os.environ['META_DATA_PATH']
work_dir = os.environ['WORK_DIR']
tokenizer_cache_dir = os.environ['TOKENIZER_CACHE_DIR']
global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', 8))

model_path = "/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"

# 训练超参数
sample_max_length = 32768
pack_max_length = 32768
processor_path = model_path
rand_video_max_frames = 24
add_vision_id = True
num_workers = 8
total_epoch = 1
hf_interval = 1000000
hf_max_keep = 2
checkpoint_interval = 1000000
checkpoint_maxkeep = 2
lr = 1e-5
lr_min = 1e-5
weight_decay = 0.05
warmup_ratio = 0.03
recompute_ratio = 1.0
loss_reduction = "square"

sp_size = 2
ep_size = 1
enable_fp8=False
torch_compile=False

# model config
model_cfg = Qwen3_5_VLMoE35BA3Config(only_llm_forward=False)

fp8_config= Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE, scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE)
if enable_fp8:
    model_cfg.text_config.float8_cfg=fp8_config

if ep_size>1:
    model_cfg.text_config.ep_size = ep_size
    model_cfg.text_config.dispatcher='deepep'

# dataset config
if ceph_config is not None:
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
else:
    oss_loader_cfg = None


has_pretrain = False
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    if _data.get('text_pretrain', False) or has_pretrain:
        has_pretrain = True

    # VLMJsonlDataset -> soft_pack 多个样本pack到一起，不会对样本内部进行拆分处理
    # JsonlDataset -> hard_pack 多个样本pack到一起，样本过长，会对样本内部进行拆分处理，目前只用于处理纯文本预训练数据
    # 可能存在问题的点：纯文本预训练数据不会和其他数据一起pack到一起
    class_name = 'JsonlDataset' if _data.get('text_pretrain', False) else 'VLMJsonlDataset'

    if _data.get('text_pretrain', False):
        tokenize_fn = PretrainTokenizeFunctionConfig(hash=_data.get('hash', None))
    else:
        tokenize_fn = Qwen3VLTokenizeFnConfig(
            max_length=sample_max_length,
            processor_path=processor_path,
            video_min_total_pixels=_data.get('video_min_total_pixels', None),
            video_max_total_pixels=_data.get('video_max_total_pixels', None),
            video_min_frames=_data.get('video_min_frames', None),
            video_max_frames=_data.get('video_max_frames', None),
            fps=_data.get('fps', None),
            rand_video_max_frames=rand_video_max_frames,
            add_vision_id=add_vision_id,
            system_message=_data.get('system_message', None),
            hash=_data.get('hash', None),
            oss_loader_cfg=oss_loader_cfg
        )

    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name=class_name,
                                          enable_sequential_sampler=True,
                                          cache_tag='cache_tags_v1',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": tokenize_fn
                 }
    dataset_config.append(_data_cfg)

if has_pretrain:
    pack_level = 'mllm_hybrid'
else:
    pack_level = 'soft'

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level=pack_level,
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)
# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio,
                      ep_size=ep_size,
                      torch_compile=torch_compile,
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
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
    # profile_step=[20, 30, 40]
)
