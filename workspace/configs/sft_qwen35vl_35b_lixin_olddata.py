from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os
import shutil
from pathlib import Path
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import QWEN3VL_COMPILE_CFG
QWEN3VL_COMPILE_CFG.pop("xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward")

# 路径配置
# ceph_config = "/mnt/shared-storage-user/huanghaian/petreloss.conf"
# meta_data_path = '/mnt/shared-storage-user/gaozhangwei/workspace_glx/data_analysis_tools/tiny_sample_project/outputs/interns1_1_base02_20260120b_tiny/meta_json/interns1_1_base02_20260120b_tiny_rollout_v2.json'
# # model_path = '/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2_preview/decay/interns2_preview_verify_decay_20260304d_32k_1_0/20260317231623/hf-12000'
# model_path = "/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"
# work_dir = os.environ['WORK_DIR']
# tokenizer_cache_dir = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_tokenizer_cache/interns2_preview/slow_tokenize_sft_ml_256k_tokenizer_rollout_v2"

ceph_config = os.environ.get('CEPH_CONFIG_PATH')
meta_data_path = os.environ['META_DATA_PATH']
work_dir = os.environ['WORK_DIR']
tokenizer_cache_dir = os.environ['TOKENIZER_CACHE_DIR']
global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', 8))

model_path = "/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"

# 将当前配置文件拷贝到work_dir
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
current_file = __file__
shutil.copy(current_file, work_dir)

# 训练超参数
sample_max_length = 256 * 1024
pack_max_length = 256 * 1024
rand_video_max_frames = 24
num_workers = 8
global_batch_size = 32
total_epoch = 1
hf_interval = 400
hf_max_keep = 10
checkpoint_interval = 400
checkpoint_maxkeep = 5
lr = 2e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.1
recompute_ratio = 1.0
loss_reduction = "square"
max_pixels = 16777216  # 16384 * 32 * 32

ep_size = 1
sp_size = 4
enable_fp8 = False
torch_compile = True

# model config
fp8_config= Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE, scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE)

model_cfg = Qwen3_5_VLMoE35BA3Config()
# with (Path(model_path) / "config.json").open() as f:
#     model_hf_config = json.load(f)
# model_cfg.text_config.vocab_size = model_hf_config["text_config"]["vocab_size"]  

if enable_fp8:
    model_cfg.text_config.float8_cfg = fp8_config

if ep_size > 1:
    model_cfg.text_config.ep_size = ep_size
    model_cfg.text_config.dispatcher='deepep'

# dataset config
if ceph_config is not None:
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
else:
    oss_loader_cfg = None

ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    tokenize_fn = Qwen3VLTokenizeFnConfig(
        llm_pack_weight=-3.2,
        visual_pack_weight=5.0,
        max_length=sample_max_length,
        processor_path=model_path,
        rand_video_max_frames=rand_video_max_frames,
        oss_loader_cfg=oss_loader_cfg,
        max_pixels=max_pixels
    )

    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          enable_sequential_sampler=True,  # 为了保证可复现性，使用顺序采样，入表的数据已经全局shuffle
                                          cache_tag='cache_tags_v1',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": tokenize_fn
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level='soft',
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
    # tokenizer_hash='4bc329f24e7ea505',
)
# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio,
                      torch_compile=torch_compile,
                      ep_size=ep_size,
                      checkpoint_preserve_rng_state=False)

resume_cfg = ResumeConfig(auto_resume=True)

# trainer config
trainer = TrainerConfig(
    sp_size=sp_size,
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
)
