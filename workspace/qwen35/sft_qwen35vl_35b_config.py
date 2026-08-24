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
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from pathlib import Path

ceph_config = os.environ.get('CEPH_CONFIG_PATH')
meta_data_path = os.environ['META_DATA_PATH']
work_dir = os.environ['WORK_DIR']
tokenizer_cache_dir = os.environ['TOKENIZER_CACHE_DIR']
global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', 8))

model_path = "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
# model_path = '/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2_preview/cpt/official_interns2_preview_stable_20260401a_256k_0_1_20000_decay_20260304d_256k_0_0_10000_npp_decay_0_0_12000_cpt_260416_normal_combine_gqpv1_500B_256k_muon/20260416115637/hf-10382-dmtp'

# 训练超参数
sample_max_length = 32*1024
pack_max_length = 32*1024
min_num_frames = 8
max_num_frame = 36
total_epoch = 1
hf_interval = 5000
checkpoint_interval = 5000
checkpoint_maxkeep = 10
lr = 8e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.1
recompute_ratio = 1.0
loss_reduction = "square"

sp_size = 1
ep_size = 4
enable_fp8=False
torch_compile=True
chat_template="qwen3.5-vl"


# model config
fp8_config= Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE, scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE)

from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import QWEN3VL_COMPILE_CFG
QWEN3VL_COMPILE_CFG.pop("xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward")

model_cfg = Qwen3_5_VLMoE35BA3Config(only_llm_forward=False)
with (Path(model_path) / "config.json").open() as f:
    model_hf_config = json.load(f)
model_cfg.text_config.vocab_size = model_hf_config["text_config"]["vocab_size"]  

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

ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          cache_tag='cache_tags_v2',
                                          cache_dir=tokenizer_cache_dir
                                          ),
                 "tokenize_fn": Qwen3VLTokenizeFnConfig(processor_path=model_path,
                                                        chat_template=chat_template,
                                                        llm_pack_weight=-3.2,
                                                        visual_pack_weight=5.0,
                                                        max_length=sample_max_length,
                                                        video_min_frames=min_num_frames,
                                                        video_max_frames=max_num_frame,
                                                        oss_loader_cfg=oss_loader_cfg,
                                                        rand_video_max_frames=max_num_frame)
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=4,
    pack_extra_buffer_size=20,
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio, 
                      torch_compile=torch_compile,
                      ep_size=ep_size,
                      checkpoint_preserve_rng_state=False)

resume_cfg = ResumeConfig(auto_resume=False)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    sp_size=sp_size,
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
    work_dir=work_dir,
    # profile_step=10
)
