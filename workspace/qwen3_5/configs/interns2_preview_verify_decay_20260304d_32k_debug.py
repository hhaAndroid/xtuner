from cyclopts import token
from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
    FSDPConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig, PretrainTokenizeFunctionConfig, OpenaiTokenizeFunctionConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.loss import CELossConfig
from xtuner.v1.float8 import Float8Config, ScalingGranularity
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.utils.internal_metrics import InternalMetricsConfig
from xtuner.v1.model.moe.moe import ZLossConfig, BalancingLossConfig
from xtuner.v1.utils.logger import get_logger
from pathlib import Path

from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import QWEN3VL_COMPILE_CFG
QWEN3VL_COMPILE_CFG.pop("xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward")


import os
from pathlib import Path
import json
import pandas as pd

# from pretrain.task_configs.interns1_air_stable_20251118a_0_0_52000_decay_20260129a_1_0 import WORK_DIR


logger = get_logger()


EP_SIZE = 1
SP_SIZE = 1

INTRA_LAYER_MICRO_BATCH = 1
SEED = 1024
LR = 3e-5
LR_MIN = 1e-6
WD = 0.1

SEQ_LEN = 32768
MICRO_BATCH = 1

GLOBAL_BS = 1 # 1024

TOTAL_STEP = 5  # stable 52000 steps

CHECKPOINT_INTERVAL = 1000
SNAPSHOT_INTERVAL = 500
CHECK_INTERVAL = 200
INTERNAL_METRIC_INTERVAL = 100

DATA_VERSION = "decay_20260304d"

HF_MODEL_PATH = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2_preview/patched/InternS2_Qwen3_5_model"  # noqa: E501

CACHE_DIR = (
    "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2_preview/cache/decay_20260304d_v2"
)

DATASET_CLS = "/mnt/shared-storage-user/songdemin/user/nil0x9/codebase/pretrain_formula/decay_20260304d.json"
with open(DATASET_CLS) as f:
    datasets_info = json.load(f)


formula = "/mnt/shared-storage-user/songdemin/user/nil0x9/codebase/pretrain_formula/formula/decay_20260304d.csv"
df = pd.read_csv(formula)
weights = df.set_index('subset_name')['weight_result'].to_dict()
epochs = df.set_index('subset_name')['specified_epochs'].to_dict()


dataset_config = []
for subset in datasets_info:
    dataset_name = subset["name"]
    # if dataset_name == "P~Reflection~reasoning~sft_decay_opensource_meta_cognition_251227_rft~2.0.0~0.0":
    #     continue
    dataset_path = Path(subset["path"])
    dataset_type = subset["dataset_type"]
    dataset_weight = subset["weight"]
    assert abs(dataset_weight - weights[dataset_name]) <= 1e-8, \
    f"dataset_name: {dataset_name}, formula weight: {dataset_weight}, table weight: {weights[dataset_name]}"
    specified_epochs = epochs[dataset_name]
    assert specified_epochs >= 0

    sample_ratio = specified_epochs * 1.1  # some excess epochs for safety
    for jsonl in dataset_path.rglob("*.jsonl"):
        tokenizer_fn = None
        if dataset_type == "pretrain":
            tokenizer_fn = PretrainTokenizeFunctionConfig()
        elif dataset_type == "openai":
            tokenizer_fn = OpenaiTokenizeFunctionConfig(chat_template="qwen3")
        elif dataset_type == "ftdp":
            # tokenizer_fn = FTDPTokenizeFnConfig(chat_template="qwen2")
            logger.warning(f"subset {dataset_name} file `{jsonl}` is skipped bc it is of type ftdp!")
        else:
            logger.warning(f"subset {dataset_name} file `{jsonl}` is skipped for no template is matched")

        if tokenizer_fn:
            dataset_config.append(
                {
                    "dataset": DatasetConfig(
                        name="data1",
                        anno_path=str(jsonl),  # noqa: E501
                        sample_ratio=sample_ratio,
                        cache_dir=CACHE_DIR,
                        cache_tag="cache_tag_v1",
                    ),
                    "tokenize_fn": tokenizer_fn,
                }
            )

logger.info(f"Total datasets: {len(dataset_config)}")

dataloader_config = DataloaderConfig(
    pack_max_length=SEQ_LEN * MICRO_BATCH,
    num_workers=1,
    pack_level="hard",
    tokenizer_hash="8553c9064f1e581a",
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=LR, weight_decay=WD)
lr_cfg = LRConfig(lr_type="linear", lr_min=LR_MIN, warmup_ratio=1000)

fsdp_cfg = FSDPConfig(
    torch_compile=True,
    ep_size=EP_SIZE,
)

# FP8 训练 
float8_cfg = Float8Config(
    scaling_granularity_gemm=ScalingGranularity.TILEWISE,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)
model_cfg = Qwen3_5_VLMoE35BA3Config()

with (Path(HF_MODEL_PATH) / "config.json").open() as f:
    model_hf_config = json.load(f)

model_cfg.text_config.vocab_size = model_hf_config["text_config"]["vocab_size"]

# model_cfg: Qwen3_5_VLMoE35BA3Config = get_model_config_from_hf(HF_MODEL_PATH)  # type: ignore[assignment]

model_cfg.text_config.z_loss_cfg = ZLossConfig(z_loss_alpha=0)  # type: ignore[attr-defined]
model_cfg.text_config.balancing_loss_cfg = BalancingLossConfig(balancing_loss_alpha=0) # type: ignore[attr-defined]
model_cfg.text_config.float8_cfg = float8_cfg # type: ignore[attr-defined]
model_cfg.only_llm_forward = True

resume_cfg = ResumeConfig(auto_resume=True)

script_path = os.path.abspath(__file__)
script_name = Path(script_path).stem

WORK_DIR = os.environ['WORK_DIR']

# internal_metrics_cfg = InternalMetricsConfig(
#     internal_metrics_interval=INTERNAL_METRIC_INTERVAL,
#     monitor_weights_rms_norm=True,
#     monitor_attn_logits_stats=True,
#     monitor_moe_router_logits_stats=True,  # only applies to MoE models
#     monitor_moe_load_balance_stats=True,
# )

trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataset_cfg=dataset_config,  # type: ignore[arg-type]
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=GLOBAL_BS,
    sp_size=SP_SIZE,
    intra_layer_micro_batch=INTRA_LAYER_MICRO_BATCH,
    total_step=TOTAL_STEP,
    load_from=HF_MODEL_PATH,
    strict_load=False,  # FoPE
    seed=SEED,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    hf_interval=CHECKPOINT_INTERVAL,
    snapshot_interval=SNAPSHOT_INTERVAL,
    resume_cfg=resume_cfg,
    work_dir=WORK_DIR,
    tokenizer_path=HF_MODEL_PATH,
    exp_tracker="tensorboard",
    skip_checkpoint_validation=True,
    check_health_interval=CHECK_INTERVAL,
    internal_metrics_cfg=None,
)
