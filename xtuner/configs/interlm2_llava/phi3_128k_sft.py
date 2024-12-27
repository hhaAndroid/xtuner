# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, AutoModel)

from xtuner.dataset import InternVL_V1_5_LLaVADataset, InternVL_v1_5_LLaVAProxyEvalDataset, process_hf_dataset
from xtuner.dataset.collate_fns import mm_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, alpaca_map_fn
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.model import InternVL_v1_5_LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.evaluation import MMEDataset, MultipleChoiceDataset, POPEDataset, \
    HallusionDataset, TextVQADataset, GQADataset, VQAv2Dataset, ChartQADataset, GeneralVQADataset
from xtuner.dataset import ConcatDataset
from xtuner.engine.runner import TrainLoop, ValLoop, TestLoop
from mmengine.dataset import DefaultSampler
from xtuner.dataset.utils import internvl_1_5_encode_fn
import torch

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = '/mnt/hwfile/xtuner/huanghaian/model/Phi-3-mini-128k-instruct/'
visual_encoder_name_or_path = '/mnt/hwfile/xtuner/huanghaian/model/InternViT-300M-448px/'
pretrained_pth = '/mnt/petrelfs/huanghaian/code/mm/xtuner/intervit_300m_phi3_128k_projector.pth'

# the_cauldron
the_cauldron_data_path = '/mnt/hwfile/xtuner/linzhihao/dataset/the_cauldron_save/240528_clean/cleaned/mix_1.json'
the_cauldron_image_folder = 's3://xtuner/datasets/xtuner-vldata-v1/data/the_cauldron'

# internvl sft
internvl_sft_data_root = '/mnt/hwfile/xtuner/linzhihao/dataset/internvl_sft/'

sharegpt4v_caption_data_path = internvl_sft_data_root + 'with_image_wh/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl'
sharegpt4v_caption_image_folder = internvl_sft_data_root + 'data'

sharegpt4v_data_path = internvl_sft_data_root + 'with_image_wh/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_wo_textcaps_ocrvqa.jsonl'
sharegpt4v_image_folder = internvl_sft_data_root + 'data'

dvqa_data_path = internvl_sft_data_root + 'with_image_wh/dvqa_train_200k.jsonl'
dvqa_image_folder = internvl_sft_data_root + 'data/dvqa'

chartqa_data_path = internvl_sft_data_root + 'with_image_wh/chartqa_train_18k.jsonl'
chartqa_image_folder = internvl_sft_data_root + 'data/chartqa'

ai2d_data_path = internvl_sft_data_root + 'with_image_wh/ai2d_train_12k.jsonl'
ai2d_image_folder = internvl_sft_data_root + 'data/ai2d'

docvqa_data_path = internvl_sft_data_root + 'with_image_wh/docvqa_train_10k.jsonl'
docvqa_image_folder = internvl_sft_data_root + 'data/docvqa'

# allava
allava_data_root = '/mnt/hwfile/xtuner/huanghaian/data/ALLaVA-4V/'

allava_laion_data_path = allava_data_root + 'allava_laion/ALLaVA-Instruct-LAION-4V_llava.json'
allava_laion_image_folder = '/mnt/hwfile/openmmlab/zhaoxiangyu/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/624bd4c5fedc2209cf952eedf75712413d8d912c/'

allava_vflan_data_path = allava_data_root + 'allava_vflan/ALLaVA-Instruct-VFLAN-4V_llava.json'
allava_vflan_image_folder = '/mnt/hwfile/openmmlab/zhaoxiangyu/'

# MMC-Instruction
mmc_data_root = '/mnt/hwfile/xtuner/linzhihao/dataset/MMC/MMC-Instruction/'
mmc_data_path = mmc_data_root + 'mmc_instruction_clean_0528.json'
mmc_image_folder = mmc_data_root

# SVIT
svit_data_root = '/mnt/hwfile/xtuner/linzhihao/dataset/SVIT/'
svit_data_path = svit_data_root + 'SVIT_core_150K_w_image_wh.jsonl'
svit_image_folder = internvl_sft_data_root + 'data'

# LLaVAR
llavar_data_root = '/mnt/hwfile/xtuner/huanghaian/data/LLaVAR/'
llavar_data_path = llavar_data_root + 'llavar_20k_llava.json'
llavar_image_folder = llavar_data_root

# Alpaca-GPT4
alpaca_gpt4_json_path = '/mnt/hwfile/xtuner/linzhihao/dataset/Alpaca-GPT4/alpaca_gpt4.json'
# OpenHermes-2_5
openhermes2_5_json_path = '/mnt/hwfile/xtuner/linzhihao/dataset/OpenHermes-2_5/openhermes2_5_cleaned.json'

# comvint
comvint_data_root = '/mnt/hwfile/xtuner/huanghaian/data/comvint/'
comvint_data_path = comvint_data_root + 'ComVint_llava_clear_1.json'
comvint_image_folder = comvint_data_root

# DocOwl_v1_5
docowl_v1_5_data_root = '/mnt/hwfile/xtuner/huanghaian/data/DocOwl_v1_5/'
docowl_reason_data_path = docowl_v1_5_data_root + 'DocReason25K/detailed_explanation_llava_clear.json'
docowl_reason_image_folder = docowl_v1_5_data_root + 'DocReason25K'


# EST-VQA
est_vqa_data_root = '/mnt/hwfile/xtuner/huanghaian/data/EST-VQA/'
est_vqa_data_path = est_vqa_data_root + 'est_vqa_llava.json'
est_vqa_image_folder = est_vqa_data_root

# SciGraphQA-295K-train
scigraphqa_data_root = '/mnt/hwfile/xtuner/huanghaian/data/SciGraphQA-295K-train/'
scigraphqa_data_path = scigraphqa_data_root + 'SciGraphQA-295K-train_llava.json'
scigraphqa_image_folder = scigraphqa_data_root

# VisDial
visdial_data_root = '/mnt/hwfile/xtuner/huanghaian/data/VisDial/'
visdial_data_path = visdial_data_root + 'visdial_train_123k_llava_clear.json'
visdial_image_folder = visdial_data_root

# viquae
viquae_data_root = '/mnt/hwfile/xtuner/huanghaian/data/viquae/'
viquae_data_path = viquae_data_root + 'train_val_llava_1.json'
viquae_image_folder = viquae_data_root

# screen_qa
screen_qa_data_root = '/mnt/hwfile/xtuner/huanghaian/data/screen_qa/'
screen_qa_data_path = screen_qa_data_root + 'screen_qa_all_llava_clear_1.json'
screen_qa_image_folder = screen_qa_data_root

# KVQA
kvqa_data_root = '/mnt/hwfile/xtuner/huanghaian/data/kvqa/OpenDataLab___KVQA/raw/'
kvqa_data_path = kvqa_data_root + 'dataset_llava_clear.json'
kvqa_image_folder = kvqa_data_root

# DenseCaps
densecaps_data_root = '/mnt/hwfile/xtuner/huanghaian/data/Detailed_Caption/'
densecaps_data_path = densecaps_data_root + 'densecap_llava_clear.json'
densecaps_image_folder = densecaps_data_root

# laion_gpt4v
laion_gpt4v_data_root = '/mnt/hwfile/xtuner/huanghaian/data/laion_gpt4v/'
laion_gpt4v_data_path = laion_gpt4v_data_root + 'laion_gpt4v_llava_clear.json'
laion_gpt4v_image_folder = laion_gpt4v_data_root

# TextOCR-GPT4V
textocr_gpt4v_data_root = '/mnt/hwfile/xtuner/huanghaian/data/TextOCR-GPT4V/'
textocr_gpt4v_data_path = textocr_gpt4v_data_root + 'train_llava.json'
textocr_gpt4v_image_folder = textocr_gpt4v_data_root

# LVIS-Instruct4V
lvis_instruct4v_data_root = '/mnt/hwfile/xtuner/huanghaian/data/LVIS-Instruct4V/'
lvis_instruct4v_data_path = lvis_instruct4v_data_root + 'lvis_instruct4v_220k_llava.json'
lvis_instruct4v_image_folder = lvis_instruct4v_data_root

prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# official bs=1024 lr=4e-5

# Scheduler & Optimizer
batch_size = 4  # per_device 16x32g
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 5000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 5000
SYSTEM = ''

min_num = 1
max_num = 12
downsample_ratio = 0.5
image_size = 448
patch_size = 16

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=InternVL_v1_5_LLaVAModel,
    visual_select_layer=-1,
    custom_mlp=True,
    use_lldr=True,  # xxxxxxx
    downsample_ratio=downsample_ratio,
    tokenizer=tokenizer,
    template=prompt_template,
    image_processor=image_processor,
    freeze_llm=False,
    freeze_visual_encoder=False,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    visual_encoder=dict(
        type=AutoModel.from_pretrained,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        pretrained_model_name_or_path=visual_encoder_name_or_path))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
tokenized_root = '/mnt/hwfile/xtuner/huanghaian/tokenized/internvl_phi3_128k_300m_sft_v1/'

the_cauldron_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'the_cauldron_dataset_mix_v5',
    data_path=the_cauldron_data_path,
    image_folder=the_cauldron_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

sharegpt4v_caption_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'sharegpt4v_caption_dataset',
    data_path=sharegpt4v_caption_data_path,
    image_folder=sharegpt4v_caption_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

sharegpt4v_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'sharegpt4v_dataset',
    data_path=sharegpt4v_data_path,
    image_folder=sharegpt4v_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

dvqa_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'dvqa_dataset',
    data_path=dvqa_data_path,
    image_folder=dvqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

chartqa_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'chartqa_dataset',
    data_path=chartqa_data_path,
    image_folder=chartqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

ai2d_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'ai2d_dataset',
    data_path=ai2d_data_path,
    image_folder=ai2d_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

docvqa_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'docvqa_dataset',
    data_path=docvqa_data_path,
    image_folder=docvqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

allava_laion_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'allava_laion_dataset',
    data_path=allava_laion_data_path,
    image_folder=allava_laion_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

allava_vflan_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'allava_vflan_dataset',
    data_path=allava_vflan_data_path,
    image_folder=allava_vflan_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

mmc_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'mmc_dataset',
    data_path=mmc_data_path,
    image_folder=mmc_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

svit_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'svit_dataset',
    data_path=svit_data_path,
    image_folder=svit_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

llavar_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'llavar_dataset',
    data_path=llavar_data_path,
    image_folder=llavar_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

alpaca_gpt4_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'alpaca_gpt4_dataset',
    data_path=alpaca_gpt4_json_path,
    image_folder='',
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=alpaca_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

OpenHermes2_5_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'OpenHermes2_5_dataset',
    data_path=openhermes2_5_json_path,
    image_folder='',
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

comvint_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'comvint_dataset',
    data_path=comvint_data_path,
    image_folder=comvint_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

docowl_reason_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'docowl_reason_dataset',
    data_path=docowl_reason_data_path,
    image_folder=docowl_reason_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

est_vqa_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'est_vqa_dataset',
    data_path=est_vqa_data_path,
    image_folder=est_vqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

scigraphqa_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'scigraphqa_dataset',
    data_path=scigraphqa_data_path,
    image_folder=scigraphqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

visdial_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'visdial_dataset',
    data_path=visdial_data_path,
    image_folder=visdial_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

viquae_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'viquae_dataset',
    data_path=viquae_data_path,
    image_folder=viquae_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

screen_qa_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'screen_qa_dataset',
    data_path=screen_qa_data_path,
    image_folder=screen_qa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

kvqa_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'kvqa_dataset',
    data_path=kvqa_data_path,
    image_folder=kvqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

densecaps_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'densecaps_dataset',
    data_path=densecaps_data_path,
    image_folder=densecaps_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

laion_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'laion_dataset',
    data_path=laion_gpt4v_data_path,
    image_folder=laion_gpt4v_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

textocr_gpt4v_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'textocr_gpt4v_dataset',
    data_path=textocr_gpt4v_data_path,
    image_folder=textocr_gpt4v_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        image_size=image_size,
        patch_size=patch_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

lvis_instruct4v_dataset = dict(
    type=InternVL_V1_5_LLaVADataset,
    custom=True,
    image_size=image_size,
    min_num=min_num,
    max_num=max_num,
    downsample_ratio=downsample_ratio,
    offline_processed_text_folder=tokenized_root+'lvis_instruct4v_dataset',
    data_path=lvis_instruct4v_data_path,
    image_folder=lvis_instruct4v_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    encode_map_fn=dict(
        type=internvl_1_5_encode_fn,
        patch_size=patch_size,
        image_size=image_size,
        min_num=min_num,
        max_num=max_num),
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        the_cauldron_dataset, sharegpt4v_caption_dataset, sharegpt4v_dataset,
        dvqa_dataset, chartqa_dataset, ai2d_dataset, docvqa_dataset,
        allava_laion_dataset, allava_vflan_dataset, mmc_dataset, svit_dataset,
        llavar_dataset, alpaca_gpt4_dataset, comvint_dataset,
        docowl_reason_dataset, est_vqa_dataset,
        scigraphqa_dataset, visdial_dataset, viquae_dataset, screen_qa_dataset,
        kvqa_dataset, densecaps_dataset, laion_dataset, textocr_gpt4v_dataset,
        lvis_instruct4v_dataset, OpenHermes2_5_dataset
    ])

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=mm_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    constructor='LearningRateDecayOptimWrapperConstructor',  # ====================
    paramwise_cfg=dict(layer_decay_rate=0.9),  # vit-l
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=save_steps)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

# ==================== val and test cfg =======================
val_dataset = [
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/MMBench_DEV_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
]

test_dataset = [
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/MMBench_DEV_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/MMBench_TEST_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/SEEDBench_IMG.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/ScienceQA_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/ScienceQA_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/MMMU_DEV_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/AI2D_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=TextVQADataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/orig_llava_eval/textvqa/llava_textvqa_val_v051_ocr.jsonl',
        ann_file='/mnt/hwfile/xtuner/huanghaian/LMUData/text_vqa/TextVQA_0.5.1_val.json',
        image_folder='/mnt/hwfile/xtuner/huanghaian/LMUData/text_vqa/train_images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MMEDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/MME.tsv',
        image_folder='/mnt/hwfile/xtuner/linzhihao/dataset/mme/MME_Benchmark_release',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=HallusionDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/HallusionBench.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=POPEDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file=[
            '/mnt/hwfile/xtuner/linzhihao/dataset/POPE/coco_pope_adversarial.json',
            '/mnt/hwfile/xtuner/linzhihao/dataset/POPE/coco_pope_popular.json',
            '/mnt/hwfile/xtuner/linzhihao/dataset/POPE/coco_pope_random.json'
        ],
        coco_val_path='/mnt/hwfile/xtuner/linzhihao/dataset/coco/val2014/',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=GQADataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/linzhihao/dataset/gqa/gqa_llava_eval/llava_gqa_testdev_balanced.jsonl',
        ann_file='/mnt/hwfile/xtuner/linzhihao/dataset/gqa/gqa_llava_eval/testdev_balanced_questions.json',
        # image_folder='/mnt/hwfile/xtuner/linzhihao/dataset/gqa/images',
        image_folder='/mnt/petrelfs/share_data_old/basemodel/dataset/multimodality/gqa/images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/linzhihao/dataset/MMStar/MMStar.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor),
    dict(
        type=ChartQADataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file=['/mnt/hwfile/xtuner/huanghaian/LMUData/ChartQA/ChartQA Dataset/test/test_human.json',
                   '/mnt/hwfile/xtuner/huanghaian/LMUData/ChartQA/ChartQA Dataset/test/test_augmented.json'],
        image_folder='/mnt/hwfile/xtuner/huanghaian/LMUData/ChartQA/ChartQA Dataset/test/png',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor
    ),
    dict(
        type=GeneralVQADataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/DocVQA_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor
    ),
    dict(
        type=GeneralVQADataset,
        proxy_eval_dataset=dict(type=InternVL_v1_5_LLaVAProxyEvalDataset, custom=True, min_num=min_num, max_num=max_num),
        data_file='/mnt/hwfile/xtuner/huanghaian/LMUData/InfoVQA_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor
    )
]

# TODO: We are not currently using val_evaluator
# Don't support num_workers > 0
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=0,
#     drop_last=False,
#     sampler=dict(type=DefaultSampler, shuffle=False),
#     dataset=dict(type=ConcatDataset, datasets=val_dataset),
#     collate_fn=dict(type=mm_collate_fn, extra_collate_keys=['img_id']))
# val_evaluator = dict()
# val_cfg = dict(type=ValLoop)

# TODO: We are not currently using test_evaluator
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
    collate_fn=dict(type=mm_collate_fn, extra_collate_keys=['img_id'])
)

test_evaluator = {}
test_cfg = dict(type=TestLoop, select_metric='first')
