import argparse
from transformers import AutoTokenizer
import time
from xtuner.v1.utils import Config
from xtuner.v1.datasets import build_datasets
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DatasetConfig
from typing import List, Literal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--cfg', type=str,
                        default='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/examples/v1/cpt_qwen3vl_8b_config.py')
    args = parser.parse_args()

    trainer_cfg = Config.fromfile(args.cfg)['trainer']

    tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path, trust_remote_code=True)
    
    sep = "=" * 80
    color_prefix = "\033[31m"
    color_suffix = "\033[0m"
    current_string = ""

    current_type: Literal["positive", "negative"]
    token_type: Literal["positive", "negative"]
    
    def flush_tokens(current_tokens) -> str:
        if not current_tokens:
            return ""
        text = tokenizer.decode(current_tokens, skip_special_tokens=False)
        if current_type == "positive":
            return f"{color_prefix}{text}{color_suffix}"
        return text

    _data = {
        "P~Agent~en~MiroVerse~1.0.0~0.0": {
        "sample_ratio": 1.0,
        "annotation": "/mnt/shared-storage-user/puyullmgpu-shared/puyu3-delivery/internlm3/P~Agent~en~MiroVerse~1.0.0~0.0/jsonl/",
        "length": 147985
    },
    "P~Agent~en~toolace2~1.0.0~0.0": {
        "sample_ratio": 1.0,
        "annotation": "/mnt/shared-storage-user/puyullmgpu-shared/puyu3-delivery/internlm3/P~Agent~en~toolace2~1.0.0~0.0/jsonl/",
        "length": 11295
    },
    "P~Agent~en~mengjie~1.0.0~0.0": {
        "sample_ratio": 1.0,
        "annotation": "/mnt/shared-storage-user/puyullmgpu-shared/puyu3-delivery/internlm3/P~Agent~en~mengjie~1.0.0~0.0/jsonl/",
        "length": 3014
    },
    "P~Agent~en~chuangzhi~3.0.0~0.0": {
        "sample_ratio": 1.0,
        "annotation": "/mnt/shared-storage-user/puyullmgpu-shared/puyu3-delivery/internlm3/P~Agent~en~chuangzhi~3.0.0~0.0/jsonl/",
        "length": 56680
    },
    "P~Agent~en~yining~2.0.0~0.0": {
        "sample_ratio": 1.0,
        "annotation": "/mnt/shared-storage-user/puyullmgpu-shared/puyu3-delivery/internlm3/P~Agent~en~yining~2.0.0~0.0/jsonl/",
        "length": 21314
    },
    "P~Agent~en~funreason~2.0.0~0.0": {
        "sample_ratio": 1.0,
        "annotation": "/mnt/shared-storage-user/puyullmgpu-shared/puyu3-delivery/internlm3/P~Agent~en~funreason~2.0.0~0.0/jsonl/",
        "length": 16978
    }
    }
    
    _data = {
        "sample_ratio": 1.0,
        "annotation": "/mnt/shared-storage-user/puyullmgpu-shared/puyu3-delivery/internlm3/P~Agent~en~xlam~2.0.0~0.0/jsonl/",
        "length": 147985
    }
    
    dataset_config = []

    tokenize_fn = Qwen3VLTokenizeFnConfig(
        max_length=32768,
        processor_path=trainer_cfg.tokenizer_path,
        min_pixels=_data.get('min_pixels', None),
        max_pixels=_data.get('max_pixels', None),
        video_min_total_pixels=_data.get('video_min_total_pixels', None),
        video_max_total_pixels=_data.get('video_max_total_pixels', None),
        video_min_frames=_data.get('video_min_frames', None),
        video_max_frames=_data.get('video_max_frames', None),
        fps=_data.get('fps', None),
        rand_video_max_frames=24,
        add_vision_id=True,
        system_message=_data.get('system_message', None),
        hash=_data.get('hash', None),
        enable_3d_rope=False,
        debug=False,
        oss_time_log_thr=10
    )
    _data_cfg = {"dataset": DatasetConfig(name='aa',
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          enable_sequential_sampler=True,
                                          cache_tag='cache_tags_v1',
                                          cache_dir='aa'),
                 "tokenize_fn": tokenize_fn
                 }
    dataset_config.append(_data_cfg)
    
    dataset = build_datasets(dataset_config, tokenizer)[0]

    time_before_get_data = time.time()
    for i, data in enumerate(dataset):
        num_img_tokens = data['num_img_tokens']
        len_input_ids = len(data['input_ids'])
        time_before_train_step = time.time()
        data_time = time_before_train_step - time_before_get_data
        print(f"{i} {data['is_error']}")
        if data['is_error']:
            print('===================================')
            token_ids, labels = data["input_ids"], data["labels"]
            current_string = ""
            current_tokens = []
            current_type = "negative"

            for i, label in zip(token_ids, labels):
                token_type = "positive" if label >= 0 else "negative"
                if token_type != current_type:
                    current_string += flush_tokens(current_tokens)
                    current_type = token_type
                    current_tokens = []
                current_tokens.append(i)

            current_string += flush_tokens(current_tokens)
            current_string += f"\n{sep}\n"
            print(current_string)
            
            a=1/0
        time_before_get_data = time_before_train_step
