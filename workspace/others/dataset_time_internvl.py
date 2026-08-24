import argparse
from transformers import AutoTokenizer
import time
from xtuner.v1.utils import Config
from xtuner.v1.datasets import build_datasets
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.datasets.config import DatasetConfig
from xtuner.v1.model import InternVL3P5Dense8BConfig
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--cfg', type=str,
                        default='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/examples/v1/cpt_internvl_3p5_8b_config.py')
    args = parser.parse_args()

    trainer_cfg = Config.fromfile(args.cfg)['trainer']

    tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path, trust_remote_code=True)

    _data = {
        "sample_ratio": 0.00625,
        "annotation": "/mnt/shared-storage-user/intern-multi-modal-delivery/internvl_delivery/internvl3_5/P~other~unknown~image-text_GMAI-VL-5__5M_cls_100w_sft_data_internvl_final~1.0.0~0.0/jsonl/",
        "media_root": "yidian_ssd:s3://intern-multi-modal-h-delivery/internvl_delivery/internvl3_5/P~other~unknown~image-text_GMAI-VL-5__5M_cls_100w_sft_data_internvl_final~1.0.0~0.0/multimodal_elements/",
        "length": 199569
    }

    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": '/mnt/shared-storage-user/huanghaian/petreloss.conf'})

    dataset_config = []

    model_cfg = InternVL3P5Dense8BConfig()
    tokenize_fn = InternS1VLTokenizeFnConfig(model_cfg=model_cfg,
                                             max_length=32768,
                                             max_dynamic_patch=_data.get('max_dynamic_patch',
                                                                         None),
                                             min_dynamic_patch=_data.get('min_dynamic_patch',
                                                                         None),
                                             data_augment=_data.get('data_augment', False),
                                             system_message=_data.get('system_message', None),
                                             hash=_data.get('hash', None),
                                             oss_loader_cfg=oss_loader_cfg,
                                             template_name="internvl-3.5",
                                             debug=True,
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
        print(f"{i} data_time: {data_time:.4f} {num_img_tokens} {len_input_ids}")
        time_before_get_data = time_before_train_step
