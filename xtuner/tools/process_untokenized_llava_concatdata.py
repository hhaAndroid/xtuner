# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

from mmengine import Config
import numpy as np

from xtuner.registry import BUILDER
from tqdm import tqdm
from mmengine.logging import MMLogger

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file name or path.')
    args = parser.parse_args()
    return args


def build_llava_dataset(config):
    dataset = BUILDER.build(config)
    return dataset


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    logger = MMLogger.get_instance(
        name='xtuner',
        log_file='benchmark_test.log')

    datasets = cfg.train_dataloader.dataset.datasets
    for dataset_cfg in tqdm(datasets):
        offline_processed_text_folder = dataset_cfg.pop('offline_processed_text_folder')
        logger.info('=================================================================')
        logger.info(f'offline_processed_text_folder: {offline_processed_text_folder}')
        try:
            llava_dataset = build_llava_dataset(dataset_cfg)
            text_data = llava_dataset.text_data

            length_list = text_data['length']
            length_np = np.array(length_list)
            if (length_np <= 0).any():
                print('has pure text data')
            length_np = np.abs(length_list)
            min_, max_, mid_ = np.min(length_np), np.max(length_np), np.median(length_np)
            logger.info(f'token len({length_np.shape[0]}): max: {max_}, min: {min_}, mid: {mid_}')
            text_data.save_to_disk(offline_processed_text_folder)
        except Exception as e:
            logger.error(f'--------Error: {e}')
            raise NotImplementedError
