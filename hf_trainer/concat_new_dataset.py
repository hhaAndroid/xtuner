# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import ConcatDataset as TorchConcatDataset

from xtuner.registry import BUILDER
import numpy as np
import logging

logger = logging.getLogger()


class LENConcatDataset(TorchConcatDataset):

    def __init__(self, datasets):
        logger.info('==== start LENConcatDataset ====')
        datasets_instance = []
        for cfg in datasets:
            logger.info(f'==== start datasets {cfg.type} ====')
            datasets_instance.append(BUILDER.build(cfg))
        super().__init__(datasets=datasets_instance)

        length = []
        for sub_dataset in self.datasets:
            length.extend(getattr(sub_dataset, 'modality_length'))

        # length = np.abs(np.array(length).reshape(-1))
        # self.length = length.tolist()  # 全部变成正数，不区分模态
        self.length = length  # 区分模态
        logger.info('==== end LENConcatDataset ====')

    def __repr__(self):
        main_str = 'Dataset as a concatenation of multiple datasets. \n'
        main_str += ',\n'.join(
            [f'{repr(dataset)}' for dataset in self.datasets])
        return main_str
