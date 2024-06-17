# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import ConcatDataset as TorchConcatDataset

from xtuner.registry import BUILDER


class LENConcatDataset(TorchConcatDataset):

    def __init__(self, datasets):
        datasets_instance = []
        for cfg in datasets:
            datasets_instance.append(BUILDER.build(cfg))
        super().__init__(datasets=datasets_instance)

        length = []
        for sub_dataset in self.datasets:
            length.extend(getattr(sub_dataset, 'length'))
        self.length = length
        self.length = [abs(x) for x in self.length]  # 全部变成正数，不区分模态

    def __repr__(self):
        main_str = 'Dataset as a concatenation of multiple datasets. \n'
        main_str += ',\n'.join(
            [f'{repr(dataset)}' for dataset in self.datasets])
        return main_str
