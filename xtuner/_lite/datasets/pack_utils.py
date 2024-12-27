import random
import numpy as np
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from xtuner._lite.accelerate import profile_time_and_memory
from xtuner._lite.datasets import SoftPackDataset
from xtuner._lite import get_logger
from datasets import Dataset

logger = get_logger()


def closest_sum_indices(buffer, value):
    buffer = np.array(buffer)
    sorted_indices = np.argsort(buffer)
    closest_sum = 0
    closest_indices = []

    for idx in sorted_indices:
        closest_sum += buffer[idx]
        if closest_sum <= value:
            closest_indices.append(int(idx))
        if closest_sum >= value:
            break

    return closest_indices


class RichSoftPackDataset(SoftPackDataset):
    def __init__(self, *args, pack_len_type='total_block', flash_attn_block_size=128, pack_extra_buffer_size=1000,
                 **kwargs):
        self.pack_len_type = pack_len_type
        assert self.pack_len_type in ['total_block', 'max_block'], f'Invalid pack_len_type: {self.pack_len_type}'
        self.flash_attn_block_size = flash_attn_block_size
        self.pack_extra_buffer_size = pack_extra_buffer_size
        super().__init__(*args, **kwargs)

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        inds = [i for i in range(len(dataset))]
        random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        longest = 0
        num_patch = 0

        pack_infos = []

        while len(inds) > 0:
            shfl_i = inds.pop()

            if num_tokens[shfl_i] + sum(length_buffer) <= self.target:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                num_patch += (num_tokens[shfl_i] // self.flash_attn_block_size) ** 2 // 2
                longest = max(longest, num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    if sum(length_buffer) == self.target:
                        info = {
                            'dataset_id': dataset_id,
                            'indices': item_buffer,
                        }
                        if self.pack_len_type == 'total_block':
                            info['longest'] = int(num_patch)
                        elif self.pack_len_type == 'max_block':
                            info['longest'] = int(longest)
                        pack_infos.append(info)
                    else:
                        if self.pack_extra_buffer_size > 0:
                            # Try to find the most suitable.
                            buffer_index = inds[-self.pack_extra_buffer_size:]
                            buffer = num_tokens[buffer_index]
                            closest_indices = closest_sum_indices(buffer, self.target - sum(length_buffer))
                            indices_to_remove = []
                            for closest_inds in closest_indices:
                                indices_to_remove.append(closest_inds + len(inds) - len(buffer_index))
                                item_buffer.append(buffer_index[closest_inds])
                                length_buffer.append(num_tokens[buffer_index[closest_inds]])
                                num_patch += (num_tokens[
                                                  buffer_index[closest_inds]] // self.flash_attn_block_size) ** 2 // 2
                                longest = max(longest, num_tokens[buffer_index[closest_inds]])

                            indices_to_remove = sorted(indices_to_remove, reverse=True)
                            for index in indices_to_remove:
                                inds.pop(index)

                        info = {
                            'dataset_id': dataset_id,
                            'indices': item_buffer,
                        }
                        if self.pack_len_type == 'total_block':
                            info['longest'] = int(num_patch)
                        elif self.pack_len_type == 'max_block':
                            info['longest'] = int(longest)

                        pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                longest = num_tokens[shfl_i]
                num_patch = (num_tokens[shfl_i] // self.flash_attn_block_size) ** 2 // 2

        if len(item_buffer) > 0:
            info = {
                'dataset_id': dataset_id,
                'indices': item_buffer,
            }
            if self.pack_len_type == 'total_block':
                info['longest'] = int(num_patch)
            elif self.pack_len_type == 'max_block':
                info['longest'] = int(longest)
            pack_infos.append(info)

        total_index = []
        for infos in pack_infos:
            total_index.extend(infos['indices'])
        assert len(dataset) == len(total_index) == len(set(total_index))

        pack_infos = Dataset.from_list(pack_infos)
        return pack_infos


def build_dataset(args, datasets):
    assert len(datasets) > 0, 'No dataset found.'
    if args.dset_pack:
        with profile_time_and_memory('RichSoftPackDataset'):
            input_args = {
                'datasets': datasets,
                'target': args.pack_max_length,
                'blend': args.concat_before_pack,
            }
            if hasattr(args, 'pack_len_type'):
                input_args['pack_len_type'] = args.pack_len_type
            if hasattr(args, 'flash_attn_block_size'):
                input_args['flash_attn_block_size'] = args.flash_attn_block_size
            if hasattr(args, 'pack_extra_buffer_size'):
                input_args['pack_extra_buffer_size'] = args.pack_extra_buffer_size
            train_dataset = RichSoftPackDataset(**input_args)
    else:
        train_dataset = ConcatDataset(datasets)
        if dist.get_rank() == 0:
            logger.info(f'[Dataset] (Original) {len(train_dataset)} samples.')
    return train_dataset
