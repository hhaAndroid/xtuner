import torch
from collections import OrderedDict

# 可以考虑加载预习训练好的，然后重新训练下，以适应新的 sliding window vit
root = '/mnt/petrelfs/huanghaian/code/xtuner/llava-internlm2-7b-pretrain/'
state_dict = torch.load(root+'epoch_1.pth')

new_ckpt = OrderedDict()

for k, v in list(state_dict.items()):
    if 'projector.model.0.weight' in k:
        v = v.repeat(1, 4)
        print(v.shape)
    new_ckpt[k] = v

torch.save(new_ckpt, root+'epoch_1_repeat.pth')
