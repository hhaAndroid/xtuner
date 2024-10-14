import torch
from transformers import CLIPVisionModel
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
import os
import torch.distributed as dist
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_tp_world_size, get_tp_group, get_tp_mesh, setup_parallel)
from xtuner._lite.accelerate import dispatch_modules
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather


def new_conv2d_forward(self, input):
    print('xxx')
    tp_world_size = get_tp_world_size()
    tp_group = get_tp_group()
    tp_weight = self.weight.chunk(tp_world_size, dim=0)[dist.get_rank(tp_group)]
    output = F.conv2d(input, tp_weight, None, self.stride,
                      self.padding, self.dilation, self.groups)
    if tp_world_size > 1:
        output = all_gather(output, tp_group)
        output = torch.cat(output, dim=1)
    return output


# MASTER_PORT=29502 srun -p llm_razor --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --ntasks-per-node=1 --debug python clip_tp_demo.py
# MASTER_PORT=29503 srun -p llm_razor --gres=gpu:2 --ntasks=2 --cpus-per-task=16 --ntasks-per-node=2 --debug python clip_tp_demo.py
if __name__ == '__main__':

    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(42)
    world_size = int(os.environ['WORLD_SIZE'])
    tp_size = world_size

    setup_parallel(tp_size=tp_size)
    tp_mesh = get_tp_mesh()

    rank = dist.get_rank()

    vit_path = '/mnt/hwfile/xtuner/linzhihao/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
    vit = CLIPVisionModel.from_pretrained(vit_path)
    vit = vit.to(dtype=torch.bfloat16, device='cuda')
    dispatch_modules(vit)

    vit.vision_model.embeddings.patch_embedding.forward = \
        new_conv2d_forward.__get__(vit.vision_model.embeddings.patch_embedding)

    # 准备假图片
    set_random_seed(42)
    image = torch.randn(2, 3, 336, 336).cuda()
    image = image.to(dtype=torch.bfloat16)

    output = vit(image, output_hidden_states=True)
    print('======', dist.get_rank(), output.hidden_states[-1].shape, output.hidden_states[-1].mean(),
          output.hidden_states[-2].mean(), flush=True)

    if tp_size > 1:
        layer_tp_plan = {
            'self_attn.k_proj': ColwiseParallel(),
            'self_attn.q_proj': ColwiseParallel(),
            'self_attn.v_proj': ColwiseParallel(),
            'self_attn.out_proj': RowwiseParallel(),
            'mlp.fc1': ColwiseParallel(),
            'mlp.fc2': RowwiseParallel(),
        }

        for layer in vit.vision_model.encoder.layers:
            attention = layer.self_attn
            attention.num_heads = attention.num_heads // tp_mesh.size()
            parallelize_module(
                module=layer,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )

    output = vit(image, output_hidden_states=True)
    # 发现 output.hidden_states[-1].sum() 会有差异，但是 mean 是一样的
    print('=qqqqq=', dist.get_rank(), output.hidden_states[-1].shape, output.hidden_states[-1].mean(),
          output.hidden_states[-2].mean(), flush=True)

