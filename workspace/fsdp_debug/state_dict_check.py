import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
import os
import sys

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128, bias=False)
        self.fc2 = nn.Linear(128, 128, bias=False)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    setup(rank, world_size)
    
    print(f"\n{'='*60}")
    print(f"Rank {rank}: 开始测试")
    print(f"{'='*60}\n")

    # Step 1: 创建完整权重（只在 rank 0）
    if rank == 0:
        print(f"Rank {rank}: 创建普通模型并生成完整权重")
        normal_model = SimpleModel().cuda()
        full_state_dict = normal_model.state_dict()
        
        with torch.no_grad():
            full_state_dict['fc1.weight'].fill_(1.0)
            full_state_dict['fc1.weight'][0, 0] = 99.0
            full_state_dict['fc2.weight'].fill_(2.0)
        
        print(f"Rank {rank}: fc1.weight 完整形状: {full_state_dict['fc1.weight'].shape}")
        print(f"Rank {rank}: fc1.weight device: {full_state_dict['fc1.weight'].device}")
        print(f"Rank {rank}: fc1.weight[0,0] = {full_state_dict['fc1.weight'][0, 0].item()}")
    else:
        full_state_dict = None
    
    # 广播完整权重 - 所有 rank 都需要有数据
    print(f"Rank {rank}: 开始广播权重")
    if rank == 0:
        objects = [full_state_dict]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    full_state_dict = objects[0]
    
    # ✅ 关键修复：将 broadcast 后的 tensor 移到正确的 GPU
    print(f"Rank {rank}: 收到完整权重，检查设备...")
    for key, value in full_state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"Rank {rank}: {key} device before: {value.device}")
            if not value.is_cuda or value.device.index != rank:
                # 移到当前 rank 的 GPU
                full_state_dict[key] = value.cuda(rank)
                print(f"Rank {rank}: {key} 移动到 cuda:{rank}")
            print(f"Rank {rank}: {key} device after: {full_state_dict[key].device}")

    dist.barrier()
    print(f"Rank {rank}: 设备检查完成\n")
    
    # Step 2: 创建 FSDP 模型
    print(f"Rank {rank}: 创建 FSDP 模型")
    fsdp_model = SimpleModel().cuda()
    fsdp_model = fully_shard(fsdp_model)
    
    print(f"Rank {rank}: FSDP 后 fc1.weight 类型: {type(fsdp_model.fc1.weight)}")
    print(f"Rank {rank}: FSDP 后 fc1.weight 全局形状: {fsdp_model.fc1.weight.shape}")
    
    if hasattr(fsdp_model.fc1.weight, '_local_tensor'):
        local_tensor = fsdp_model.fc1.weight._local_tensor
        print(f"Rank {rank}: fc1.weight 本地分片形状: {local_tensor.shape}")
    
    dist.barrier()
    
    # Step 3: 手动转换为 DTensor - 所有 rank 都参与
    print(f"\nRank {rank}: ===== 开始转换为 DTensor =====")
    
    # 获取 device_mesh
    mesh = fsdp_model.fc1.weight.device_mesh
    print(f"Rank {rank}: Device mesh: {mesh}")
    
    # 将完整权重转换为 Replicate DTensor
    from torch.distributed.tensor import distribute_tensor, Replicate
    
    converted_state_dict = {}
    for key, value in full_state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"Rank {rank}: 转换 {key} 为 DTensor (device: {value.device})")
            
            # ✅ 再次确认 tensor 在正确设备上
            assert value.is_cuda, f"Rank {rank}: {key} 不在 CUDA 上！"
            assert value.device.index == rank, f"Rank {rank}: {key} 在错误的 GPU 上！"
            
            dtensor_value = distribute_tensor(value, mesh, [Replicate()])
            converted_state_dict[key] = dtensor_value
            print(f"Rank {rank}: 完成转换 {key}")
        else:
            converted_state_dict[key] = value
    
    dist.barrier()
    print(f"Rank {rank}: 所有权重转换完成\n")
    
    # 加载转换后的 state_dict
    print(f"Rank {rank}: 加载 DTensor state_dict")
    incompatible = fsdp_model.load_state_dict(converted_state_dict, strict=False)
    print(f"Rank {rank}: 加载完成")
    print(f"Rank {rank}: Missing keys: {incompatible.missing_keys}")
    print(f"Rank {rank}: Unexpected keys: {incompatible.unexpected_keys}")
    
    dist.barrier()
    
    # Step 4: 验证
    print(f"\nRank {rank}: ===== 验证加载结果 =====")
    
    if hasattr(fsdp_model.fc1.weight, '_local_tensor'):
        local_tensor = fsdp_model.fc1.weight._local_tensor
        print(f"Rank {rank}: 加载后本地分片形状: {local_tensor.shape}")
        print(f"Rank {rank}: 加载后本地分片前5个值: {local_tensor.flatten()[:5]}")
        
        if rank == 0:
            first_value = local_tensor[0, 0].item()
            print(f"Rank {rank}: local_tensor[0,0] = {first_value} (期望 99.0)")
            has_special_marker = abs(first_value - 99.0) < 0.01
            print(f"Rank {rank}: 包含特殊标记: {has_special_marker}")
        else:
            print(f"Rank {rank}: 验证后半部分数据...")
            all_ones = torch.allclose(local_tensor, torch.ones_like(local_tensor))
            print(f"Rank {rank}: 数据全为 1.0: {all_ones}")
    
    dist.barrier()
    
    # Step 5: 测试前向传播
    print(f"\nRank {rank}: ===== 测试前向传播 =====")
    input_tensor = torch.ones(4, 128).cuda() * 0.1
    
    with torch.no_grad():
        output = fsdp_model(input_tensor)
        print(f"Rank {rank}: 输出形状: {output.shape}")
        print(f"Rank {rank}: 输出前5个值: {output[0, :5]}")
    
    dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("测试完成！")
        print("关键要点：")
        print("1. broadcast_object_list 后的 tensor 可能在 CPU 或错误的 GPU")
        print("2. distribute_tensor 要求 tensor 在当前 rank 的正确 GPU 上")
        print("3. 需要手动将 tensor 移到 cuda(rank)")
        print("4. distribute_tensor(..., [Replicate()]) 是集体操作")
        print(f"{'='*60}")
    
    cleanup()

if __name__ == "__main__":
    world_size = 2
    
    if len(sys.argv) > 1:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        run_test(rank, world_size)
    else:
        import torch.multiprocessing as mp
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)