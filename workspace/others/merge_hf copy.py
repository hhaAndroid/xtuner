import os
import json
from pathlib import Path
from collections import OrderedDict
from safetensors.torch import load_file, save_file
from pathlib import Path
from shutil import copy, copytree
from safetensors import safe_open

def get_tensor_size_bytes(tensor):
    """计算tensor的字节大小"""
    return tensor.element_size() * tensor.numel()

def load_and_reshard_safetensors(
    input_dir: str,
    output_dir: str,
    max_shard_size_gb: float = 5.0,
    index_filename: str = "model.safetensors.index.json"
):
    """
    将大型safetensors文件重新分片
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        max_shard_size_gb: 每个分片的最大大小(GB)
        index_filename: 索引文件名
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    max_shard_size = int(max_shard_size_gb * 1024 * 1024 * 1024)
    
    # 获取所有safetensors文件并排序
    safetensors_files = sorted(
        [f for f in input_path.glob("*.safetensors") if f.name != index_filename]
    )
    
    print(f"找到 {len(safetensors_files)} 个safetensors文件")
    
    # 生成器：依次读取所有tensor
    def tensor_generator():
        """生成器，逐个yield (name, tensor)"""
        for safetensors_file in safetensors_files:
            print(f"正在读取: {safetensors_file.name}")
            tensors = load_file(str(safetensors_file))
            
            # 按照原始index中的顺序排序
            sorted_names = sorted(tensors.keys())
            
            for name in sorted_names:
                tensor = tensors[name]
                yield name, tensor
    
    # 开始分片
    current_shard = OrderedDict()
    current_shard_size = 0
    shard_idx = 0
    weight_map = {}
    total_size = 0
    
    for name, tensor in tensor_generator():
        tensor_size = get_tensor_size_bytes(tensor)
        total_size += tensor_size
        
        # 如果当前分片加上新tensor会超过限制，先保存当前分片
        if current_shard and (current_shard_size + tensor_size > max_shard_size):
            shard_idx += 1

            shard_filename = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            shard_path = output_path / shard_filename

            print(f"保存分片 {shard_idx}: {shard_filename}, "
                  f"大小: {current_shard_size / 1024 / 1024 / 1024:.2f} GB, "
                  f"包含 {len(current_shard)} 个tensors")
            
            save_file(current_shard, str(shard_path))
            
            # 更新weight_map
            for tensor_name in current_shard.keys():
                weight_map[tensor_name] = shard_filename
            
            # 重置当前分片
            current_shard = OrderedDict()
            current_shard_size = 0
            
        
        # 添加tensor到当前分片
        current_shard[name] = tensor
        current_shard_size += tensor_size
    
    # 保存最后一个分片
    if current_shard:
        shard_idx += 1
        total_shards = shard_idx
        shard_filename = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        shard_path = output_path / shard_filename
        
        print(f"保存最后分片 {shard_idx}: {shard_filename}, "
              f"大小: {current_shard_size / 1024 / 1024 / 1024:.2f} GB, "
              f"包含 {len(current_shard)} 个tensors")
        
        save_file(current_shard, str(shard_path))
        
        for tensor_name in current_shard.keys():
            weight_map[tensor_name] = shard_filename

    total_shards = shard_idx
    
    # 重命名所有文件，添加正确的总数
    print("\n重命名分片文件...")
    for i in range(1, total_shards + 1):
        old_name = f"model-{i:05d}-of-XXXXX.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old_path = output_path / old_name
        new_path = output_path / new_name
        
        if old_path.exists():
            old_path.rename(new_path)
        
        # 更新weight_map中的文件名
        for key in weight_map:
            if weight_map[key] == old_name:
                weight_map[key] = new_name
    
    # 生成新的index文件
    new_index = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }
    
    index_output_path = output_path / index_filename
    with open(index_output_path, 'w') as f:
        json.dump(new_index, f, indent=2)
    
    print(f"\n完成!")
    print(f"总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"生成分片数: {total_shards}")
    print(f"索引文件: {index_output_path}")


if __name__ == "__main__":
    input_directory = "/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B"
    output_directory = "/mnt/shared-storage-user/huanghaian/Qwen3-8B-resharded"
    
    load_and_reshard_safetensors(
        input_dir=input_directory,
        output_dir=output_directory,
        max_shard_size_gb=5.0
    )
    hf_dir = Path(output_directory)

    for file in Path(input_directory).iterdir():
        if file.suffix != ".safetensors":
            target_path = hf_dir / file.name
            if file.is_file():
                if not file.name.startswith(".") and not file.name.endswith("model.safetensors.index.json"):
                    copy(file, target_path)
            else:
                copytree(file, target_path, ignore_dangling_symlinks=True, dirs_exist_ok=True)
    
    # check
    origin_index_path = Path(input_directory) / "model.safetensors.index.json"
    saved_index_path = hf_dir / "model.safetensors.index.json"
    with open(origin_index_path, "r") as f:
            origin_index = json.load(f)
    with open(saved_index_path, "r") as f:
            saved_index = json.load(f)
    assert len(origin_index["weight_map"]) == len(saved_index["weight_map"])

    def get_all_keys_from_safetensors(directory: Path):
        all_keys = set()
        for safetensors_file in directory.glob("*.safetensors"):
            with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                all_keys.update(f.keys())
        return all_keys
    
    origin_keys = get_all_keys_from_safetensors(Path(input_directory))
    saved_keys = get_all_keys_from_safetensors(hf_dir)
    assert origin_keys == saved_keys

    print("验证通过，所有文件和键均匹配。")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        output_directory,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    print("模型加载成功。")
