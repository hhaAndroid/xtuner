import os
import json
import math
from pathlib import Path
from collections import OrderedDict
from safetensors.torch import save_file
from safetensors import safe_open
from shutil import copy, copytree
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch

def get_tensor_metadata(file_path):
    """
    只读取safetensors的header，不加载数据到内存。
    返回: {tensor_name: {'file': file_path, 'size': byte_size}}
    """

    # 标准且安全的方法（不会导致内存溢出，因为我们不保存 tensor）：
    tensors_info = []
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            slice_obj = f.get_slice(k)
            shape = slice_obj.get_shape()
            # 获取 dtype 的字节数。
            # safe_open 没直接给 dtype，我们只能读出来。
            # 对于大模型，这步会有点慢，但比重写文件快得多。
            # 优化：只读一个元素？ slice_obj[0:1] ? 不支持。
            
            # 终极优化：直接读取 tensor，计算完 size 立刻 del
            t = f.get_tensor(k)
            numel = t.numel()
            elem_size = t.element_size()
            size = numel * elem_size
            tensors_info.append((k, size))
            del t
            
    return tensors_info

def plan_sharding(input_dir, max_shard_size_bytes, index_filename):
    """
    第一阶段：规划。
    遍历所有输入文件，决定哪些 Tensor 应该放在哪个输出分片中。
    """
    input_path = Path(input_dir)
    safetensors_files = sorted(
        [f for f in input_path.glob("*.safetensors") if f.name != index_filename]
    )
    
    print("正在扫描输入模型元数据 (Planning Phase)...")
    
    # 收集所有的 tensor 信息
    # 结构: [ (name, size, source_file_path), ... ]
    all_tensors = []
    
    for sf in tqdm(safetensors_files, desc="Scanning files"):
        file_tensors = get_tensor_metadata(str(sf))
        for name, size in file_tensors:
            all_tensors.append({
                "name": name,
                "size": size,
                "source": str(sf)
            })
    
    # 按名称排序，保证确定性
    all_tensors.sort(key=lambda x: x["name"])
    
    # 开始装箱 (Bin Packing)
    shards_plan = [] # [ {"shard_id": 1, "tensors": [tensor_info, ...], "size": total_bytes}, ... ]
    
    current_shard_tensors = []
    current_shard_size = 0
    shard_idx = 1
    
    for tensor in all_tensors:
        if current_shard_size + tensor["size"] > max_shard_size_bytes and current_shard_tensors:
            # 封箱当前分片
            shards_plan.append({
                "shard_idx": shard_idx,
                "tensors": current_shard_tensors,
                "total_size": current_shard_size
            })
            shard_idx += 1
            current_shard_tensors = []
            current_shard_size = 0
            
        current_shard_tensors.append(tensor)
        current_shard_size += tensor["size"]
    
    # 最后一个分片
    if current_shard_tensors:
        shards_plan.append({
            "shard_idx": shard_idx,
            "tensors": current_shard_tensors,
            "total_size": current_shard_size
        })
        
    return shards_plan, all_tensors

def process_shard_task(task_config):
    """
    Worker 函数：处理单个分片的生成。
    task_config 包含: output_path, plan (list of tensors to fetch)
    """
    output_path = task_config["output_path"]
    tensors_plan = task_config["tensors"]
    
    # 为了减少文件打开次数，我们将需要读取的 tensors 按源文件分组
    # source_file -> [tensor_name1, tensor_name2, ...]
    source_map = {}
    for t in tensors_plan:
        src = t["source"]
        if src not in source_map:
            source_map[src] = []
        source_map[src].append(t["name"])
    
    # 结果字典
    state_dict = OrderedDict()
    
    # 按源文件读取
    for src_file, tensor_names in source_map.items():
        # 使用 safe_open 按需读取，避免加载整个源文件
        with safe_open(src_file, framework="pt", device="cpu") as f:
            for name in tensor_names:
                # 获取 tensor
                tensor = f.get_tensor(name)
                state_dict[name] = tensor
                
    # 保存该分片
    # save_file 会自动处理 header 和 metadata
    save_file(state_dict, output_path)
    
    # 返回该分片包含的 keys，用于主进程更新 global index
    return {
        "filename": Path(output_path).name,
        "keys": list(state_dict.keys())
    }

def parallel_reshard(
    input_dir: str,
    output_dir: str,
    max_shard_size_gb: float = 5.0,
    num_workers: int = 4,
    index_filename: str = "model.safetensors.index.json"
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    max_shard_size_bytes = int(max_shard_size_gb * 1024 * 1024 * 1024)
    
    # 1. 规划阶段
    print(f"--- Phase 1: Planning ---")
    shards_plan, all_tensors_flat = plan_sharding(input_dir, max_shard_size_bytes, index_filename)
    total_shards = len(shards_plan)
    total_model_size = sum(t["size"] for t in all_tensors_flat)
    
    print(f"总大小: {total_model_size / 1024**3:.2f} GB")
    print(f"计划分片数: {total_shards}")
    
    # 2. 并行执行阶段
    print(f"\n--- Phase 2: Parallel Execution (Workers: {num_workers}) ---")
    print(f"注意：最大内存占用约为 {num_workers} * {max_shard_size_gb} GB")
    
    tasks = []
    for plan in shards_plan:
        idx = plan["shard_idx"]
        # 直接生成最终文件名，无需后续重命名
        filename = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
        
        task_config = {
            "output_path": str(output_path / filename),
            "tensors": plan["tensors"]
        }
        tasks.append(task_config)
    
    # 收集 weight_map 信息
    final_weight_map = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交任务
        futures = [executor.submit(process_shard_task, task) for task in tasks]
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Shards"):
            try:
                result = future.result()
                filename = result["filename"]
                for key in result["keys"]:
                    final_weight_map[key] = filename
            except Exception as e:
                print(f"Worker 发生错误: {e}")
                raise e

    # 3. 生成 Index 文件
    print(f"\n--- Phase 3: Finalizing Index ---")
    new_index = {
        "metadata": {
            "total_size": total_model_size
        },
        "weight_map": final_weight_map
    }
    
    with open(output_path / index_filename, 'w') as f:
        json.dump(new_index, f, indent=2)
        
    print("完成！")

if __name__ == "__main__":
    # 配置区
    # INPUT_DIR = "/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/interns1_1_delivery/interns1-1-g8-1t-delivery-rl-base02-20260120b-data260124rc0-52k-sp2/20260124081312/hf-40"
    INPUT_DIR = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/Turner_1T/merge_model/dare-Base-d260124rc0_320-Model-d260119rc0_480_R1-d260120rc0_320_R1-d260120rc0_400_R1-d260120rc0_480_R1-fp8"
    OUTPUT_DIR = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/Turner_1T/merge_model/dare-Base-d260124rc0_320-Model-d260119rc0_480_R1-d260120rc0_320_R1-d260120rc0_400_R1-d260120rc0_480_R1-fp8-resharded"
    REPLACE_PATH_DIR = '/mnt/shared-storage-user/huanghaian/InternS1_1_1T_A22_1217'

    # INPUT_DIR = "/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B"
    # OUTPUT_DIR = "/mnt/shared-storage-user/huanghaian/Qwen3-8B-resharded"
    MAX_SHARD_SIZE_GB = 12
    
    # 关键参数：并行进程数
    # 如果每个分片5GB，开启8个进程，内存峰值可能达到 40GB+
    # 请根据机器内存调整。对于 1T 数据，建议设为 8-16 左右 (如果内存够大)
    NUM_WORKERS = 16
    
    # 1. 执行分片
    parallel_reshard(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_shard_size_gb=MAX_SHARD_SIZE_GB,
        num_workers=NUM_WORKERS
    )
    
    # 2. 复制非权重文件 (config.json, tokenizer等)
    print("\n--- Copying auxiliary files ---")
    hf_dir = Path(OUTPUT_DIR)
    for file in Path(REPLACE_PATH_DIR).iterdir():
        if file.suffix != ".safetensors":
            target_path = hf_dir / file.name
            if file.is_file():
                if not file.name.startswith(".") and not file.name.endswith("model.safetensors.index.json"):
                    copy(file, target_path)
            else:
                copytree(file, target_path, ignore_dangling_symlinks=True, dirs_exist_ok=True)

    # 3. 校验
    print("\n--- Verifying ---")
    origin_index_path = Path(INPUT_DIR) / "model.safetensors.index.json"
    saved_index_path = hf_dir / "model.safetensors.index.json"
    
    if origin_index_path.exists():
        with open(origin_index_path, "r") as f:
                origin_index = json.load(f)
        with open(saved_index_path, "r") as f:
                saved_index = json.load(f)
        
        print(f"原始 Key 数量: {len(origin_index['weight_map'])}")
        print(f"新 Key 数量: {len(saved_index['weight_map'])}")
        assert len(origin_index["weight_map"]) == len(saved_index["weight_map"])
    
    # 深度校验 Key 集合
    def get_all_keys_from_safetensors(directory: Path):
        all_keys = set()
        for safetensors_file in directory.glob("*.safetensors"):
            with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                all_keys.update(f.keys())
        return all_keys
    
    print("正在对比文件内部 Keys...")
    # 注意：如果文件数巨大，这步也会比较慢，可以根据需要注释掉
    origin_keys = get_all_keys_from_safetensors(Path(INPUT_DIR))
    saved_keys = get_all_keys_from_safetensors(hf_dir)
    
    missing = origin_keys - saved_keys
    extra = saved_keys - origin_keys
    
    if len(missing) > 0:
        print(f"错误: 丢失 keys: {list(missing)[:5]}...")
    if len(extra) > 0:
        print(f"错误: 多余 keys: {list(extra)[:5]}...")
        
    assert origin_keys == saved_keys
    print("验证通过，所有文件和键均匹配。")
    
    check_from_ptrtrained = False
    if check_from_ptrtrained:
        # 4. 尝试加载
        try:
            from transformers import AutoModelForCausalLM
            print("正在尝试加载模型配置...")
            model = AutoModelForCausalLM.from_pretrained(
                OUTPUT_DIR,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto"
            )
            print("模型加载成功 (MetaData Load)。")
        except Exception as e:
            print(f"模型加载测试失败: {e}")