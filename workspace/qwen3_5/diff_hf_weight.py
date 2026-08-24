from pathlib import Path
import json
from safetensors import safe_open
import torch

if __name__=='__main__':

    QWEN3_VL_MOE_PATH='/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
    xtuner_path = '/mnt/shared-storage-user/huanghaian/code/temp/xtuner/work_dirs/qwen3_5_35b_sft/20260228075523/hf-5'

    origin_hf_path = Path(QWEN3_VL_MOE_PATH)
    xtuner_path = Path(xtuner_path)
    origin_index_path = origin_hf_path / "model.safetensors.index.json"
    saved_index_path = xtuner_path / "model.safetensors.index.json"

    cache_save_fh = {}

    with open(origin_index_path, "r") as f:
        origin_index = json.load(f)
        with open(saved_index_path, "r") as f:
            saved_index = json.load(f)

        for key in origin_index["weight_map"].keys():
            if 'mtp' in key:
                continue
            origin_safetensor_name = origin_index["weight_map"][key]
            saved_safetensor_name = saved_index["weight_map"][key]

            origin_sf_fh_name = str(origin_hf_path / origin_safetensor_name)
            expected_sf_fh_name = str(xtuner_path / saved_safetensor_name)

            if origin_safetensor_name not in cache_save_fh:
                cache_save_fh[origin_safetensor_name] = safe_open(origin_sf_fh_name, framework="pt")
            if saved_safetensor_name not in cache_save_fh:
                cache_save_fh[saved_safetensor_name] = safe_open(expected_sf_fh_name, framework="pt")

            origin_fh = cache_save_fh[origin_safetensor_name]
            saved_fh = cache_save_fh[saved_safetensor_name]

            origin_tensor = origin_fh.get_tensor(key)
            saved_tensor = saved_fh.get_tensor(key)
            is_equal= torch.equal(origin_tensor, saved_tensor)
            if not is_equal:
                print(f"{key} is NOT equal in origin and saved safetensors.")

        # Test the tensor number in safetensors match the tensor number in model index
        safetensor_keys = []
        for safetensor_path in xtuner_path.glob("*.safetensors"):
            fh = cache_save_fh[safetensor_path.name]
            safetensor_keys.extend(fh.keys())
            safetensor_keys.sort()
        model_index_keys = list(saved_index["weight_map"].keys())
        model_index_keys.sort()
        
        if safetensor_keys != model_index_keys:
            print("The tensor keys in safetensors do NOT match the tensor keys in model index.")

