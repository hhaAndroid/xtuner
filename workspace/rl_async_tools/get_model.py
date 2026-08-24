
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLMoE30BA3Config, Qwen3VLDense4BConfig
from transformers import Qwen3VLMoeForConditionalGeneration

model_path='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-VL-30B-A3B-Instruct_MOE'

model_cfg = get_model_config_from_hf(model_path)
expected_cfg = Qwen3VLMoE30BA3Config()

from pydantic import BaseModel

def _compare_dicts(d1, d2, path, differences):
    """Recursively compare two dictionaries."""
    all_keys = set(d1.keys()) | set(d2.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        if key not in d1:
            differences.append(f"{current_path}: missing in first model")
        elif key not in d2:
            differences.append(f"{current_path}: missing in second model")
        elif type(d1[key]) != type(d2[key]):
            differences.append(f"{current_path}: type mismatch ({type(d1[key]).__name__} vs {type(d2[key]).__name__})")
        elif isinstance(d1[key], dict):
            _compare_dicts(d1[key], d2[key], current_path, differences)
        elif isinstance(d1[key], (list, tuple)):
            _compare_sequences(d1[key], d2[key], current_path, differences)
        elif d1[key] != d2[key]:
            differences.append(f"{current_path}: {d1[key]} != {d2[key]}")

def _compare_sequences(seq1, seq2, path, differences):
    """Compare two sequences (lists or tuples)."""
    if len(seq1) != len(seq2):
        differences.append(f"{path}: length mismatch ({len(seq1)} vs {len(seq2)})")
        return
    
    for i, (item1, item2) in enumerate(zip(seq1, seq2)):
        current_path = f"{path}[{i}]"
        if isinstance(item1, dict) and isinstance(item2, dict):
            _compare_dicts(item1, item2, current_path, differences)
        elif item1 != item2:
            differences.append(f"{current_path}: {item1} != {item2}")

def compare_pydantic_models(model1: BaseModel, model2: BaseModel):
    """Compare two Pydantic models by their __dict__ attributes."""
    dict1 = model1.model_dump()
    dict2 = model2.model_dump()
    
    # diff = DeepDiff(dict1, dict2, ignore_order=True)

    differences = []
    _compare_dicts(dict1, dict2, "", differences)

    if not differences:
        return True
    else:
        print('Differences found:')
        for diff in differences:
            print(f"  {diff}")
        return False

print("Models are equal:", compare_pydantic_models(model_cfg, expected_cfg))

model_cfg.hf_config.save_pretrained('./qwen3_vl_moe_30b_a3b_instruct')
