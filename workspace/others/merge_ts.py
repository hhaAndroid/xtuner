
from safetensors import safe_open

path='/mnt/shared-storage-user/huanghaian/Intern-S1-Pro/model-time_series-00002-of-00002.safetensors'

total_size = 0
with safe_open(path, framework="pt", device="cpu") as f:
    for k in f.keys():
        slice_obj = f.get_slice(k)
        shape = slice_obj.get_shape()
        t = f.get_tensor(k)
        numel = t.numel()
        elem_size = t.element_size()
        size = numel * elem_size
        total_size += size
print(f"Total size of tensors in {path}: {total_size}") # 291946400+10240000=302186400

# 302186400