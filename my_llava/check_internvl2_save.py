import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from xtuner._lite.internvl.v1_5.modeling_intern_vit import InternVisionModel

model = 'OpenGVLab/InternVL2-2B'
_cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)

vision_model = InternVisionModel(_cfg.vision_config)
internvl = AutoModel.from_pretrained(
    model,
    vision_model=vision_model,
    use_flash_attn=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

internvl.save_pretrained('test_internvl')
tokenizer.save_pretrained('test_internvl')
