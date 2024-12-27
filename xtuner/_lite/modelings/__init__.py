from .internlm2 import InternLM2Config, InternLM2ForCausalLM
from .llava.modeling_llava import LlavaForConditionalGeneration
from .llava.configuration_llava import EnhancedLlavaConfig
from .llava.processing_llava import LlavaProcessor
from .qwen2 import Qwen2Config, Qwen2ForCausalLM

def register_remote_code():
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register('internlm2', InternLM2Config, exist_ok=True)
    AutoModelForCausalLM.register(
        InternLM2Config, InternLM2ForCausalLM, exist_ok=True)
    AutoConfig.register('qwen2', Qwen2Config, exist_ok=True)
    AutoModelForCausalLM.register(
        Qwen2Config, Qwen2ForCausalLM, exist_ok=True)

