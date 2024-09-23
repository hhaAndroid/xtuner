from torch import nn

from xtuner._lite import get_logger

logger = get_logger()
# 如果没有加这个，则整个模型当一个 fsdp 模块，会导致显存增加，速度速度可能还会更慢
_LAYERS = [
    'InternLM2DecoderLayer', 'CLIPVisionModel', 'LlavaMultiModalProjector',
    'LlamaDecoderLayer', 'Phi3DecoderLayer', 'InternVisionModel',
]
# 'InternVisionEncoderLayer' 由于太小了，没有必要 wrap，直接对整个 model warp 就行


def layer_auto_wrap_policy(
    module,
    recurse: bool,
    nonwrapped_numel: int,
    layer_cls=_LAYERS,
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for
        # the leaf node or reminder
        return module.__class__.__name__ in layer_cls


def token_embedding_wrap_policy(
    module,
    recurse: bool,
    nonwrapped_numel: int,
    vocab_size: int,
) -> bool:
    if recurse:
        # always recurse
        return True

    if isinstance(module, (nn.Embedding, nn.Linear)):
        if module.weight.size(0) == vocab_size:
            return True

    return False


def all_required_grad_wrap_policy(
    module,
    recurse: bool,
    nonwrapped_numel: int,
) -> bool:
    if recurse:
        # always recurse
        return True

    requires_grads = [p.requires_grad for p in module.parameters()]

    if len(requires_grads) and all(requires_grads):
        logger.debug(module)
        return True

    return False
