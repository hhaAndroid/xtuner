
# mmengine
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from transformers import AutoProcessor
import torch
from xtuner.v1.model.compose.qwen3_5 import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.moe import SequenceContext

torch.set_printoptions(precision=8, sci_mode=False)

def forward_hf(path,inputs):
    from transformers import Qwen3_5MoeForConditionalGeneration
    hf_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # flash_attention_2/3 一开就 Assertion `probability tensor contains either `inf，其余是可以的，默认跑的是 spda
        device_map="cuda",
        trust_remote_code=True
    )
    
    try:
        with torch.no_grad():
            output = hf_model(
                input_ids = inputs.input_ids,
                labels = inputs.input_ids.clone(),
                # attention_mask = inputs.attention_mask,
                use_cache = False
            )
        print('hf forward:',output.loss, output.aux_loss)
    except ZeroDivisionError:
        print('hf forward: ZeroDivisionError')
    finally:
        del hf_model
        torch.cuda.empty_cache()


def forward_xtuenr(path, inputs):
    with torch.device("meta"):
        model_cfg = Qwen3_5_VLMoE35BA3Config(compile_cfg=False)
        qwen3vl_model = model_cfg.build().to(torch.bfloat16)

    qwen3vl_model.from_hf(path)
    
    input_ids = inputs.input_ids
    labels = inputs.input_ids.clone()

    shift_input_ids = input_ids[:, :-1]
    shifted_labels = labels[:, 1:]
    # shift_input_ids = input_ids
    # shifted_labels = labels
    seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
    seq_ctx_list = [seq_ctx]

    loss_cfg = CELossConfig()
    LossContext = loss_cfg.loss_ctx_cls
    loss_ctx = loss_cfg.build(shifted_labels=shifted_labels)
    loss_ctx_list = [loss_ctx]
    loss_ctx_list = LossContext.build_batches(loss_ctx_list)
    loss_ctx = loss_ctx_list[0]
    seq_ctx = seq_ctx_list[0]

    qwen3vl_model.to('cuda')
    with torch.no_grad():
        output = qwen3vl_model(
            seq_ctx=seq_ctx,
            loss_ctx=loss_ctx,
        )
    torch.cuda.empty_cache()
    loss = output["loss"]
    balancing_loss = output["balancing_loss"]
    print('xtuner forward:', loss, balancing_loss)


if __name__ == '__main__':
    debug = False
    # debug = True
    if debug:
        import debugpy
        debugpy.connect(('10.103.23.59', 5680))

    path='/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
    processor = AutoProcessor.from_pretrained(path,trust_remote_code=True)
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你觉得上海如何？请推荐几个景点？"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages1,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        # padding=True,
    )
    inputs = inputs.to('cuda')
    
    # forward_hf(path, inputs)
    forward_xtuenr(path,inputs)
