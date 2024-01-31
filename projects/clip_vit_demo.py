from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)
import torch
import torch.nn as nn

visual_encoder_name_or_path = '/home/PJLAB/huanghaian/models--openai--clip-vit-large-patch14-336/snapshots' \
                              '/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
visual_encoder = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path=visual_encoder_name_or_path)

input_size = 672
backbone_output_stride = 14
backbone_output_channel = visual_encoder.config.hidden_size
sliding_window_stride = 336
sliding_window_size = 336
h_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
w_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
window_pos_embed = nn.Parameter(torch.randn(1, (input_size // backbone_output_stride) ** 2,
                                            visual_encoder.config.hidden_size))  # 1,5193,2048

visual_encoder.requires_grad_(False)


def sliding_window_vit_forward(pixel_values):
    batch_size = pixel_values.shape[0]
    output_features = torch.zeros(
        (batch_size, input_size // backbone_output_stride, input_size // backbone_output_stride,
         backbone_output_channel), dtype=pixel_values.dtype, device=pixel_values.device
    )
    counters = torch.zeros(
        (batch_size, input_size // backbone_output_stride, input_size // backbone_output_stride,
         1), dtype=pixel_values.dtype, device=pixel_values.device
    )

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * sliding_window_stride
            x1 = w_idx * sliding_window_stride
            y2 = min(y1 + sliding_window_size, input_size)
            x2 = min(x1 + sliding_window_size, input_size)
            y1 = max(y2 - sliding_window_size, 0)
            x1 = max(x2 - sliding_window_size, 0)
            cur_pixel_values = pixel_values[..., y1:y2, x1:x2]

            cur_visual_outputs = visual_encoder(cur_pixel_values, output_hidden_states=True)
            last_hidden_state = cur_visual_outputs.hidden_states[-2][:, 1:]

            output_features[:, y1 // backbone_output_stride:y2 // backbone_output_stride,
            x1 // backbone_output_stride:x2 // backbone_output_stride] += last_hidden_state.view(
                batch_size, sliding_window_size // backbone_output_stride,
                            sliding_window_size // backbone_output_stride, -1)
            counters[:, y1 // backbone_output_stride:y2 // backbone_output_stride,
            x1 // backbone_output_stride:x2 // backbone_output_stride] += 1

    output_features /= counters
    encoded_pixel_features = output_features.view(batch_size, -1, backbone_output_channel)
    return encoded_pixel_features


with torch.no_grad():
    visual_outputs = sliding_window_vit_forward(torch.ones([1, 3, input_size, input_size]))
    visual_outputs += window_pos_embed
    bs, pn, hs = visual_outputs.shape
    # token merge
    visual_outputs = visual_outputs.view(bs, int(pn / 4), int(hs * 4))
    print(visual_outputs.shape)  # 1, 576, 4096
