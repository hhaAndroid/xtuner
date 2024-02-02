from segment_anything import build_sam_vit_h
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn

pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def preprocess(x: torch.Tensor, img_size=1024) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]  # 往后 padding
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def get_visual_embs(pixel_values: torch.FloatTensor, visual_model):
    with torch.no_grad():
        image_embeddings_list = []
        for i in range(pixel_values.shape[0]):
            torch.cuda.empty_cache()
            image_embeddings = visual_model.image_encoder(
                pixel_values[i].unsqueeze(0)
            )  # stride 16
            image_embeddings_list.append(image_embeddings)
        torch.cuda.empty_cache()
        image_embeddings = torch.cat(image_embeddings_list, 0)
    return image_embeddings

# 模型部分
vision_pretrained = None
visual_model = build_sam_vit_h(vision_pretrained)
for param in visual_model.parameters():
    param.requires_grad = False

visual_model.mask_decoder.train()
for param in visual_model.mask_decoder.parameters():
    param.requires_grad = True

in_dim = 4096
out_dim = 256

# llm 输出特征投影到 sam decoder 输入维度
text_fc = [
    nn.Linear(in_dim, in_dim),
    nn.ReLU(inplace=True),
    nn.Linear(in_dim, out_dim),
    nn.Dropout(0.0),
]
text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
text_hidden_fcs.train()
for param in text_hidden_fcs.parameters():
    param.requires_grad = True

transform = ResizeLongestSide(1024)

image_path = '/home/PJLAB/huanghaian/both.png'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ori_size = image.shape[:2]  # height, width
label_list = [torch.zeros((ori_size[0], ori_size[1]))]  # mask label shape，最原始图片尺度

image = transform.apply_image(image)
resize_list = [image.shape[:2]]  # 网络训练的输入尺寸

image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
print('网络输入 shape:', image.shape)

image = image[None]  # 单张图片转 bs

image_embeddings = get_visual_embs(image, visual_model)

# 先假装提取了 <SEG> Token 对应位置的隐含层预测 位置是 2
output_hidden_states = [torch.randn(1, 576, 4096)]

last_hidden_state = text_hidden_fcs[0](output_hidden_states[-1])
pred_embeddings = last_hidden_state[0:1, 2:3, :]

multimask_output = False
pred_masks = []
for i in range(len(pred_embeddings)):  # bs 维度
    (
        sparse_embeddings,
        dense_embeddings,
    ) = visual_model.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
        text_embeds=pred_embeddings[i].unsqueeze(1),
    )
    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
    low_res_masks, iou_predictions = visual_model.mask_decoder(
        image_embeddings=image_embeddings[i].unsqueeze(0),
        image_pe=visual_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
    )
    pred_mask = visual_model.postprocess_masks(
        low_res_masks,
        input_size=resize_list[i],
        original_size=label_list[i].shape,
    )
    pred_masks.append(pred_mask[:, 0])

print('预测 mask shape: ', pred_masks[0].shape)
