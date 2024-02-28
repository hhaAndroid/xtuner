from xtuner.engine.hooks import EvaluateChatHook
import torch
from mmengine.model import is_model_wrapper
from xtuner.dataset.utils import expand2square
import torchvision.transforms.functional as F

from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np

sam_pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
sam_pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
sam_transform = ResizeLongestSide(1024)


class RRREvaluateChatHook(EvaluateChatHook):
    def __init__(self, *args, img_size=(672, 672), **kwargs):
        self.img_size = img_size
        super().__init__(*args, **kwargs)

    def _generate_samples(self, runner, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        is_checkpointing = model.llm.is_gradient_checkpointing
        use_cache = model.llm.config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        model.llm.config.use_cache = True
        model.eval()
        if self.evaluation_images is not None:
            for sample_image, sample_input in zip(self.evaluation_images,
                                                  self.evaluation_inputs):
                old_w, old_h = F.get_image_size(sample_image)
                if model.use_sam:
                    mask_data_dict = {'orig_size': [(old_h, old_w)]}  # 考虑 bs
                    sam_image = sam_transform.apply_image(np.array(sample_image))
                    padding_h, padding_w = sam_image.shape[:2]  # 网络训练的输入尺寸,不包括 padding 部分
                    mask_data_dict['padding_size'] = [(padding_h, padding_w)]
                    sam_image = self.sam_preprocess(torch.from_numpy(sam_image).permute(2, 0, 1).contiguous())
                    mask_data_dict['sam_pixel_values'] = sam_image.unsqueeze(0).to(device)  # 考虑 bs

                scale_factor = min(self.img_size[0] / max(old_h, old_w),
                                   self.img_size[0] / min(old_h, old_w))
                neww = int(old_w * float(scale_factor) + 0.5)
                newh = int(old_h * float(scale_factor) + 0.5)
                image = F.resize(sample_image, size=(newh, neww), interpolation=F.InterpolationMode.BICUBIC)
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                image = image.to(device)
                if isinstance(sample_input, list):
                    gt_bboxes = sample_input[1]
                    sample_input = sample_input[0]

                inputs = (self.system + self.instruction).format(
                    input=sample_input, round=1, **runner.cfg)
                input_ids = self.tokenizer.encode(inputs)
                input_ids = torch.tensor(input_ids).to(device)

                input_dict = {'input_ids': input_ids.unsqueeze(0),
                              'pixel_values': image.unsqueeze(0)}
                if model.sampler is not None:
                    # with bbox inputs
                    input_dict['gt_bboxes_masks'] = [gt_bboxes]  # batch

                # 设置也是无效的，因为我们输入到模型的是 input_embeds，此时 hf 不允许设置为 false
                self.gen_config.use_cache = False
                with torch.no_grad():
                    mm_inputs = model.prepare_for_eval(input_dict)
                    generation_output = model.generate(
                        **mm_inputs,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        max_new_tokens=max_new_tokens,
                        generation_config=self.gen_config,
                        bos_token_id=self.tokenizer.bos_token_id,
                        stopping_criteria=self.stop_criteria)
                    pred_masks = model.postprocess_for_eval(generation_output, mask_data_dict)

                    # TODO 可视化结果保存

                runner.logger.info(
                    f'Sample output:\n'
                    f'{inputs + self.tokenizer.decode(generation_output.sequences[0])}\n'
                )
        else:
            for sample_input in self.evaluation_inputs:
                inputs = (self.system + self.instruction).format(
                    input=sample_input, round=1, **runner.cfg)
                input_ids = self.tokenizer.encode(inputs, return_tensors='pt')
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        generation_config=self.gen_config,
                        stopping_criteria=self.stop_criteria)
                runner.logger.info(
                    f'Sample output:\n'
                    f'{self.tokenizer.decode(generation_output[0])}\n')

        # Cast to training mode
        if is_checkpointing:
            model.activation_checkpointing_enable()
        model.llm.config.use_cache = use_cache
        model.train()

    def sam_preprocess(self, x: torch.Tensor, img_size=1024) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - sam_pixel_mean) / sam_pixel_std

        # Pad
        h, w = x.shape[-2:]  # 往后 padding
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

