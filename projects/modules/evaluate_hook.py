from xtuner.engine.hooks import EvaluateChatHook
import torch
from mmengine.model import is_model_wrapper
from xtuner.dataset.utils import expand2square
import torchvision.transforms.functional as F


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
                inputs = (self.system + self.instruction).format(
                    input=sample_input, round=1, **runner.cfg)
                input_ids = self.tokenizer.encode(inputs)
                input_ids = torch.tensor(input_ids).to(device)
                with torch.no_grad():
                    mm_inputs = model.prepare_for_eval({'input_ids': input_ids.unsqueeze(0),
                                                        'pixel_values': image.unsqueeze(0)})
                    generation_output = model.generate(
                        **mm_inputs,
                        max_new_tokens=max_new_tokens,
                        generation_config=self.gen_config,
                        bos_token_id=self.tokenizer.bos_token_id,
                        stopping_criteria=self.stop_criteria)
                runner.logger.info(
                    f'Sample output:\n'
                    f'{inputs + self.tokenizer.decode(generation_output[0])}\n'
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
