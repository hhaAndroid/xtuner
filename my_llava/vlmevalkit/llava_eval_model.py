from vlmeval.vlm.base import BaseModel
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, StoppingCriteriaList
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE
from xtuner._lite.modelings import register_remote_code
from xtuner.utils import StopWordStoppingCriteria

from transformers.feature_extraction_utils import BatchFeature


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LLaVAEvalModel(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_pth, stop_words=['<|im_end|>'], **kwargs):
        super().__init__()
        assert osp.exists(model_pth)
        register_remote_code()
        with torch.device('cpu'):
            self.model = LlavaForConditionalGeneration.from_pretrained(model_pth, torch_dtype=torch.float16)
            self.model.eval()
        self.model = self.model.cuda()
        self.processor = AutoProcessor.from_pretrained(model_pth, trust_remote_code=True)
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1,
                              use_cache=True)  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.processor.tokenizer, word))

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        # TODO: Support interleave text and image
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        prompt = prompt.replace('<image>', '')
        conversation = [
            {"role": "user", "content": '<image>\n' + prompt}]

        prompt = self.processor.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # prompt=prompt[len('<s>'):]
        # print(prompt)
        raw_image = Image.open(image_path).convert('RGB')
        images = expand2square(
            raw_image,
            tuple(
                int(x * 255) for x in self.processor.image_processor.image_mean))
        pixel_values = self.processor.image_processor(images, return_tensors='pt')["pixel_values"]
        text_inputs = self.processor.tokenizer(
            prompt, return_tensors='pt', padding=False, truncation=None, max_length=None
        )
        inputs = BatchFeature(data={**text_inputs, "pixel_values": pixel_values}).to('cuda', torch.float16)
        # inputs = self.processor(prompt, raw_image, return_tensors='pt').to('cuda', torch.float16)
        with torch.inference_mode():
            output = self.model.generate(**inputs, stopping_criteria=self.stop_criteria, **self.kwargs)
            output = self.processor.tokenizer.decode(output[0][len(inputs['input_ids'][0]):],
                                                     skip_special_tokens=True).strip()
        return output
