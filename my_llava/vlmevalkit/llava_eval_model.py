from vlmeval.vlm.base import BaseModel
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE


class LLaVAEvalModel(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_pth,  **kwargs):
        super().__init__()
        assert osp.exists(model_pth)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_pth)
        self.processor = AutoProcessor.from_pretrained(model_pth)
        self.model = self.model.cuda()
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1,
                              use_cache=True)  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

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
        # msg2 = [
        #     dict(type='image', value=IMAGE_URL),
        #     dict(type='text', value='How many apples are there in these images?')
        # ]
        texts, images = [], []
        for msg in message:
            if msg['type'] == 'text':
                texts.append(msg['value'])
            elif msg['type'] == 'image':
                images.append(msg['value'])
        assert len(texts) == 1 and len(images) == 1

        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": texts[0]},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(images[0]).convert('RGB')
        inputs = self.processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self.kwargs)
            output = self.processor.decode(output[0][2:], skip_special_tokens=True).strip()
        return output
