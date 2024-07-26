from vlmeval.vlm.base import BaseModel
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, StoppingCriteriaList
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE
from xtuner._lite.modelings import register_remote_code
from xtuner.utils import StopWordStoppingCriteria


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
        inputs = self.processor(prompt, raw_image, return_tensors='pt').to('cuda', torch.float16)
        with torch.inference_mode():
            output = self.model.generate(**inputs, stopping_criteria=self.stop_criteria, **self.kwargs)
            output = self.processor.tokenizer.decode(output[0][len(inputs['input_ids'][0]):],
                                                     skip_special_tokens=True).strip()
        return output
