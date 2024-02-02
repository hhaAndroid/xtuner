# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from mmengine.config import Config, DictAction
from xtuner.registry import BUILDER
import torch
from transformers import GenerationConfig, StoppingCriteriaList
from mmengine.visualization import Visualizer
import numpy as np
import cv2
from xtuner.utils import StopWordStoppingCriteria


def parse_args():
    parser = argparse.ArgumentParser(description='Inference LLM')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('weighs', help='model weights file name or path.')
    parser.add_argument('--output-dir', default='rrr_infernce_results', help='the dir to save logs and models')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    vis = Visualizer()
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.model.pretrained_pth = args.weighs
    model = BUILDER.build(cfg.model).to(args.device).half()
    dataset = BUILDER.build(cfg.inference_dataset)
    tokenizer = BUILDER.build(cfg.tokenizer)

    max_new_tokens = 600
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else
        tokenizer.eos_token_id,
    )
    stop_criteria = StoppingCriteriaList()
    stop_words = cfg.prompt_template.get('STOP_WORDS', [])
    for word in stop_words:
        stop_criteria.append(
            StopWordStoppingCriteria(tokenizer, word))

    mean = dataset.image_processor.image_mean
    std = dataset.image_processor.image_std
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    for i, data in enumerate(dataset):
        print(data['conversation'])
        image_file = data['image']

        new_data = {'pixel_values': data['pixel_values'].to(args.device)[None].half(),
                    'input_ids': torch.tensor(data['input_ids']).to(args.device)[None]}
        if 'bbox' in data:
            new_data['gt_bboxes'] = [data['bbox']]

        with torch.no_grad():
            mm_inputs = model.prepare_for_eval(new_data)
            generation_output = model.generate(
                **mm_inputs,
                max_new_tokens=max_new_tokens,
                generation_config=gen_config,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria)
            output = tokenizer.decode(generation_output[0])

        pixel_values = data['pixel_values']
        pixel_values = pixel_values * std + mean
        pixel_values = pixel_values * 255
        pixel_values = torch.permute(pixel_values, (1, 2, 0))

        vis.set_image(pixel_values.numpy())
        vis.draw_bboxes(np.array([data['bbox']]), edge_colors='r', line_widths=4)
        vis.draw_texts(output, np.array([10, 10]), colors='r', font_sizes=20)

        drawn_image = vis.get_image()
        base_name = osp.basename(image_file)
        name, extension = osp.splitext(base_name)
        out_file = osp.join(args.output_dir, name + '_' + str(i) + extension)
        cv2.imwrite(out_file, drawn_image[..., ::-1])


if __name__ == '__main__':
    main()
