import argparse
import json
import random
from pycocotools.coco import COCO
from tqdm import tqdm
import torch
import numpy as np
from projects.modules.utils import merge_multi_segment
import os.path as osp
from mmengine.visualization import Visualizer
from PIL import Image
import torchvision.transforms.functional as F
from xtuner.dataset.utils import expand2square
import copy


IMAGE_SIZE = 672
MIN_BBOX_SIZE = 40
MAX_REGION_NUM = 30

OVD_TEMPLATE_ONE = [
    "What is the class of the region <region_feat> within the image?",
    "Classify region <region_feat> in the image.",
    "Identify the region <region_feat> in the image.",
    "Describe the region <region_feat> in a short phrase.",
    "What is in the region <region_feat>? Describe in a phrase.",
    "Capture in a phrase: what's near region <region_feat> in the picture?"
]

OUT_TEMPLATE_ONE = [
    "It's <p><c></p>.",
    "The region is <p><c></p>.",
    "This is <p><c></p>.",
    "This region is <p><c></p>.",
]

OVD_TEMPLATE_MANY = [
    "In the conversation below, you simply answer the category name based on what you see in the imagery inside a "
    "region. I will input multiple regions. Please reply strictly in the order of the "
    "input. Region: <r>."
]

OUT_TEMPLATE_MANY = [
    "Sure, the categories for these regions are as follows: <p><c></p>."
]


def coco2ovd(args):
    coco = COCO(args.input)
    cats = coco.loadCats(coco.getCatIds())
    names = {cat['id']: cat['name'] for cat in cats}

    with_mask = True

    # 同时生成 mask 和 bbox 两份数据，方便做对比实验
    if args.output is None:
        out_mask_json_path = args.input[:-5] + '_rrrvlm_region1_mask.json'
        out_bbox_json_path = args.input[:-5] + '_rrrvlm_region1_bbox.json'
        out_mask_pth_path = args.input[:-5] + '_rrrvlm_region1_mask.pth'
    else:
        out_mask_json_path = args.output
        out_bbox_json_path = args.output[:-5] + '_bbox.json'
        out_mask_pth_path = args.output[:-5] + '.pth'

    out_datas = []
    out_masks = []
    # 对每一张图片进行操作
    img_ids = coco.getImgIds()
    random.shuffle(img_ids)
    total_num = 0
    while total_num < args.num:
        for img_id in img_ids:
            img_info = coco.loadImgs([img_id])[0]
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            ann_info = coco.loadAnns(ann_ids)

            # 过滤无效数据
            new_ann_info = []
            for i, ann in enumerate(ann_info):
                if ann.get('ignore', False) or ann.get('iscrowd', False):
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue

                bbox_xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]

                # Resize to IMAGE_SIZE
                old_w = img_info['width']
                old_h = img_info['height']
                scale_factor = min(IMAGE_SIZE / max(old_h, old_w),
                                   IMAGE_SIZE / min(old_h, old_w))
                neww = int(old_w * float(scale_factor) + 0.5)
                newh = int(old_h * float(scale_factor) + 0.5)

                if neww > newh:
                    padding_h = (neww - newh) // 2
                    padding_w = 0
                else:
                    padding_w = (newh - neww) // 2
                    padding_h = 0
                bbox_xyxy = [
                    int(bbox_xyxy[0] * neww / img_info['width']) + padding_w,
                    int(bbox_xyxy[1] * newh / img_info['height']) + padding_h,
                    int(bbox_xyxy[2] * neww / img_info['width']) + padding_w,
                    int(bbox_xyxy[3] * newh / img_info['height']) + padding_h,
                ]

                # 过滤掉特别小的框
                new_h = bbox_xyxy[3] - bbox_xyxy[1]
                new_w = bbox_xyxy[2] - bbox_xyxy[0]
                if new_h < MIN_BBOX_SIZE or new_w < MIN_BBOX_SIZE:
                    continue

                new_data={'bbox': bbox_xyxy, 'label': names[ann['category_id']]}

                if with_mask:
                    gt_mask = ann['segmentation']
                    if isinstance(gt_mask, list):
                        gt_mask = [
                            np.array(polygon) for polygon in gt_mask
                            if len(polygon) % 2 == 0 and len(polygon) >= 6
                        ]
                        if len(gt_mask) == 0:
                            # ignore
                            continue
                        else:
                            if len(gt_mask) > 1:
                                new_gt_masks = merge_multi_segment(gt_mask)
                            else:
                                new_gt_masks = gt_mask[0]
                    else:
                        raise NotImplementedError(
                            'Only supports mask annotations in polygon '
                            'format currently')

                    # sam 计算 loss 时候需要一份原始尺度的
                    new_data['sam_mask'] = copy.deepcopy(new_gt_masks)

                    # 和 bbox 一样进行数值变换
                    w_scale = neww / img_info['width']
                    h_scale = newh / img_info['height']
                    new_gt_masks[0::2] = new_gt_masks[0::2] * w_scale
                    new_gt_masks[1::2] = new_gt_masks[1::2] * h_scale
                    new_gt_masks[0::2] = new_gt_masks[0::2] + padding_w
                    new_gt_masks[1::2] = new_gt_masks[1::2] + padding_h
                    new_data['mask'] = new_gt_masks
                new_ann_info.append(new_data)

            if len(new_ann_info) == 0:
                continue

            # 随机选择 1 ～ MAX_REGION_NUM 个区域
            if len(new_ann_info) <= MAX_REGION_NUM:
                max_region_num = len(new_ann_info)
            else:
                max_region_num = MAX_REGION_NUM
            num_indices = random.randint(1, max_region_num)
            sample_ann_info = random.sample(new_ann_info, num_indices)
            # 打乱顺序
            random.shuffle(sample_ann_info)

            # 提取信息
            out_data = {'id': str(img_id)+"_"+str(total_num), 'image': img_info['file_name']}

            all_bboxes = [ann['bbox'] for ann in sample_ann_info]
            all_names = [ann['label'] for ann in sample_ann_info]
            out_data['bbox'] = all_bboxes
            out_data['name'] = all_names

            if with_mask:
                all_masks = [ann['mask'] for ann in sample_ann_info]
                all_sam_masks = [ann['sam_mask'] for ann in sample_ann_info]
                out_masks.append({'id': str(img_id)+"_"+str(total_num), 'mask': all_masks, 'sam_mask': all_sam_masks})

            if len(all_bboxes) == 1:
                temp = random.choice(OVD_TEMPLATE_ONE)
                out_temp = random.choice(OUT_TEMPLATE_ONE)
                out_temp = out_temp.replace('<c>', all_names[0])
            else:
                temp = random.choice(OVD_TEMPLATE_MANY)

                out_str = ''
                for i in range(len(all_bboxes)):
                    if i != len(all_bboxes) - 1:
                        out_str += f'{i + 1}.<region_feat>;'
                    else:
                        out_str += f'{i + 1}.<region_feat>'
                temp = temp.replace('<r>', out_str)

                out_temp = random.choice(OUT_TEMPLATE_MANY)
                out_str = ''
                for i in range(len(all_names)):
                    if len(all_bboxes) == 1:
                        if with_mask:
                            out_str += f'{all_names[i]}<seg>'
                        else:
                            out_str += f'{all_names[i]}'
                    else:
                        if i != len(all_names) - 1:
                            if with_mask:
                                out_str += f'{i + 1}.{all_names[i]}<seg>;'
                            else:
                                out_str += f'{i + 1}.{all_names[i]};'
                        else:
                            if with_mask:
                                out_str += f'{i + 1}.{all_names[i]}<seg>'
                            else:
                                out_str += f'{i + 1}.{all_names[i]}'

                out_temp = out_temp.replace('<c>', out_str)
            out_data['conversations'] = [{'from': 'human', 'value': '<image>\n' + temp},
                                         {'from': 'gpt', 'value': out_temp}]
            out_datas.append(out_data)

            total_num += 1
            if total_num >= args.num:
                break

    assert len(out_datas) == len(out_masks)
    torch.save(out_masks, out_mask_pth_path)

    with open(out_mask_json_path, 'w') as file:
        json.dump(out_datas, file)

    # 同时保存一份 bbox 数据，方便对比实验
    for data in out_datas:
        num_mask = len(data['bbox'])
        gpt_data = data['conversations'][1]['value']
        gpt_data = gpt_data.replace('<seg>', '')
        data['conversations'][1]['value'] = gpt_data
    with open(out_bbox_json_path, 'w') as file:
        json.dump(out_datas, file)


def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    scales = 0.5 + (areas - min_area) // (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def show(args):
    vis = Visualizer()

    with_mask = True

    # 同时生成 mask 和 bbox 两份数据，方便做对比实验
    if args.output is None:
        out_path = args.input[:-5] + '_rrrvlm_region1_mask.json'
        out_mask_path = args.input[:-5] + '_rrrvlm_region1_mask.pth'
    else:
        out_path = args.output
        out_mask_path = args.output[:-5] + '.pth'

    json_data = json.load(open(out_path))

    if with_mask:
        masks = torch.load(out_mask_path)
        combined = list(zip(json_data, masks))
        random.shuffle(combined)
        json_data, masks = zip(*combined)
    else:
        random.shuffle(json_data)

    for i, data in enumerate(json_data):
        image_name = data['image']
        directory = osp.dirname(args.input)
        image_path = osp.join(directory, '..', args.img_prefix, image_name)
        image = Image.open(image_path).convert('RGB')
        old_w, old_h = F.get_image_size(image)
        scale_factor = min(IMAGE_SIZE / max(old_h, old_w),
                           IMAGE_SIZE / min(old_h, old_w))
        neww = int(old_w * float(scale_factor) + 0.5)
        newh = int(old_h * float(scale_factor) + 0.5)
        image = F.resize(image, size=(newh, neww), interpolation=F.InterpolationMode.BICUBIC)
        image = expand2square(image, 0)

        vis.set_image(np.array(image))

        bboxes = data['bbox']
        name = data['name']

        image2colors = []
        for _ in range(len(bboxes)):
            colors = np.random.random((1, 3)) * 0.7 + 0.3
            colors = (colors * 255).astype(int).tolist()[0]
            image2colors.append(tuple(colors))

        bboxes = np.array(bboxes).reshape(-1, 4)
        positions = bboxes[:, :2] + 3
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        vis.draw_bboxes(bboxes, edge_colors=image2colors, line_widths=3)
        vis.draw_texts(
            name,
            positions,
            colors='g',
            font_sizes=[int(13 * s) for s in scales],
            bboxes=[{
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            }] * len(scales))

        if with_mask:
            mask = masks[i]['mask']
            for i, m in enumerate(mask):
                vis.draw_polygons(m.reshape(-1, 2), edge_colors='w', face_colors=image2colors[i])

        vis.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to ovd format.', add_help=True)
    parser.add_argument('--input',
                        default='data/coco/annotations/instances_train2017.json',
                        type=str, help='input json file name')
    parser.add_argument(
        '--img-prefix', type=str, default='train2017')
    parser.add_argument('--num', '-n', type=int, default=300000)  # 300000
    parser.add_argument(
        '--output', '-o', type=str, help='output json file name')
    parser.add_argument(
        '--show', '-s', action='store_true', help='Whether to add mask')
    args = parser.parse_args()

    coco2ovd(args)

    # show data
    if args.show:
        show(args)
