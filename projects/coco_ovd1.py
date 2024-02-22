import argparse
import json
import random
from pycocotools.coco import COCO
from tqdm import tqdm

# 可以输入单个区域或者多个区域
# 为了减少输入序列长度，坐标值就不输入了

IMAGE_SIZE = 672
MIN_BBOX_SIZE = 40
MAX_REGION_NUM = 30

OVD_TEMPLATE = [
    "In the conversation below, you simply answer the category name based on what you see in the imagery inside a "
    "region. If you don't find the category name in the provided list of categories, you should output other. "
    "I will input one or multiple regions and provide a list of categories. Please reply strictly in the order of the "
    "input. Categories: <c>. Region: <r>."
]

OUT_TEMPLATE_ONE_REGION = [
    "Sure. It's <p><c></p>.",
    "This is <p><c></p>."
]

OUT_TEMPLATE_MANY_REGION = [
    "Sure, the categories for these regions are as follows: <p><c></p>."
]


def coco2ovd(args):
    coco = COCO(args.input)
    cats = coco.loadCats(coco.getCatIds())
    names = {cat['id']: cat['name'] for cat in cats}
    all_name = list(names.values())

    if args.output is None:
        out_path = args.input[:-5] + '_rrrvlm_ovd1.json'
    else:
        out_path = args.output

    out_datas = []
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

                # TODO 对应的处理 mask
                new_ann_info.append({'bbox': bbox_xyxy, 'label': names[ann['category_id']]})

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
            out_data = {'id': img_id, 'image': img_info['file_name']}

            all_bboxes = [ann['bbox'] for ann in sample_ann_info]
            all_names = [ann['label'] for ann in sample_ann_info]
            out_data['bbox'] = all_bboxes
            out_data['name'] = all_names

            out_str = ''
            for i in range(len(all_bboxes)):
                if i != len(all_bboxes)-1:
                    out_str += f'{i + 1}.<region_feat>;'
                else:
                    out_str += f'{i + 1}.<region_feat>'

            temp = random.choice(OVD_TEMPLATE)
            temp = temp.replace('<r>', out_str)

            # 50 概率保留所有类类别
            if random.random() > 0.5:
                cat_name = all_name
            else:
                # 随机选择 3 ～ 80 个类别
                num_indices = random.randint(3, len(all_name))
                cat_name = random.sample(all_name, num_indices)
                #  gt label 必须要在随机类别里面
                for gt_name in all_names:
                    if gt_name not in cat_name:
                        cat_name.append(gt_name)
            random.shuffle(cat_name)
            cat_name = ','.join(cat_name)
            temp = temp.replace('<c>', cat_name)

            if len(all_bboxes) == 1:
                out_temp = random.choice(OUT_TEMPLATE_ONE_REGION)
            else:
                out_temp = random.choice(OUT_TEMPLATE_MANY_REGION)

            out_str = ''
            for i in range(len(all_names)):
                if len(all_bboxes) == 1:
                    out_str += f'{all_names[i]}'
                else:
                    if i != len(all_names)-1:
                        out_str += f'{i + 1}.{all_names[i]};'
                    else:
                        out_str += f'{i + 1}.{all_names[i]}'

            out_temp = out_temp.replace('<c>', out_str)

            out_data['conversations'] = [{'from': 'human', 'value': '<image>\n' + temp},
                                         {'from': 'gpt', 'value': out_temp}]
            out_datas.append(out_data)

            total_num += 1
            if total_num >= args.num:
                break

    with open(out_path, 'w') as file:
        json.dump(out_datas, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to ovd format.', add_help=True)
    parser.add_argument('--input', default='/home/PJLAB/huanghaian/dataset/coco/annotations/instances_train2017.json',
                        type=str, help='input json file name')
    parser.add_argument('--num', '-n', type=int, default=100)
    parser.add_argument(
        '--output', '-o', type=str, help='output json file name')
    args = parser.parse_args()

    coco2ovd(args)
