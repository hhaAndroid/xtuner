import argparse
import json
import random
from pycocotools.coco import COCO

IMAGE_SIZE = 532
MIN_BBOX_SIZE = 32

GLOABL_TEMPLATE = 'In images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. Increasing x ' \
                  f'moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: {IMAGE_SIZE}x{IMAGE_SIZE}.'
OVD_TEMPLATE = [
    "In the conversation below, you simply answer the category name based on what you see in the imagery inside a "
    "region <region>. If you don't find the category name in the provided list of categories, you should output other. "
    "Categories Containing {}.",
    "In the ensuing discussion, your task is to identify the category name by analyzing the imagery within a specific "
    "region <region>. If you don't find the category name in the provided list of categories, you should output other. "
    "Categories will consist of {}.",
    "In the dialogue that follows, your task is to identify the category name by observing the visual elements within "
    "a specific region <region>. If you don't find the category name in the provided list of categories, you should "
    "output other. Categories include {}.",
    "During the conversation that follows, your role is to determine the category name by examining the visual "
    "representation within a designated region <region>. If you don't find the category name in the provided list of "
    "categories, you should output other. Categories will contain {}."
]

OUT_TEMPLATE = [
    "It's <category>.",
    "This is <category>."
]


def coco2ovd(args):
    coco = COCO(args.input)
    cats = coco.loadCats(coco.getCatIds())
    names = {cat['id']: cat['name'] for cat in cats}
    all_name = list(names.values())

    if args.output is None:
        out_path = args.input[:-5] + '_rrrvlm_ovd.json'
    else:
        out_path = args.output

    # 确保类别均衡
    out_data = []
    num_instance_pre_cat = args.num // len(names)
    for id, name in names.items():
        # 从每个类里面随机抽取num_instance_pre_cat个instance
        _num_instance = 0
        ids = coco.getImgIds(catIds=[id])
        random.shuffle(ids)

        ann_ids_set = set()
        while _num_instance < num_instance_pre_cat:
            for img_id in ids:
                img_info = coco.loadImgs([img_id])[0]
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[id])
                if len(ann_ids) > 0:
                    # 随机抽一个实例, 如果没有 bbox 或者没有符合的则跳过
                    ann_id = random.choice(ann_ids)
                    if ann_id in ann_ids_set:
                        continue

                    ann = coco.loadAnns(ids=[ann_id])[0]
                    if ann.get('ignore', False):
                        continue
                    x1, y1, w, h = ann['bbox']
                    inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                    inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if ann['area'] <= 0 or w < 1 or h < 1:
                        continue
                    if ann.get('iscrowd', False):
                        continue
                    bbox_xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]

                    # resize to IMAGE_SIZE
                    old_w = img_info['width']
                    old_h = img_info['height']
                    scale_factor = min(IMAGE_SIZE / max(old_h, old_w),
                                       IMAGE_SIZE / min(old_h, old_w))
                    neww = int(old_w * float(scale_factor))
                    newh = int(old_h * float(scale_factor))

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

                    temp = random.choice(OVD_TEMPLATE)
                    temp = temp.replace('<region>', str(bbox_xyxy) + '<region_feat><seg>')

                    # 50 概率保留所有类类别
                    if random.random() > 0.5:
                        cat_name = all_name
                    else:
                        # 随机选择 3 ～ 80 个类别
                        num_indices = random.randint(3, len(all_name))
                        cat_name = random.sample(all_name, num_indices)
                        #  gt label 必须要在随机类别里面
                        if name not in cat_name:
                            cat_name.append(name)
                    random.shuffle(cat_name)
                    cat_name = ','.join(cat_name)
                    temp = temp.replace('{}', cat_name)

                    out_temp = random.choice(OUT_TEMPLATE)
                    out_temp = out_temp.replace('<category>', name)

                    data = {'id': img_id, 'image': img_info['file_name'], 'bbox': bbox_xyxy,
                            'conversations': [{'from': 'human', 'value': '<image>\n' + temp},
                                              {'from': 'gpt', 'value': out_temp}]}
                    out_data.append(data)
                    # 下一次循环迭代时候不能再次抽到这个实例
                    ann_ids_set.add(ann_id)
                    _num_instance += len(ann_ids)

                    if _num_instance >= num_instance_pre_cat:
                        break

    with open(out_path, 'w') as file:
        json.dump(out_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to ovd format.', add_help=True)
    parser.add_argument('--input', default='/home/PJLAB/huanghaian/dataset/coco/annotations/instances_train2017.json',
                        type=str, help='input json file name')
    parser.add_argument('--num', '-n', type=int, default=100)
    parser.add_argument(
        '--output', '-o', type=str, help='output json file name')
    args = parser.parse_args()

    coco2ovd(args)
