import pandas
import csv
from PIL import Image
import base64
import os
import os.path as osp
from uuid import uuid4


def encode_image_file_to_base64(image_path, target_size=-1):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size)


def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret


root = '/mnt/petrelfs/huanghaian/code/xtuner/LMUData/'
path = root + 'MME.tsv'

df = pandas.read_csv(path, sep='\t')
for idx in range(len(df)):
    i_dict = df.iloc[idx]
    img_path = i_dict['image_path']
    img_path = osp.join('/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release', img_path)
    img_b64 = encode_image_file_to_base64(img_path)
    df.at[idx, 'image'] = img_b64

df.to_csv(path, sep='\t', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
