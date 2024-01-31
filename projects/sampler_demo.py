from projects.modules import GeoRegionSampler
import torch

sampler = GeoRegionSampler(4096,
                           4096,
                           512,
                           num_sub_point=[128, 32],
                           num_neighbor=[24, 24])

visual_outputs = torch.randn((2, 576, 4096))
bbox = [[100, 100, 234, 344], [50, 23, 100, 200]]
raw_w = 672
raw_h = 672

region_mask = []
for b in bbox:
    coor_mask = torch.zeros((raw_w, raw_h))
    coor_mask[b[0]:b[2], b[1]:b[3]] = 1
    assert len(coor_mask.nonzero()) != 0
    region_mask.append([coor_mask])  # 可以运行每张图片存在多个 bbox 的情况，因此外层会多一个 []

region_feats = sampler(visual_outputs, region_mask)
print(region_feats[0].shape, region_feats[1].shape)
