from xtuner.dataset.llava import LLaVADataset
import json
import torch
from torchvision.ops import batched_nms


class BoxLLaVADataset(LLaVADataset):
    def __init__(self, *args, box_json_path, iou_threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_data = json.load(open(box_json_path))
        self.json_data = {item['id']: item for item in self.json_data}
        self.iou_threshold = iou_threshold

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data_dict = self.text_data[idx]
        if data_dict.get('image', None) is not None:
            box_data = self.json_data[data_dict['id']]
            boxes = box_data['boxes']
            labels = box_data['labels']
            scores = box_data['scores']

            # nms
            char_to_index = {char: index for index, char in enumerate(set(labels))}
            index_list = [char_to_index[char] for char in labels]
            boxes = torch.tensor(boxes).reshape(-1, 4)
            scores = torch.tensor(scores)
            labels = torch.tensor(index_list)

            keep = batched_nms(boxes, scores, labels, self.iou_threshold)
            boxes = boxes[keep]
            labels = labels[keep]
            data['boxes'] = boxes
            data['labels'] = labels
        else:
            data['boxes'] = torch.tensor([])
            data['labels'] = torch.tensor([])
        return data
