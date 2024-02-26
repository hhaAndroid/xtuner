import numpy as np
from typing import Tuple, List
import pycocotools.mask as maskUtils

def min_index(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
    """Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        tuple: a pair of indexes.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(gt_masks: List[np.ndarray]) -> List[np.ndarray]:
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
    idx_list = [[] for _ in range(len(gt_masks))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    # first round: first to end, i.e. A->B(partial)->C
    # second round: end to first, i.e. C->B(remaining)-A
    for k in range(2):
        # forward first round
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                # add the idx[0] point for connect next segment
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate(
                    [segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                # deal with the middle segment
                # Note that in the first round, only partial segment
                # are appended.
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])
        # forward second round
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                # deal with the middle segment
                # append the remaining points
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return np.concatenate(s).reshape(-1, )


def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(bool)
    return bitmap_mask
