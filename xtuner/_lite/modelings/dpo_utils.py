from dataclasses import dataclass
import dataclasses
import torch
import json
from typing import Tuple
from mmengine.dist import is_distributed, is_main_process, all_reduce


# copy from https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
@dataclass
class RunningMoments:
    """
    Calculates the running mean and standard deviation of a data stream. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
    """

    mean: float = 0
    std: float = 1
    var: float = 1
    count: float = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if is_distributed():
            xs_mean, xs_var, xs_count = get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count).item()
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).float().sqrt().item()
        self.var = new_var.item()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        # save everything except accelerator
        if is_main_process():
            save_dict = dataclasses.asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if k != "accelerator"})
            json_string = json.dumps(save_dict, indent=2, sort_keys=True) + "\n"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        # load everything except accelerator
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

@torch.no_grad()
def get_global_statistics(
    xs: torch.Tensor, mask=None, device="cpu"
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    all_reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    all_reduce(sum_var)
    global_var = sum_var / count

    return global_mean.to(device), global_var.to(device), count.item()
