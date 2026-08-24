import os
import torch
import pandas as pd
import fire


class CompareHelper:
    def __init__(self, res_lmdeploy_dir, res_xtuner_dir, end_token_idx=-1, start_token_idx: int=0, rtol=None, atol=None):
        self.res_lmdeploy_dir = res_lmdeploy_dir
        self.res_xtuner_dir = res_xtuner_dir
        self.rtol = rtol
        self.atol = atol
        self.stats = []
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx

    def _load_tensor(self, name, is_lmdeploy=True):
        workdir = self.res_lmdeploy_dir if is_lmdeploy else self.res_xtuner_dir
        file = os.path.join(workdir, f'step0.{name}.pt')
        assert os.path.exists(file), file
        final_tensor = torch.load(file, weights_only=False)
        final_tensor = final_tensor[:, self.start_token_idx:self.end_token_idx]
        return final_tensor
    
    def load_data(self, name):
        t_lmdeploy = self._load_tensor(name, True)
        t_xtuner = self._load_tensor(name, False)
        print(f'tensor_lmdeploy {(t_lmdeploy.shape, t_xtuner.dtype)} vs {(t_xtuner.shape, t_xtuner.dtype)}')
        if t_lmdeploy.isnan().sum() != 0:
            print(f'!Found nan in {name} from lmdeploy {(t_lmdeploy.shape, t_lmdeploy.dtype, t_lmdeploy.device)}')
        if t_xtuner.isnan().sum() != 0:
            print(f'!Found nan in {name} from xtuner {(t_xtuner.shape, t_xtuner.dtype, t_xtuner.device)}')
        return t_lmdeploy, t_xtuner

    def get_stats(self, name, ref, src):
        ref_abs = ref.abs()
        src_abs = src.abs()
        abs_diff = (src - ref).abs()
        ref_abs_mean = ref_abs.float().mean()
        ref_abs_max = ref_abs.max()
        src_abs_mean = src_abs.float().mean()
        src_abs_max = src_abs.max()
        abs_diff_mean = abs_diff.float().mean()
        abs_diff_max = abs_diff.max()
        rel_diff = (src - ref) / (ref_abs + 1e-8)
        rel_diff_mean = rel_diff.mean()
        rel_diff_max = rel_diff.max()
        print(f'avg abs ({src_abs_mean:.8f}, {ref_abs_mean:.8f}) max abs=({src_abs_max:.8f}, {ref_abs_max:.8f})')
        print(f'avg abs diff {abs_diff_mean:.8f} max abs diff={abs_diff_max:.8f}')
        print(f'avg rel diff {rel_diff_mean:.8f} max rel diff={rel_diff_max:.8f}')
        out = [src_abs_mean, ref_abs_mean, src_abs_max, ref_abs_max, abs_diff_mean, abs_diff_max, rel_diff_mean, rel_diff_max]
        out = [i.cpu().item() for i in out]
        return out


    def compare_tensor(self, name, rtol=1e-2, atol=1e-5):
        print(f'>>>>-------- {name:<40}')
        l, x = self.load_data(name)
        if l.dtype != x.dtype:
            print(f'Changed l.dtype from {l.dtype} to {x.dtype}')
            l = l.to(dtype=x.dtype)
        try :
            torch.testing.assert_close(l, x, rtol=rtol, atol=atol, equal_nan=False)
        except AssertionError as e:
            print(e)
        return self.get_stats(name, l, x)

    def check_info(self, name):
        l, x = self.load_data(name)
        print(name, l.shape, l.dtype, x.shape, x.dtype)



def main(res_lmdeploy_dir: str, res_xtuner_dir: str, output_text: str, end_token_idx:int, start_token_idx: int=0, num_layers: int=48):

    cmp = CompareHelper(res_lmdeploy_dir, res_xtuner_dir, end_token_idx=end_token_idx, start_token_idx=start_token_idx)

    header = ['layer_index', 'module', 'in/out', 'avg abs src', 'avg abs ref', 'max abs src', 'max abs ref', 'avg abs diff', 'max abs diff', 'avg rel diff', 'max rel diff']
    stats = []
    
    # cmp.compare_tensor('input_ids', rtol=1e-5, atol=1e-5)
    for layer_idx in range(num_layers):
        if layer_idx == 0:
            stats.append([layer_idx, 'layer', 'input'] + cmp.compare_tensor(f'layer{layer_idx}.input_hidden_states'))

        stats.append([layer_idx, 'attn', 'input'] + cmp.compare_tensor(f'layer{layer_idx}.attn.input_hidden_states'))
        stats.append([layer_idx, 'attn', 'output'] + cmp.compare_tensor(f'layer{layer_idx}.attn.output_hidden_states'))
        stats.append([layer_idx, 'moe', 'input'] + cmp.compare_tensor(f'layer{layer_idx}.mlp.input_hidden_states'))
        stats.append([layer_idx, 'moe', 'output'] + cmp.compare_tensor(f'layer{layer_idx}.mlp.output_hidden_states'))
        cmp.compare_tensor(f'layer{layer_idx}.mlp.gate.topk_ids')
        # cmp.compare_tensor(f'layer{layer_idx}.mlp.gate.topk_weights')
        cmp.compare_tensor(f'layer{layer_idx}.mlp.gate.logits')
        stats.append([layer_idx, 'layer', 'output'] + cmp.compare_tensor(f'layer{layer_idx}.output_hidden_states'))
    
    # stats.append([layer_idx, 'lm_head', 'output'] + cmp.compare_tensor(f'logits'))

    df = pd.DataFrame(stats, columns=header)
    df.to_csv(output_text, index=True)
    print(f'Saved to {output_text}')


if __name__ == '__main__':
    fire.Fire(main)


