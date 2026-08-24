from typing import List, Literal, Iterable
from cyclopts import App
from pathlib import Path
import json
from xtuner.v1.datasets import  CachableTokenizeFunction, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
import jsonlines

from transformers import AutoTokenizer


def parse_xpuyu_json_cfg(
    path,
    tokenize_fn_cfg,
    max_prompt_length,
    data_judger_mapping,
    ignore_multimodal_info=False,
):
    with open(path) as f:
        json_cfg = json.load(f)
    converted_cfg = []
    for ds_name, ds_cfg in json_cfg.items():
        annotation = ds_cfg["annotation"]
        if isinstance(annotation, str):
            annotation = [annotation]
        for ann in annotation:
            converted_cfg.append(
                {
                    "dataset": DatasetConfig(
                        name=ds_name,
                        anno_path=ann,
                        sample_ratio=ds_cfg["sample_ratio"],
                        media_root=ds_cfg.get("media_root", None),
                        class_name="VLMJsonlDataset",
                    ),
                    "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                        processor_path=tokenizer_path,
                        min_pixels=None,
                        # max_pixels=None,
                        # max_pixels=2097152, # ----------------------------------------------------------------
                        video_min_total_pixels=None,
                        video_max_total_pixels=None,
                        video_min_frames=None,
                        video_max_frames=None,
                        fps=None,
                        rand_video_max_frames=24,
                        add_vision_id=True,
                        system_message=None,
                        hash=None,
                        enable_3d_rope=False,
                        oss_loader_cfg=None,
                        debug=True,
                        oss_time_log_thr=10,
                        max_length=max_prompt_length,
                        # data_judger_mapping=data_judger_mapping,
                        ignore_multimodal_info=ignore_multimodal_info,
                        chat_template="qwen3.5-vl",
                        add_generation_prompt=True,
                        enable_thinking=True,
                    ),
                }
            )
    return converted_cfg


def show_iterable(data: Iterable[dict], tokenize_fn: CachableTokenizeFunction, tokenizer, media_root):
    sep = "=" * 80
    color_prefix = "\033[31m"
    color_suffix = "\033[0m"
    current_string = ""

    current_type: Literal["positive", "negative"]
    token_type: Literal["positive", "negative"]

    def flush_tokens(current_tokens: List[int]) -> str:
        if not current_tokens:
            return ""
        text = tokenizer.decode(current_tokens, skip_special_tokens=False)
        if current_type == "positive":
            return f"{color_prefix}{text}{color_suffix}"
        return text

    count = 0
    for messages in data:

        if count > 10:
            break
        count += 1

        res = tokenize_fn(messages, media_root=media_root)
        token_ids= res.prompt_ids
        num_tokens = len(token_ids)

        decode_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(f"-" * 80, '\n')
        print(f"num_tokens: {num_tokens}, decode_text: {decode_text}")


def main(tokenizer_path: Path, data_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    data_judger_mapping = dict(
            nemotron_tau_pivot={"nemo_single_step_tool_use": 1.0},
    )
    train_dataset_cfg = parse_xpuyu_json_cfg(
        data_path, None, 8192, data_judger_mapping
    )
    with open(data_path) as f:
        json_cfg = json.load(f)

    data_path = json_cfg["nemotron_tau_pivot"]["annotation"]
    media_root = json_cfg["nemotron_tau_pivot"]["media_root"]
    tokenize_fn = train_dataset_cfg[0]["tokenize_fn"].build(tokenizer, anno_name="nemotron_tau_pivot")

    with jsonlines.open(data_path) as reader:
        show_iterable(reader, tokenize_fn, tokenizer, media_root)
    

if __name__ == "__main__":
    tokenizer_path = "/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1_share_layer4/20260411031507/hf-5089"
    data_path = "/mnt/shared-storage-user/huanghaian/code/gitlab/xtuner/workspace/demo_data_new.json"
    main(tokenizer_path, data_path)

