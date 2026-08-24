# KV Cache Size Calculator 截图计算拆解

本文按 `/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/hha_code/imgs` 里的截图，结合本地
`/mnt/shared-storage-user/huanghaian/code/transformers` 源码，拆解
[https://kvcache.ai/tools/kv-cache-size-calculator/](https://kvcache.ai/tools/kv-cache-size-calculator/) 里 GLM-5.2 和 DeepSeek V4 Pro 的 KV cache 数字。

截图里的输入相同：

- `tokens per sequence = 1024`
- `sequences = 1`
- `KV precision = BF16 / FP16 = 2 bytes`
- `indexer precision = BF16 / FP16 = 2 bytes`
- `Include draft KV cache = true`

注意：网页显示两个单位：左边 `GiB` 是按 `1024^3` 换算，右边 `GB` 是按 `1000^3` 换算。

## GLM-5.2

截图参数来自 `zai-org/GLM-5.2/config.json`。2026-07-22 拉到的关键字段是：

- `num_hidden_layers = 78`
- `num_nextn_predict_layers = 1`
- `kv_lora_rank = 512`
- `qk_rope_head_dim = 64`
- `index_head_dim = 128`
- `indexer_types`: 21 个 `full`，57 个 `shared`

本地 Transformers 对应模型是 `glm_moe_dsa`：

- `configuration_glm_moe_dsa.py` 定义 `num_hidden_layers=78`、`kv_lora_rank=512`、`qk_rope_head_dim=64`、`index_head_dim=128` 等字段。
- `modeling_glm_moe_dsa.py` 的 `GlmMoeDsaAttention` 先做 `kv_a_proj_with_mqa(hidden_states)`，输出宽度是 `kv_lora_rank + qk_rope_head_dim`，随后才把 latent KV 展开成 `k_pass/value_states`。
- `GlmMoeDsaIndexer` 的 `wk(hidden_states)` 只产生 `[B, S, index_head_dim]` 的 indexer key，并通过 `past_key_values.update_indexer()` 追加。
- `config.indexer_types[layer_idx] == "shared"` 的层不运行自己的 indexer，而是复用前一个 full layer 的 `topk_indices`。

源码锚点：

- `src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py:94`
- `src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py:166`
- `src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py:347`
- `src/transformers/cache_utils.py:294`

### 网页怎么算

网页用的是生产态 latent KV cache：

```text
active_layers = main_layers + draft_layers = 78 + 1 = 79
active_indexer_layers = main_full_indexer_layers + draft_indexer_layers = 21 + 1 = 22

kv_elements_per_token = active_layers * (kv_lora_rank + qk_rope_head_dim) # 虽然是 sparse atten，但是 kv 肯定存的是全量的，head 是 1， dim 是 kv_lora_rank + qk_rope_head_dim
                      = 79 * (512 + 64)
                      = 45,504

indexer_elements_per_token = active_indexer_layers * index_head_dim # 没有 v cache
                           = 22 * 128
                           = 2,816

per_token_elements = 45,504 + 2,816 = 48,320
per_token_bytes = 48,320 * 2 = 96,640 bytes = 94.375 KiB
```

所以：

```text
KV cache size = 45,504 * 1024 * 2 bytes = 93,192,192 bytes = 88.875 MiB
Indexer cache size = 2,816 * 1024 * 2 bytes = 5,767,168 bytes = 5.5 MiB
Total = 98,959,360 bytes = 0.09216 GiB = 0.09896 GB
```

这正好对应截图里的：

- `KV cache size = 88.87500 MiB`
- `Indexer cache size = 5.50000 MiB`
- `Per token size = 94.37500 KiB`
- `Total cache size = 0.09216 GiB = 0.09896 GB`

### 和 Transformers 源码的关系

这里最容易误会的是：截图算的不是本地 Transformers 当前 eager 路径里 `past_key_values.update(key_states, value_states)` 的展开 K/V 实占。

`GlmMoeDsaAttention.forward()` 里实际传给 `past_key_values.update()` 的是已经展开后的：

- `key_states = cat(k_pass, k_rot)`，形状接近 `[B, num_attention_heads, S, qk_nope_head_dim + qk_rope_head_dim]`
- `value_states`，形状接近 `[B, num_attention_heads, S, v_head_dim]`

也就是说，Transformers 通用 cache 兼容路径会缓存 expanded K/V；网页下方也写了 “Expanded HF-compatible cache is not included”。网页的 GLM 数字来自 MLA 的 latent KV 设计，即只存 `kv_a_proj_with_mqa` 产生的 `kv_lora_rank + qk_rope_head_dim`，需要计算时再展开或由专用 kernel 消费。

因此 GLM-5.2 这张图可以理解为：按模型结构的生产态 cache 算法估算，而不是按当前 HF eager cache 张量逐字节统计。

## DeepSeek V4 Pro

截图参数来自 `deepseek-ai/DeepSeek-V4-Pro/config.json`。2026-07-22 拉到的关键字段是：

- `num_hidden_layers = 61`
- `num_nextn_predict_layers = 1`
- `head_dim = 512`
- `num_key_value_heads = 1`
- `sliding_window = 128`
- `index_head_dim = 128`
- `compress_ratios = [128, 128, 4, 128, ... , 4, 0]`

`compress_ratios` 统计后是：

- ratio=4: 30 层
- ratio=128: 31 层
- ratio=0: 1 层，也就是截图里勾选 draft 后额外包含的 draft KV cache

本地 Transformers 对应模型是 `deepseek_v4`：

- `configuration_deepseek_v4.py` 定义三种 attention layer：`sliding_attention`、`compressed_sparse_attention`、`heavily_compressed_attention`。ratio=4 映射到 CSA，ratio=128 映射到 HCA，ratio=0 映射到 sliding-only。
- `DeepseekV4Attention` 明确是 shared-KV MQA：`num_key_value_heads = 1`，`kv_proj` 只产生一个 `head_dim` 宽的 KV，attention 时同一个 `kv` 同时作为 key/value。
- `DeepseekV4HCACache` 在 sliding cache 外保存 `compressed_kv["compressor"]`，每 `compress_rate=128` 个 token 产生一个 compressed entry。
- `DeepseekV4CSACache` 继承 HCA cache，并额外保存 `compressed_kv["indexer"]`，每 `compress_rate=4` 个 token 产生一个 indexer compressed entry。
- `DeepseekV4CSACompressor` 会把 compressed KV 拼到 sliding KV 后面；`DeepseekV4Indexer` 对 compressed indexer KV 做 top-k。

源码锚点：

- `src/transformers/models/deepseek_v4/configuration_deepseek_v4.py:21`
- `src/transformers/models/deepseek_v4/configuration_deepseek_v4.py:246`
- `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:171`
- `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:362`
- `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:462`
- `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:589`
- `src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:755`

### 网页怎么算

DeepSeek V4 Pro 截图使用的也是生产态 cache：

```text
active_layers = main_layers + draft_layers = 61 + 1 = 62
sliding_window = 128
head_dim = 512

sliding_window_elements = active_layers * sliding_window * head_dim
                        = 62 * 128 * 512
                        = 4,063,232
```

这里没有乘 2 表示 K 和 V，因为 V4 源码里 K=V，是同一个 shared KV tensor。

长程 compressed KV 只来自 ratio>0 的主模型层：

```text
compressed_elements =
    ratio4_layers * floor(tokens / 4) * head_dim
  + ratio128_layers * floor(tokens / 128) * head_dim

= 30 * 256 * 512 + 31 * 8 * 512
= 3,932,160 + 126,976
= 4,059,136
```

所以 KV 总元素是：

```text
kv_elements = sliding_window_elements + compressed_elements
            = 4,063,232 + 4,059,136
            = 8,122,368

kv_bytes = 8,122,368 * 2 = 16,244,736 bytes = 15.49219 MiB
```

Indexer cache 只出现在 ratio=4 的 CSA 层：

```text
indexer_elements = ratio4_layers * floor(tokens / 4) * index_head_dim
                 = 30 * 256 * 128
                 = 983,040

indexer_bytes = 983,040 * 2 = 1,966,080 bytes = 1.875 MiB
```

总量：

```text
total_bytes = (8,122,368 + 983,040) * 2
            = 18,210,816 bytes
            = 0.01696 GiB
            = 0.01821 GB

per_token = 18,210,816 / 1024
          = 17,784 bytes
          = 17.36719 KiB
```

这对应截图里的：

- `KV cache size = 15.49219 MiB`
- `Indexer cache size = 1.87500 MiB`
- `Per token size = 17.36719 KiB`
- `Total cache size = 0.01696 GiB = 0.01821 GB`

### 和 Transformers 源码的关系

DeepSeek V4 Pro 的网页公式和本地 Transformers 自定义 cache 结构基本一致：

- `DeepseekV4Attention.forward()` 对每层先更新 sliding KV：`past_key_values.update(kv, kv, layer_idx)`。
- `DeepseekV4HCACache.update()` 注释写明 V4 用 shared-KV MQA，keys 和 values 指向同一份存储；代码也设置 `self.values = self.keys`。
- HCA compressor 每 128 token append 一个 `head_dim=512` 的 compressed KV entry。
- CSA compressor 每 4 token append一个 `head_dim=512` 的 compressed KV entry。
- CSA indexer 每 4 token append 一个 `index_head_dim=128` 的 compressed indexer KV entry。

网页没有把以下小状态算进去：

- CSA/HCA compressor 的 `buffer_kv`、`buffer_gate`
- CSA overlap 的 `overlap_kv`、`overlap_gate`
- `entry_count` 这类 Python/int 状态
- projection weights、position bias、attention mask、top-k indices 临时张量

另外，Transformers 的通用 `DynamicSlidingWindowLayer` 保存历史时是 `sliding_window - 1` 个 token；V4 自定义 cache 的 `update()` 也写成 `-self.sliding_window + 1`。网页用 `sliding_window` 直接估算，所以它是偏保守的整窗口估算，差异约为每层一个 token 的 KV。

## 总结

GLM-5.2 和 DeepSeek V4 Pro 这两张图的核心差别是：

- GLM-5.2：每层缓存 latent MLA KV，宽度是 `kv_lora_rank + qk_rope_head_dim = 576`；只有 full indexer 层另存 `index_head_dim = 128` 的 indexer key。shared indexer 层复用前一个 full 层的 top-k，不额外存 indexer key。
- DeepSeek V4 Pro：每层先有一个 shared-KV sliding window；ratio=4/128 层再按压缩率追加 long-range compressed KV；只有 ratio=4 的 CSA 层有 Lightning Indexer cache。

所以 GLM-5.2 在 1024 tokens 下是 `94.375 KiB/token`，DSv4 Pro 是 `17.36719 KiB/token`。DSv4 Pro 小很多，主要因为它把 full sequence history 变成了固定 sliding window 加低频 compressed entries；GLM-5.2 则仍然按每层每 token 保存 latent KV，只是没有保存展开后的多头 K/V。