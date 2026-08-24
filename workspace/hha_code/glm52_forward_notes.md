# GLM-5.2 Forward 过程学习笔记

本文按“先概念、再训练、再推理、最后 MTP”的顺序整理 GLM-5.2。材料来源有三类：

- 官方博客：`https://z.ai/blog/glm-5.2`
- 本地 Transformers 源码：`/mnt/shared-storage-user/huanghaian/code/transformers/src/transformers/models/glm_moe_dsa/`
- 本地 SGLang 推理源码：`/mnt/shared-storage-user/huanghaian/code/slime_package/xtuner_sglang/sglang`

需要先明确边界：

- Transformers 本地实现覆盖 GLM-5/5.1/5.2 的主干语言模型 `GlmMoeDsaForCausalLM`。
- Transformers 文档明确说不包含 MTP layer，所以训练/普通推理 forward 可以从 Transformers 源码读，MTP speculative decoding 需要结合博客和 SGLang。
- SGLang 中 GLM-5.2 DSA MTP 测试走的是 `EAGLE/NextN` 路径，不是 `FROZEN_KV_MTP` 路径。

## 0. 阅读路线

建议按下面顺序读：

1. 先读第 1 节，建立 DSA、indexer、top-k、KV cache 的基本词汇。
2. 再读第 2 节，理解博客到底在强调什么。
3. 然后读第 3、4 节，从 Transformers 源码理解训练 forward。
4. 再读第 5 节，看主模型推理里的 prefill/decode/KV cache。
5. 最后读第 6 节，专门理解 MTP、IndexShare、KVShare，以及 SGLang 真实实现。

## 1. 前置概念

### 1.1 三套容易混的东西

GLM-5.2 里讨论 cache/index 时，至少有三套对象：

```text
主 attention KV cache(flash mla):
  真正 attention 聚合 value 时读取的 K/V。

indexer K cache(indexer):
  DSA indexer 做检索/top-k 时使用的历史 key。

top-k indices(输入给 flash mla):
  indexer 选出来的位置列表，例如 2048 个历史 token index。
```

注意：在这个 DSA indexer 语境下，核心只需要 K cache，没有 V cache。因是 indexer 不做 attention value 聚合，它只做“检索位置”

```text
  当前 hidden -> q_idx

  当前 hidden -> k_idx

  q_idx @ 历史 k_idx

    -> scores

    -> top-k positions

  输出是：

  topk_indices

  不是：

  attention output

  所以它不需要像普通 attention 那样存 v cache。
```

这三者不是一回事。

主 attention KV cache 存真正被 attention 读取的内容：

```text
K_main[position], V_main[position]
```

indexer K cache 是搜索索引：

```text
K_indexer[position]
```

top-k indices 是搜索结果：

```text
[pos_1, pos_2, ..., pos_2048]
```

可以用一句话概括：

```text
indexer K cache = 搜索索引
top-k indices   = 搜索结果里的位置
主 KV cache      = 真正被读取的内容
```

### 1.2 indexer 不是一个完整小 attention

indexer 更像一个轻量检索器，不是完整 attention 层。

它有：

```text
indexer query: q_idx
indexer key:   k_idx
```

decode 时：

```text
当前 token hidden
  -> 生成 q_idx
  -> 生成 k_idx
  -> 把 k_idx 写入 indexer K cache

q_idx 和历史 indexer K cache 打分
  -> top-k token positions
```

通常不需要 indexer V cache，因为 indexer 不负责 value 聚合。它只输出 `topk_indices`，真正聚合 hidden/value 的仍是主 attention：

```text
主 attention q
  + topk_indices 指向的主 KV cache
  -> sparse attention output
```

所以随着序列变长，增长的是：

```text
主 KV cache 长度
indexer K cache 长度
```

不增长的是：

```text
indexer 模块/权重数量
```

每次选出来的 `topk_indices` 数量通常固定为 `index_topk`，GLM-5.2 config 中是 2048。

### 1.3 两种 IndexShare

本文里有两种 IndexShare，名字相似但层级不同。

第一种是主干 DSA 层间 IndexShare：

```text
full indexer layer:
  运行 indexer，产生 top-k indices

shared indexer layer:
  不运行自己的 indexer，复用前一层 top-k indices
```

GLM-5.2 主干大致是每 4 个 sparse attention layer 一组：

```text
full -> shared -> shared -> shared
```

第二种是 MTP iteration IndexShare：

```text
draft step0 产生或拿到一份 top-k indices
draft step1 复用它
draft step2 复用它
...
```

SGLang 里的 `index_share_for_mtp_iteration` 控制的是第二种，也就是 MTP 多步 draft 之间复用 top-k indices。

### 1.4 decode 时 q 和 KV 长度为什么不同

普通 decode 每次只输入一个新 token：

```text
q length = 1
KV cache length = past_len + 1
```

这不是异常，而是自回归 decode 的常态。当前 query 只来自新 token，但它要 attend 到历史所有可见 token。

DSA decode 也是类似：

```text
当前 q_idx 长度 = 1
历史 indexer K cache 长度 = past_len + 1
indexer(q_idx, K_indexer_cache) -> top-k positions

当前 attention q 长度 = 1
主 KV cache 长度 = past_len + 1
attention 只读取 top-k positions 指向的主 K/V
```

`FROZEN_KV_MTP` 路径里会出现更特殊的“draft 当前有新 q，但 KV 仍只读 target prefix”的情况；这不是当前 SGLang GLM-5.2 EAGLE/NextN 路径的默认逻辑。

## 2. 博客核心内容

GLM-5.2 博客主线不是单纯扩大参数，而是为了长上下文、长轨迹 coding agent 和高效 serving 做系统优化：

1. 支持稳定 1M token context。
2. 强化长程 coding agent 能力。
3. 用 IndexShare 优化 DSA，降低长上下文下 indexer dot product 和 top-k 重复计算。
4. 改进 MTP layer，提高 speculative decoding 的 acceptance length。
5. serving 瓶颈转向 KV-cache capacity、长上下文 kernel、CPU 侧调度和 cache 管理。

博客里最关键的 DSA 变化是 IndexShare：

```text
不是每层 sparse attention 都跑 indexer
而是多个 sparse attention layer 共享一次 indexer top-k 结果
```

博客里 MTP 的关键变化是：

```text
MTP 后续 step 复用第一步的 top-k indices
MTP 后续 step 复用第一步的 shared KV cache  # sglang 默认不是这么实现的，需要参考后续解读来理解
减少训练和推理时 MTP rollout 的分布差异
```

博客 ablation 给出的 acceptance length：

```text
Baseline                         4.56
+ IndexShare + KVShare           5.10
+ Rejection Sampling             5.29
+ End-to-end TV Loss             5.47
```

注意：博客里的 KVShare 是系统/原始实现层面的机制。Transformers 本地源码没有 MTP layer；SGLang 中也要区分 EAGLE/NextN 和 Frozen-KV 两条路径。

## 3. Checkpoint 和源码入口

### 3.1 GLM-5.2 checkpoint 主配置

不要把 `GlmMoeDsaConfig` 类默认值直接当成 GLM-5.2 checkpoint。公开 GLM-5.2 config 的主干配置大致是：

```text
architecture = GlmMoeDsaForCausalLM
num_hidden_layers = 78
hidden_size = 6144
num_attention_heads = 64
num_key_value_heads = 64

q_lora_rank = 2048
kv_lora_rank = 512
qk_nope_head_dim = 192
qk_rope_head_dim = 64
qk_head_dim = 256
v_head_dim = 256

n_routed_experts = 256
num_experts_per_tok = 8
n_shared_experts = 1
first_k_dense_replace = 3

index_topk = 2048
index_head_dim = 128
index_n_heads = 32
index_topk_freq = 4
index_skip_topk_offset = 3

max_position_embeddings = 1048576
rope_theta = 8000000
index_share_for_mtp_iteration = true
```

### 3.2 Transformers 源码结构

主要文件：

```text
configuration_glm_moe_dsa.py
modeling_glm_moe_dsa.py
cache_utils.py
```

主类关系：

```text
GlmMoeDsaForCausalLM
  ├── model: GlmMoeDsaModel
  │   ├── embed_tokens
  │   ├── layers[0..77]: GlmMoeDsaDecoderLayer
  │   │   ├── input_layernorm
  │   │   ├── self_attn: GlmMoeDsaAttention
  │   │   │   ├── MLA-style Q/KV projections
  │   │   │   └── optional GlmMoeDsaIndexer
  │   │   ├── post_attention_layernorm
  │   │   └── mlp: dense MLP or sparse MoE
  │   ├── norm
  │   └── rotary_emb
  └── lm_head
```

`GlmMoeDsaForCausalLM.forward` 做两件事：

```text
self.model(...) -> last_hidden_state
lm_head(last_hidden_state) -> logits
```

如果传入 `labels`，再计算 causal LM loss。

### 3.3 indexer_types

GLM-5.2 主干的 `indexer_types` 大致是：

```text
full, full, 
full, shared, shared, shared,
full, shared, shared, shared,
full, shared, shared, shared,
...
```

含义是：

- `full` 层有自己的 indexer，负责产生 top-k indices。
- `shared` 层没有自己的完整 indexer，复用前面层的 top-k indices。

前几层和后续周期稍有差异，核心模式是从浅层之后进入每 4 层共享一次 indexer 的节奏。

## 4. 训练视角：Transformers Forward

训练时典型调用：

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    use_cache=False,
)
```

训练通常不使用 KV cache，而是一次性处理完整序列。

### 4.1 输入和 embedding

输入一般是：

```text
input_ids: [batch, seq_len]
attention_mask: [batch, seq_len]
labels: [batch, seq_len]
```

模型先做 token embedding：

```text
input_ids -> embed_tokens -> hidden_states
hidden_states: [batch, seq_len, hidden_size]
```

GLM-5.2 hidden size 是 6144。

### 4.2 position 和 RoPE

源码会根据 attention mask 和 cache position 生成 position 信息。训练时没有 past cache，位置通常从 0 到 `seq_len - 1`。

RoPE 不在每层重复构造，而是在 `GlmMoeDsaModel` 外层先算好 position embedding，然后传给每个 decoder layer。

每层 attention 只负责把 RoPE 应用到自己的 q/k rope 部分。

### 4.3 Decoder layer 顺序

每个 decoder layer 大致是 pre-norm Transformer block：

```text
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states
```

不同层的 MLP 类型不同：

- 前 `first_k_dense_replace = 3` 层通常是 dense MLP。
- 后续层是 sparse MoE。

### 4.4 MLA 风格 Q/KV 构造

GLM-5.2 attention 不是朴素 `hidden -> q,k,v` 三个大投影，而是 MLA 风格的低秩结构。

query 路径大致是：

```text
hidden
  -> q_a_proj
  -> q_a_layernorm
  -> q_b_proj
  -> q split:
       q_nope: 不使用 RoPE 的部分
       q_rope: 使用 RoPE 的部分
```

KV 路径大致是：

```text
hidden
  -> kv_a_proj_with_mqa
  -> split:
       compressed_kv
       k_rope
  -> kv_a_layernorm(compressed_kv)
  -> kv_b_proj
  -> split:
       k_nope
       value
```

RoPE 只作用于：

```text
q_rope
k_rope
```

attention 计算时会把 nope 部分和 rope 部分组合起来。

从结构上看，GLM-5.2 的 attention block 可以理解成两部分：

```text
DSA indexer:
  负责选位置。
  q_idx @ indexer K cache -> top-k indices

MLA attention:
  负责真正 attention 聚合。
  用当前 q 和 top-k indices 指向的主 KV cache 做 sparse attention。
```

如果这一层是 `full indexer layer`：

```text
运行 indexer
得到 top-k indices
再运行 MLA attention
```

如果这一层是 `shared indexer layer`：

```text
不运行自己的 indexer
复用上一层传来的 top-k indices
仍然运行 MLA attention
```

所以 shared 层不是跳过 attention。它只是跳过 DSA indexer 检索/top-k 部分，仍然会执行自己的 Q/KV 投影、RoPE、MLA attention、输出投影，也仍然会维护这一层自己的主 KV cache。

### 4.5 DSA indexer 在训练中做什么

DSA 的核心是：不让每个 query attend 所有历史 token，而是先用 indexer 选 top-k。

训练中序列长度是 `S`，每个 query 位置都需要一个候选集合：

```text
hidden_states
  -> indexer q/k
  -> top-k historical positions for each query
  -> sparse attention mask / sparse attention layout
```

full indexer 层会运行 indexer：

```text
topk_indices = indexer(hidden_states, ...)
```

shared indexer 层会复用上一层传下来的：

```text
topk_indices = prev_topk_indices
```

所以训练 forward 中有一个很重要的层间变量：

```text
topk_indices
```

它在 full 层被刷新，在 shared 层被复用。

### 4.6 主 attention 如何使用 top-k

DSA attention 的输入仍然有当前层的 q/k/v，但 attention 聚合不看全量历史，而是看 `topk_indices` 指定的位置。

可以抽象成：

```text
for each query position i:
  selected_positions = topk_indices[i]
  output_i = attention(q_i, K[selected_positions], V[selected_positions])
```

这就是 sparse attention。`index_topk = 2048` 表示每个 query 最多选择约 2048 个位置。

### 4.7 MoE 层

MoE 层包含 routed experts 和 shared expert。

大致流程：

```text
hidden
  -> router/gate logits
  -> top num_experts_per_tok experts
  -> routed experts
  -> shared expert
  -> combine
```

GLM-5.2 config 中：

```text
n_routed_experts = 256
num_experts_per_tok = 8
n_shared_experts = 1
```

所以每个 token 会路由到 8 个 routed experts，并叠加 shared expert 输出。

### 4.8 LM head 和 loss

最后：

```text
hidden_states = final_norm(hidden_states)
logits = lm_head(hidden_states)
```

如果有 labels，则做 causal LM shift loss：

```text
logits[:, :-1] 预测 labels[:, 1:]
```

## 5. 推理视角：Prefill、Decode 和 Cache

推理分两类 forward：

```text
prefill:
  一次处理 prompt 的多个 token
  建立主 KV cache 和 indexer K cache

decode:
  每次处理一个新 token
  append 主 KV cache 和 indexer K cache
```

### 5.1 DSA cache layer

Transformers 针对 `"deepseek_sparse_attention"` 使用带 indexer 的 cache layer。

普通 attention cache 只需要：

```text
key cache
value cache
```

DSA 还需要：

```text
indexer key cache
```

所以 DSA cache 可以理解成：

```text
main K cache
main V cache
indexer K cache
```

`StaticIndexedLayer` 则是静态预分配版本，更适合 `torch.compile` 或 CUDA graph。

### 5.2 Prefill

prefill 输入 prompt：

```text
input_ids: [B, P]
use_cache=True
past_key_values=None
```

每层做：

```text
生成 P 个 query
生成 P 个 main K/V
写入 main KV cache

full indexer 层:
  生成 P 个 indexer K
  写入 indexer K cache
  对 P 个 query 计算 top-k

shared indexer 层:
  复用上一层 top-k
```

prefill 后 cache 长度是 `P`。

### 5.3 Decode

decode 输入一个新 token：

```text
input_ids: [B, 1]
past_key_values: 已有长度 P 或更长
use_cache=True
```

每层做：

```text
生成当前 token 的 query
生成当前 token 的 main K/V
append 到 main KV cache

full indexer 层:
  生成当前 token 的 indexer K
  append 到 indexer K cache
  当前 q_idx 对完整 indexer K cache 做 top-k

shared indexer 层:
  复用上一层 top-k

主 attention:
  当前 q 只 attend top-k positions 指向的 main K/V
```

所以 decode 单步：

```text
q length = 1
main KV cache length = past_len + 1
indexer K cache length = past_len + 1
top-k indices length = index_topk
```

### 5.4 IndexShare 对 decode 的作用

如果没有 IndexShare，每层都要：

```text
q_idx @ K_indexer_cache
top-k
```

GLM-5.2 主干使用层间 IndexShare 后：

```text
full 层:
  计算 top-k

后面 shared 层:
  复用 top-k
```

这减少的是 indexer 检索和 top-k 排序的重复计算，不是直接减少所有主 KV cache。

### 5.5 KV cache 为什么仍是瓶颈

GLM-5.2 支持 1M context 后，主 KV cache 的容量压力非常大。

粗略看每层每 token 主 KV 元素量：

```text
key:   num_heads * head_dim
value: num_heads * v_head_dim
```

GLM-5.2 是：

```text
64 * 256 * 2 = 32768 elements/token/layer
```

bf16 下约 64 KiB/token/layer。78 层就是数 MiB/token/batch 的量级。真实引擎会有 TP/并行/压缩/分页等优化，但长上下文 serving 的大头仍然是 cache。

indexer K cache 也随长度增长，但远小于主 KV cache：

```text
indexer K: [B, L, index_head_dim]
```

它是搜索索引，不是完整 K/V。

### 5.6 RL replay 路由信息的数据量估算

如果 RL 训练或 replay 时想把 routed expert 选择和 indexer top-k 都存下来，数据量会很大。这里先不考虑 MTP 层，只估算 GLM-5.2 主干，序列长度按 32K token，存储单位按 `uint16 = 2 bytes`。

先看 routed expert。

GLM-5.2 主干配置：

```text
num_hidden_layers = 78
first_k_dense_replace = 3
MoE layers = 78 - 3 = 75
num_experts_per_tok = 8
```

如果只存 expert id：

```text
tokens * moe_layers * experts_per_token * bytes_per_id
= 32768 * 75 * 8 * 2
= 39,321,600 bytes
= 37.5 MiB
```

如果除了 expert id，还要存每个 expert 的 routing/gate weight，并且也按 `uint16` 存一份，那么 routed expert 部分约翻倍：

```text
expert id + expert weight
≈ 75 MiB
```

再看 DSA indexer top-k。

GLM-5.2 配置：

```text
index_topk = 2048
full indexer layers ≈ 21
```

这里用 21 是因为主干不是 78 层都跑 full indexer；IndexShare 后大致每 4 层共享一个 indexer，加上前几层 full indexer，full indexer 层数量约为 21。

如果每个 token、每个 full indexer 层都存 `index_topk` 个位置，并且每个位置用 `uint16` 存：

```text
tokens * full_indexer_layers * index_topk * bytes_per_index
= 32768 * 21 * 2048 * 2
= 2,818,572,288 bytes
= 2.625 GiB
≈ 2.82 GB
```

所以总量级是：

```text
只存 expert id + indexer top-k:
  37.5 MiB + 2.625 GiB
  ≈ 2.66 GiB / sample

如果 routed expert 还存 gate weight:
  约 2.70 GiB / sample
```

结论是：大头不是 MoE routed expert，而是 DSA indexer top-k。原因很直接：

```text
routed expert: top8
indexer:       top2048
```

## 6. MTP、IndexShare、KVShare

这一节专门讲博客和 SGLang。Transformers 本地源码没有 MTP layer。

### 6.1 MTP 要解决什么

Speculative decoding 想让便宜的 draft model 先猜多个未来 token，再让 target model 一次验证。

MTP layer 可以理解为 target model 后面挂的轻量 draft module：

```text
target model 已经算到 h4
MTP 根据 h1..h4 猜 h5 / token5
再根据已有信息继续猜 h6 / token6
```

博客图中：

- `e_i` 是 token embedding。
- 蓝色 `h_i` 是 target model hidden。
- 橙色/紫色 `h_i` 是 MTP step 产生的 hidden。

### 6.2 先防止一个误读：KVShare 不是默认路径

博客里说的：

```text
MTP 后续 step 复用第一步的 top-k indices
MTP 后续 step 复用第一步的 shared KV cache
```

这句话不能直接理解成“当前 SGLang GLM-5.2 默认 EAGLE/NextN 路径不写 draft KV”。

更准确地分三层：

```text
博客机制:
  IndexShare + KVShare
  目标是让后续 MTP step 复用第一步 top-k 和 shared KV，
  减少训练/推理 rollout 的分布差异。

当前 SGLang GLM-5.2 EAGLE/NextN:
  明确实现的是 MTP step 之间复用 DSA top-k indices。
  draft worker 仍然有自己的 KV pool。
  draft forward 仍会设置 out_cache_loc 并写 draft KV。

SGLang FROZEN_KV_MTP:
  这条路径才明确表示 draft 不写自己的 KV，
  而是只读 target KV / shared KV。
```

所以后文提到 KVShare 时，要区分“博客设计意图”和“SGLang 当前 GLM-5.2 默认路径”。如果说严格的 KV cache 共享，即不把 MTP hidden 产生的 KV append 到 draft KV cache，那对应的是 `FROZEN_KV_MTP` worker，而不是默认 GLM-5.2 EAGLE/NextN。

### 6.3 训练和推理为什么不一致

训练时有完整真实序列，可以 teacher forcing：

```text
target model:
  t1,t2,t3,t4,t5,...
  -> h1,h2,h3,h4,h5,...

MTP step2 training:
  可以使用 h1,h2,h3,h4,h5
  这些都来自 target model
```

推理时 target model 只真正算到了当前位置，例如 `h4`：

```text
已有 target hidden:
  h1,h2,h3,h4

MTP step1:
  生成 h5_mtp

MTP step2:
  只能用 h1,h2,h3,h4,h5_mtp
```

于是：

```text
training step2:
  h1,h2,h3,h4,h5 = target,target,target,target,target

inference step2:
  h1,h2,h3,h4,h5 = target,target,target,target,MTP
```

这就是 MTP 的 training-inference discrepancy。

### 6.4 用 prefill/decode 术语理解 MTP step

先有 target/main model prefill 或 verify：

```text
target/main model:
  输入真实 context
  输出 target hidden h1..h4
  构建 target 自己的 KV cache
```

然后 MTP step1 可以理解成 draft prefill：

```text
MTP step1 = draft prefill:
  输入:
    target hidden h1..h4
    shifted token embeddings e2..e5

  构建:
    draft/MTP attention 自己的 KV cache

  输出:
    h5_mtp
    第一个 draft token
```

后续 MTP step 可以理解成 draft decode：

```text
MTP step2 = draft decode:
  输入:
    上一步 hidden h5_mtp
    draft token embedding

  输出:
    h6_mtp
    第二个 draft token
```

如果不是 frozen-kv 路径，draft decode 会像普通 decode 一样维护 draft KV cache。

### 6.5 博客里的 IndexShare 和 KVShare

博客设计意图：

```text
MTP 后续 step 不重新计算 top-k indices
MTP 后续 step 复用第一步由 target hidden 构成的 shared KV cache
```

抽象地看：

```text
GLM-5.1 style:
  step2 attention cache = kv1..kv4 target + kv5_mtp

GLM-5.2 blog style:
  step2 attention cache = kv1..kv4 target
  top-k indices 也复用 step1
```

这样做的目标是让训练和推理后续 MTP step 尽量看到同一种信息集合：

```text
第一步 target KV
第一步 top-k indices
```

注意：这是博客机制的抽象解释。具体推理引擎是否完全“不写 draft KV”，要看实现路径。

博客还给了两个训练侧信息：

```text
1. MTP step 数:
   ablation 中 training 和 inference 都设置为 7 个 MTP steps。

2. MTP 参数:
   和 GLM-5.1 一样，不同 MTP steps 的参数共享。
```

所以按博客语义，不是训练 7 套不同 MTP layer，而是更像：

```text
同一个 MTP module / 同一套 MTP 参数
在 step1, step2, ..., step7 上反复 unroll 使用
每个 step 预测不同 future offset 的 token
```

这和 recurrent unroll 很像：step 维度增加了，但参数没有跟着复制成多套。

如果要严格对齐博客里的 KVShare 训练，则训练侧也要复用第一步 MTP 的 KV cache 和 top-k indices。博客原意是：

```text
training:
  后续 MTP step 复用 first MTP step 的 KV cache
  后续 MTP step 复用 first MTP step 的 top-k indices

inference:
  后续 MTP step 也按同样约束工作
```

原因是如果训练时不复用 KV，而是让 step2 使用 teacher-forced target `h5_target` 形成的 `kv5_target`，那训练/推理差异仍然存在：

```text
训练 step2:
  看到 target KV，包括 kv5_target

推理 step2:
  如果普通 draft decode，会看到 MTP 产生的 kv5_mtp
```

博客里的 KVShare 正是想避免这个差异，让训练和推理后续 step 都尽量只基于第一步那份 target-prefix KV/top-k。

但是要再次区分实现边界：

```text
如果目标是对齐博客机制:
  训练确实应该复用 first MTP step 的 KV cache 和 top-k indices。

如果目标是对齐当前 SGLang GLM-5.2 默认 EAGLE/NextN 路径:
  目前代码里明确看到的是 top-k indices 复用；
  draft KV 仍按 out_cache_loc 写入。
```

### 6.6 Rejection Sampling 是怎么回事

这里的 Rejection Sampling 指 speculative decoding 的验证/采样规则，不是训练 loss。

先把它和普通 verify 区分开：

```text
不开 rejection sampling:
  draft 只提交候选 token/tree
  verify 主要检查 target 在相同位置给出的 token 是否能沿着 draft tree 走下去
  不需要保存 draft 的完整 vocab 分布 q(v)

开启 rejection sampling:
  draft 不只提交 token，还要提交每一步完整的 draft proposal distribution q(v)
  verify 用 target 分布 p(v) 和 draft 分布 q(v) 做 accept/reject
  如果拒绝，还要从 residual distribution 中补采样，保证最终分布仍等价于 target
```

所以 RS 的关键不是“多了一次 target verify”。不开 RS 也一定要跑 target verify；差别在于 verify 后的验收规则，以及 draft 侧是否要把完整 `draft_probs` 带到 verify kernel。

#### 6.6.1 原理

有两个分布：

```text
q(v): draft/MTP 在当前位置给出的 token 分布
p(v): target/main model 在同一位置给出的 token 分布
```

普通贪心式 draft 更像：

```text
draft 选一个 token
target 如果也最喜欢/接受它，就通过
否则拒绝
```

这种做法在高熵场景很脆弱。RL rollout 中 target policy 熵升高后，概率质量会分散到更多 token 上，即使 draft 和 target 分布整体很接近，单个 argmax 也可能不稳定。

Rejection Sampling 的思路是让 draft 从自己的分布里采样：

```text
x ~ q
```

然后 target 按概率比接受：

```text
accept_prob(x) = min(1, p(x) / q(x))
```

如果接受，就保留 draft token `x`。如果拒绝，则从 residual distribution 中重新采样，让最终输出分布仍然等价于 target 分布。抽象地说：

```text
accepted mass = sum_v min(p(v), q(v))
```

所以单步 acceptance rate 是：

```text
alpha = sum_v min(p(v), q(v))
      = 1 - TV(p, q)
```

这里的 TV 是 total variation distance：

```text
TV(p, q) = 1/2 * sum_v |p(v) - q(v)|
         = 1 - sum_v min(p(v), q(v))
```

这解释了为什么 rejection sampling 会提高 acceptance length：它不是要求 draft 精确猜中 target 的单个最优 token，而是利用两个完整分布的重叠面积。只要 `p` 和 `q` 的分布重叠大，acceptance 就高。

多步 MTP 时，draft 会给出多个未来 token。target 一次 verify 后，从前往后逐个按 rejection sampling 规则验证：

```text
step1 accepted -> 才有机会验证 step2
step2 accepted -> 才有机会验证 step3
...
```

如果第 `i` 步单步接受率是 `alpha_i`，那么期望接受长度可以写成：

```text
E[L] = sum_{j=1..gamma} prod_{i=1..j} alpha_i
```

这就是为什么前面的 draft step 特别重要：step1 被拒绝，后面 step2..stepN 都没有机会被接受。

#### 6.6.2 SGLang 里开/不开 RS 的 verify 差异

SGLang 的开关是：

```text
server_args.speculative_use_rejection_sampling
```

相关源码位置：

```text
sglang/python/sglang/srt/speculative/eagle_worker_v2.py
sglang/python/sglang/srt/speculative/eagle_utils.py
sglang/python/sglang/srt/speculative/eagle_info.py
```

`EagleVerifyInput.draft_probs` 的注释已经把语义写得很清楚：

```text
draft_probs:
  每个 draft step 的 q(v)，shape = (bs, num_steps, vocab)
  只在 rejection sampling 下设置
  verify kernel 会消费它
```

##### 不开 RS

draft 侧：

```text
if topk == 1:
  topk_index = argmax(draft_logits)
  topk_p = 1
  draft_probs = None
else:
  probs = renorm_draft_probs(draft_logits)
  topk_p, topk_index = fast_topk(probs, topk)
  draft_probs = None
```

也就是说，不开 RS 时 draft 只把候选 token/tree 交给 verify；即使 `topk > 1` 中间算过 `probs` 用于取 top-k，也不会把完整 vocab 分布作为 `draft_probs` 保存给 verify。

verify 侧：

```text
target forward:
  对 draft tree 上的 token 位置跑 target/main model
  得到 target logits/probs

eagle_sample:
  greedy 请求:
    target_predict = argmax(target_logits)
    verify_tree_greedy_func(...)

  非 greedy 但不开 RS:
    draft_probs = zeros_like(target_probs)
    sampling_fn = tree_speculative_sampling_target_only
```

这里的核心是 `target-only`：验证只依赖 target 侧分布/采样结果与 draft tree 的候选结构，不使用 draft 的完整 `q(v)` 做 `p/q` 概率比。对于 `topk=1` 的链式 MTP，直观理解就是：

```text
draft 给出 t1, t2, ..., tk
target 在这些位置也生成/采样自己的 next token
从前往后找最长可接受前缀
accept_lens = 接受的 draft token 数 + bonus token
```

所以普通 speculative verify 不是“无损 RS 验证”，更像“target 重新走一遍这些候选位置，然后按 tree-match/target-only 规则决定能吃掉几个 draft token”。

##### 开启 RS

draft 侧会变成：

```text
probs = renorm_draft_probs(draft_logits)
topk_p, topk_index = fast_sample(probs, num_samples=1)
draft_probs_list.append(probs)
```

注意这里有两个变化：

```text
1. draft token 来自 sample，而不是 argmax
2. 每一步完整 q(v) 都会被保存到 draft_probs_list
```

verify 侧：

```text
target_probs = softmax(target_logits / temperature)
target_probs = top_k_renorm_prob(target_probs)
target_probs = top_p_renorm_prob(target_probs)

draft_probs = verify_input.draft_probs
sampling_fn = chain_speculative_sampling_triton
```

如果开了 RS 但 `draft_probs` 不存在，SGLang 会直接报错：

```text
Rejection sampling requires a target-vocab draft proposal distribution
```

这是因为 RS kernel 必须同时看到：

```text
p(v): target_probs
q(v): draft_probs
```

然后对每个 draft token 做：

```text
accept_prob(x) = min(1, p(x) / q(x))
```

拒绝时再从 residual distribution 中采样一个 token。这样最终输出分布可以保持 target 分布，而不是简单退化成“target 和 draft token 是否相等”的判断。

##### 小结

```text
不开 RS:
  draft 传 token/tree
  verify 不需要 draft full-vocab q(v)
  使用 greedy tree verify 或 target-only tree speculative sampling

开启 RS:
  draft 传 token/tree + 每步 full-vocab q(v)
  verify 使用 target p(v) 和 draft q(v)
  按 p/q 做 accept/reject，拒绝后 residual sampling
```

这也解释了为什么博客 ablation 里 `+ Rejection Sampling` 可以在 `IndexShare + KVShare` 之后继续提升 acceptance length：它改的不是 MTP forward 的 hidden/KV/index 结构，而是 verify 阶段“如何接受 draft token”的统计规则。

### 6.7 End-to-end TV Loss 是怎么回事

TV loss 是训练 MTP/draft 分布 `q` 的目标。它和 rejection sampling 配套，因为 rejection sampling 的 acceptance rate 正好由 `1 - TV(p, q)` 决定。

传统 MTP 训练可能用：

```text
CE loss:
  让 draft 预测真实 token

KL loss:
  让 draft 分布 q 接近 target 分布 p
```

但 GLM-5.2 博客引用的 Bebop 思路认为，CE/KL 没有直接优化 speculative rejection sampling 最关心的量。Rejection sampling 真正关心的是：

```text
sum_v min(p(v), q(v))
```

也就是两个分布的重叠面积。

单步 TV loss 可以写成：

```text
L_TV = TV(p, q)
     = 1 - sum_v min(p(v), q(v))
```

训练时：

```text
target/main model 输出 p(v)
MTP/draft 输出 q(v)
把 p 当 teacher distribution
对 MTP 参数反传，最小化 TV(p, q)
```

如果只是每个 step 独立最小化 TV，可以写成：

```text
sum_i TV(p_i, q_i)
```

但博客里说的是 end-to-end TV loss。它更贴近多步 speculative decoding 的真实收益，也就是直接优化期望接受长度：

```text
alpha_i = 1 - TV(p_i, q_i)

E[L] = sum_{j=1..gamma} prod_{i=1..j} alpha_i

L_e2e = 1 - E[L] / gamma
```

这个目标和多步验证过程对齐：

```text
step1 的质量会影响所有后续 token 是否有机会被接受
step2 的质量会影响 step2 以及后续
越靠前的 step 权重越隐式更大
```

所以 “end-to-end” 的含义不是简单多加几个 CE，而是把多步 MTP rollout 的接受链路作为整体目标优化。

实现上需要同时拿到：

```text
p_i: target model 对第 i 个 draft 位置的 full vocab 分布
q_i: MTP draft 对第 i 个 draft 位置的 full vocab 分布
```

然后在 vocab 维度计算：

```text
min(p_i(v), q_i(v))
```

这也是为什么 OPD/teacher full vocab 信息很大：TV loss 要对完整 vocab 分布的重叠面积建模，至少概念上依赖 full vocab 的 `p` 和 `q`。

### 6.8 这两项和 ablation 怎么对应

表里的顺序可以这样理解：

```text
Baseline 4.56:
  常规 MTP speculative decoding。

+ IndexShare + KVShare 5.10:
  先减少 MTP 训练/推理 hidden/KV/top-k 分布差异。

+ Rejection Sampling 5.29:
  推理/rollout 的验证规则从更脆弱的方式改成概率式 RS。
  单步接受率由分布重叠面积决定。

+ End-to-end TV Loss 5.47:
  训练 MTP 时直接优化 RS 真正奖励的多步接受长度。
```

一句话概括：

```text
Rejection Sampling 改的是“怎么验 draft token”；
TV Loss 改的是“怎么训练 draft 分布，让它在这种验证规则下更容易被接受”。
```

### 6.9 SGLang 中 GLM-5.2 走哪条路径

SGLang 里有两条相关路径：

```text
FROZEN_KV_MTP:
  draft 不写自己的 KV，只读 target KV。

EAGLE/NextN:
  draft 有自己的 KV pool，draft prefill/decode 正常写 draft KV。
```

GLM-5.2 FP8 的 DSA MTP 测试：

```text
test/registered/models_e2e/test_dsa_glm52_tp_mtp.py:
  model = "zai-org/GLM-5.2-FP8"

python/sglang/test/server_fixtures/dsa_mtp_fixture.py:
  speculative_algorithm = "EAGLE"
  speculative_num_steps = 5
  speculative_eagle_topk = 1
  speculative_num_draft_tokens = 6
```

所以当前这份 SGLang 代码中，GLM-5.2 regression 走的是 EAGLE/NextN，不是 `FROZEN_KV_MTP`。

GLM-5.2 target architecture 是：

```text
GlmMoeDsaForCausalLM
```

但 draft worker 中会被改成：

```text
DeepseekV3ForCausalLMNextN
```

相关逻辑在：

```text
python/sglang/srt/configs/model_config.py
```

所以 GLM-5.2 MTP 的真实执行要重点看：

```text
python/sglang/srt/speculative/eagle_worker_v2.py
python/sglang/srt/models/deepseek_nextn.py
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
python/sglang/srt/layers/attention/dsa_backend.py
```

### 6.10 非 frozen 路径中 draft KV 怎么办

普通 EAGLE worker 明确会分配 draft KV cache pool：

```text
eagle_worker_v2.py:
  def alloc_memory_pool(...):
      """Allocate draft KV cache pools (called by scheduler)."""
```

draft prefill 的注释也明确写着：

```text
Run draft model extend to correctly fill the KV cache.
```

draft decode loop 中每一步会设置：

```text
forward_batch.out_cache_loc = out_cache_loc[i]
spec_info.hidden_states = hidden_states
draft_runner.forward(forward_batch)
```

attention 路径会用 `out_cache_loc` 写当前 draft step 的 KV。

因此，当前 SGLang GLM-5.2 EAGLE/NextN 路径不是“完全不 append `kv5_mtp`”。它仍然会从 `h5_mtp` 生成当前 step 的 Q/KV，并把输出 hidden 传给下一步。

### 6.11 `index_share_for_mtp_iteration` 到底做什么

`index_share_for_mtp_iteration` 控制的是 MTP 多步 draft 之间复用 DSA top-k indices。

它不是 KVShare，也不是“不写 KV”。

代码逻辑大致是：

```text
if hf_config.index_share_for_mtp_iteration and topk == 1:
    forward_batch.reuse_dsa_topk_indices = True
```

然后 `deepseek_nextn.py` 每次 forward 时把 carried indices 传给 decoder：

```text
prev_topk_indices =
    forward_batch.spec_info.dsa_topk_indices
    if forward_batch.reuse_dsa_topk_indices
    else None
```

NextN attention 中，因为 `is_nextn=True`：

```text
skip_topk = True
next_skip_topk = True
```

`should_run_indexer(prev_topk_indices)` 的含义是：

```text
如果 prev_topk_indices 存在:
  不跑 indexer，直接复用这份 top-k

如果 prev_topk_indices 不存在:
  NextN 允许 fallback，自己跑一次 indexer
```

### 6.12 MTP iteration 复用 index 的实际时间线

以 `speculative_num_steps = 5` 为例。

先有一次 `draft_extend`：

```text
target 刚算完真实 token，并返回 target hidden
draft_extend 用 target hidden 跑一次 draft model
产生第一个 draft token / h5_mtp
捕获最后 accepted 位置的 DSA top-k indices
放进 EagleDraftInput.dsa_topk_indices
```

然后进入 `draft_forward`：

```text
i = 0:
  使用 h5_mtp 和 draft token embedding
  复用 seed top-k indices
  forward 一次，产生下一个 draft token / hidden

i = 1:
  复用 carried top-k indices
  forward 一次

i = 2:
  复用 carried top-k indices
  forward 一次

i = 3:
  复用 carried top-k indices
  forward 一次

i = 4:
  不再 forward，只整理 draft token tree
```

所以在这个参数下，可以理解为：

```text
1 个 draft token 来自 draft_extend
4 次 draft model forward 产生后续 draft token
```

如果没有 seed top-k：

```text
step0 自己跑一次 indexer
step1/step2/... 复用 step0 产生的 top-k
```

所以你可以把 `index_share_for_mtp_iteration` 记成：

```text
decode 时 q 长度是 1，会基于历史 indexer K cache 选 top-k 个 KV 位置；
index_share_for_mtp_iteration 让后续 MTP step 复用这份位置列表，
而不是每一步重新用 indexer 选。
```

### 6.13 后面层是否也复用 index

要分场景。

主 target model 有 78 层：

```text
full indexer layer 计算 top-k
后续 shared indexer layers 复用上一层 top-k
```

MTP NextN draft 不是完整 78 层，而是一个轻量 NextN decoder：

```text
draft step 之间复用 top-k
不是完整 78 层都在跑“第一层算、后面层复用”
```

所以：

```text
主干 DSA IndexShare = 层间复用
MTP iteration IndexShare = 步间复用
```

它们可以同时存在，但不要混成一个概念。

### 6.14 Frozen-KV MTP 是另一条路径

SGLang 里确实有真正 frozen KV 的 worker：

```text
python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py
```

它的注释很直接：

```text
The frozen draft reads the target KV cache read-only and owns no KV pool
Frozen draft never writes KV
```

这条路径会：

```text
out_cache_loc = None
draft attention backend 读 target KV pool
position 固定到最后已提交 target token
不把 draft KV append 到 draft/target KV cache
```

这才符合“新的 q 读旧 target KV，不写 draft KV”的解释。

但当前 GLM-5.2 FP8 DSA MTP regression 不是走这条路径。

这里的命名要特别小心：`FROZEN_KV_MTP` 更像是 SGLang 对这条 worker/algorithm 的工程命名，不一定是 Google/Gemma 官方文档里的标准叫法。

公开资料里更常见的是：

```text
Gemma 4 Assistant / Gemma 4 MTP:
  官方 Google AI Developers 文档叫 Multi-Token Prediction，
  强调 assistant/draft model 共享输入 embedding，
  使用 target model last-layer activations。

Transformers Gemma4 Assistant 文档:
  明确说 Gemma4Assistant 整个模型 uses KV sharing，
  可以复用 target model 已经填好的 KV cache，
  从而 skip prefill。

SGLang:
  把这种“assistant 读 target KV、自己不维护普通 draft KV”的推理路径
  命名成 FROZEN_KV_MTP。
```

所以不要把“Gemma 4 第一次提出 Frozen-KV MTP”当成严格历史结论。更稳妥的说法是：

```text
Gemma 4 Assistant 是当前 SGLang 中自动触发 FROZEN_KV_MTP 的主要模型家族；
SGLang 的 FROZEN_KV_MTP 名字对应 Gemma4Assistant 的 KV sharing 推理实现；
但 KV sharing/frozen KV 这个思想在官方材料中不一定以 FROZEN_KV_MTP 这个名字出现。
```

开启方式有两个层面：

```text
显式配置:
  --speculative-algorithm FROZEN_KV_MTP

自动 promotion:
  如果用户写的是 NEXTN 或 EAGLE，
  且 speculative_draft_model_path 的 architecture 是
    Gemma4AssistantForCausalLM
    或 Gemma4UnifiedAssistantForCausalLM
  SGLang 会自动把算法提升成 FROZEN_KV_MTP。
```

也就是说，它不是 GLM-5.2 config 里的一个普通开关；自动开启基本绑定在特定 draft model architecture 上。即使显式指定 `FROZEN_KV_MTP`，draft model 也需要实现：

```text
build_frozen_kv_mtp_context(...)
bind_frozen_kv_context(...)
```

否则 worker 无法真正把 draft attention 绑定到 target KV pool。

## 7. 总结对照

### 7.1 训练 forward

```text
input_ids
  -> embedding
  -> RoPE positions
  -> 78 decoder layers
      -> MLA-style attention
      -> DSA indexer 选 top-k
      -> sparse attention
      -> dense MLP / MoE
  -> final norm
  -> lm_head
  -> loss
```

训练时重点是完整序列并行计算，通常不使用 KV cache。

### 7.2 主模型推理 forward

```text
prefill:
  输入 prompt 多 token
  建立 main KV cache
  建立 indexer K cache
  full 层计算 top-k，shared 层复用

decode:
  输入 1 个 token
  append main KV cache
  append indexer K cache
  full 层用当前 q_idx 检索历史 indexer K cache
  attention 只读 top-k 指向的 main KV
```

### 7.3 MTP 非 frozen 推理

```text
draft_extend:
  用 target hidden 产生第一个 draft token
  填 draft KV
  捕获 DSA top-k seed

draft_forward:
  每步输入上一轮 MTP hidden + draft token embedding
  每步正常写 draft KV
  index_share_for_mtp_iteration 让后续 step 复用 top-k indices
```

### 7.4 一句话总结

GLM-5.2 主干 forward 的核心是 MLA + DSA + MoE；DSA 通过 indexer K cache 检索 top-k 历史位置，再让主 attention 只读这些位置的主 KV。IndexShare 有两层含义：主干层间复用 top-k，MTP iteration 之间也可以复用 top-k。当前 SGLang GLM-5.2 EAGLE/NextN 路径明确实现了 MTP top-k indices 复用，但仍正常写 draft KV；完全不写 draft KV 的是单独的 `FROZEN_KV_MTP` 路径。
