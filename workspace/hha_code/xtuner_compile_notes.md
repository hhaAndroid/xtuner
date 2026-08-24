# XTuner torch.compile 学习笔记

本文分阶段梳理 XTuner 里的 `torch.compile` 设计。

第一部分先只理解一件事：

```text
@maybe_compile 解决“顶层函数能不能被运行时替换”的问题。
compile_cfg 解决“这次模型构建到底编译谁、用什么参数编译”的问题。
```

相关源码：

- `xtuner/v1/utils/compile.py`
- `xtuner/v1/model/base.py`
- `xtuner/v1/float8/float8_linear_tensor_wise.py`
- `xtuner/v1/module/decoder_layer/dense_decoder_layer.py`
- `xtuner/v1/model/dense/dense.py`

## 1. 先看一个顶层函数例子

例子：`per_tensor_fp8_quant`

位置：`xtuner/v1/float8/float8_linear_tensor_wise.py`

```python
@maybe_compile
def per_tensor_fp8_quant(
    tensor: torch.Tensor,
    float8_dtype=torch.float8_e4m3fn,
):
    ...
```

这是一个模块顶层函数，不属于某个 class。

它同时被注册进默认 FP8 compile 配置：

位置：`xtuner/v1/model/base.py`

```python
DEFAULT_FLOAT8_CFG = {
    "xtuner.v1.float8.float8_linear_tensor_wise.per_tensor_fp8_quant": TorchCompileOption(fullgraph=True),
}
```

所以它的生效链路是：

```text
per_tensor_fp8_quant 被 @maybe_compile 包住
        |
        v
模型的 compile_cfg 里必须包含这个函数全名
        |
        v
模型构建时调用 _maybe_enable_compile(self.compile_cfg)
        |
        v
_compile_overwrite 找到这个目标
        |
        v
发现它是 MaybeCompile 对象
        |
        v
调用 MaybeCompile.enable_compile(...)
        |
        v
wrapper 内部 self.func 从原函数变成 torch.compile(origin_func, ...)
```

这里要注意：`@maybe_compile` 本身不会立刻 compile。

要理解有了装饰器，为啥还需要在外边的 compile_cfg 里面写一遍才真的生效？ 是因为我们希望可以外面通过配置关闭 compile 功能。如果 @ 后就一定开启，那么就没法关掉了。所以对应函数来说 @ 装饰器只是表示这个函数可以被 compile，但是是否真的开启要考虑用户设置 compile_cfg 。

`MaybeCompile` 初始化时大致是：

```python
self.origin_func = func
self.func = func
```

调用时是：

```python
return self.func(*args, **kwargs)
```

所以默认还是 eager。只有外部 `compile_cfg` 命中它，并调用 `enable_compile` 后，才会变成：

```python
self.func = torch.compile(self.origin_func, **compile_options)
```

因此：

```text
@maybe_compile:
  让顶层函数具备运行时切换能力

compile_cfg:
  决定这次是否真的切换，以及 torch.compile 参数是什么
```



## 2. 为什么顶层函数需要 @maybe_compile

顶层函数的问题来自 Python import 引用。

假设有一个函数：

```python
# a.py
def foo(x):
    return x + 1
```

另一个文件提前 import 了它：

```python
# b.py
from a import foo
```

如果运行时只在 `a.py` 里做：

```python
a.foo = torch.compile(a.foo)
```

`b.py` 里已经拿到的 `foo` 引用不一定会变。它可能还指向旧函数。

`@maybe_compile` 的思路是：让大家 import 到的不是裸函数，而是同一个 wrapper 对象。

```text
b.py 里的 foo
a.py 里的 foo
        |
        v
同一个 MaybeCompile wrapper
        |
        v
wrapper.func 可以从 eager 函数切到 compiled 函数
```

这样运行时不需要替换所有 import 出去的引用，只需要改 wrapper 内部的 `self.func`。

这就是顶层函数要多绕一层的原因。

## 3. 再看一个类方法例子

例子：`DenseDecoderLayer.forward`

位置：`xtuner/v1/module/decoder_layer/dense_decoder_layer.py`

```python
class DenseDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        ...
```

这个方法没有 `@maybe_compile`。

但 Dense 默认 compile_cfg 配置里注册了它：

位置：`xtuner/v1/model/dense/dense.py`

```python
DENSE_COMPILE_CFG = {
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}
```

模型构建时，`_compile_overwrite` 找到它后，会判断它是 class-level function，然后直接改 class 上的方法属性：

位置：`xtuner/v1/model/base.py`

```python
setattr(cls, method_name, torch.compile(compiled_function, **compile_options))
```

也就是等价于：

```python
DenseDecoderLayer.forward = torch.compile(DenseDecoderLayer.forward, fullgraph=True)
```

后续所有实例：

```python
layer = DenseDecoderLayer(...)
layer.forward(...)
```

都会从 `DenseDecoderLayer` 这个 class 上拿到新的 compiled `forward`。

所以类方法不需要 `@maybe_compile`，因为直接替换 class attribute 就能影响实例调用。

## 4. 两类目标的对比

```text
顶层函数:
  例子: per_tensor_fp8_quant
  需要: @maybe_compile
  原因: 可能被其他模块提前 import 成旧引用
  生效方式: compile_cfg 命中后，替换 wrapper.func

类方法:
  例子: DenseDecoderLayer.forward
  需要: 不需要 @maybe_compile
  原因: 实例方法通过 class attribute 解析
  生效方式: compile_cfg 命中后，setattr(cls, method_name, compiled_method)
```



## 5. 当前阶段的最小理解

现在只需要记住这条主线：

```text
compile_cfg 是统一入口。

如果目标是顶层函数:
  它必须先被 @maybe_compile 包住，否则 XTuner 会报错。

如果目标是类方法:
  不需要 @maybe_compile，XTuner 可以直接替换 class 上的方法。
```

`@maybe_compile` 和 `compile_cfg` 不是重复设计。

```text
@maybe_compile 是技术适配层:
  让顶层函数可以被运行时切换。

compile_cfg 是策略配置层:
  决定哪些目标真的启用 torch.compile，以及使用 fullgraph/dynamic/mode 等参数。
```



## 6. 第二部分：compile_cfg 怎么进入模型构建流程

第一部分只解释了：

```text
compile_cfg 命中目标后，顶层函数和类方法分别怎么变成 compiled callable。
```

第二部分看上游：

```text
模型构建时，compile_cfg 是怎么从 config 解析出来，并最终传给 _maybe_enable_compile 的。
```

还是先用 Dense 模型做主线。

## 7. compile_cfg 首先是 model config 的字段

位置：`xtuner/v1/model/base.py`

```python
class XTunerBaseModelConfig(PydanticBaseModel):
    compile_cfg: dict[str, TorchCompileOption] | None | bool = None
```

它有三种主要语义：

```text
compile_cfg = None 或 True:
  使用当前模型类自己的 default_compile_cfg

compile_cfg = False:
  关闭 compile

compile_cfg = dict:
  用户手动指定要 compile 的目标和参数
```

所以 `compile_cfg` 不是一个简单 bool。

它既可以表达开关：

```python
model_cfg.compile_cfg = False
```

也可以表达精确策略：

```python
model_cfg.compile_cfg = {
    "some.module.SomeClass.forward": {"fullgraph": True},
}
```



## 8. BaseModel.**init** 先解析 compile_cfg

Dense 模型初始化时，第一步会调用父类：

位置：`xtuner/v1/model/dense/dense.py`

```python
class Dense(BaseModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        ...
```

进入 `BaseModel.__init__`：

位置：`xtuner/v1/model/base.py`

```python
class BaseModel(nn.Module):
    def __init__(self, config: XTunerBaseModelConfig):
        super().__init__()
        self.config = config
        ...
        self._compile_cfg = self._resolve_compile_cfg(self.config)
```

注意这里还没有真正调用 `torch.compile`。

这里只是把用户传入的 `config.compile_cfg` 解析成模型内部的：

```python
self._compile_cfg
```

也就是说，这一步做的是：

```text
外部配置 compile_cfg
        |
        v
BaseModel._resolve_compile_cfg(...)
        |
        v
内部字段 self._compile_cfg
```



## 9. _resolve_compile_cfg 的规则

位置：`xtuner/v1/model/base.py`

核心逻辑可以简化成：

```python
custom_cfg = config.compile_cfg

if custom_cfg is False:
    self._disable_compile_cfg(self.config)
    return {}

if DEVICE == "npu":
    self._disable_compile_cfg(self.config)
    return {}

if custom_cfg is True or custom_cfg is None:
    compile_cfg = self.default_compile_cfg
else:
    compile_cfg = custom_cfg

return compile_cfg
```

所以它的决策表是：

```text
config.compile_cfg = False:
  返回 {}
  同时递归把子 config 的 compile_cfg 也设成 False

DEVICE == "npu":
  返回 {}
  因为当前实现里 NPU 不支持 torch.compile

config.compile_cfg = None 或 True:
  返回 self.default_compile_cfg

config.compile_cfg = dict:
  返回用户传入的 dict
```

这里有一个关键点：

```text
default_compile_cfg 是模型类决定的。
```

不是 `BaseModel` 统一决定所有模型要 compile 什么。

## 10. Dense 的 default_compile_cfg

`BaseModel` 的默认实现是空：

位置：`xtuner/v1/model/base.py`

```python
@property
def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
    return {}
```

Dense 覆盖了这个属性：

位置：`xtuner/v1/model/dense/dense.py`

```python
@property
def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
    return DENSE_COMPILE_CFG
```

而 `DENSE_COMPILE_CFG` 是：

```python
DENSE_COMPILE_CFG = {
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}
```

所以当我们构建 Dense，并且没有手动关闭 compile 时：

```python
model_cfg.compile_cfg = None
```

实际会解析成：

```text
self._compile_cfg = DENSE_COMPILE_CFG
```

也就是：

```text
DenseDecoderLayer.forward
FP8 相关 MaybeCompile 函数
```

这些会成为后续真正启用 compile 的目标。

## 11. 为什么 self.compile_cfg 不是简单返回 self._compile_cfg

位置：`xtuner/v1/model/base.py`

```python
@property
def compile_cfg(self) -> dict[str, TorchCompileOption]:
    _compile_cfg = self._compile_cfg.copy()
    for module in self.modules():
        if isinstance(module, BaseModel) and module is not self:
            sub_custom_cfg = module.compile_cfg
            _compile_cfg |= sub_custom_cfg

    return _compile_cfg
```

这里不是简单返回 `self._compile_cfg`，而是会继续遍历子模块。

原因是有些模型是组合模型，比如多模态模型里可能包含：

```text
BaseComposeModel
  vision_tower
  multi_modal_projector
  language_model
```

其中 `language_model` 本身也可能是一个 `BaseModel`，也有自己的 `compile_cfg`。

所以 `self.compile_cfg` 做的是合并：

```text
当前模型自己的 _compile_cfg
        +
所有 BaseModel 子模块的 compile_cfg
```

Dense 这种单体模型通常不明显；组合模型里这个逻辑更重要。

## 12. 什么时候真正调用 _maybe_enable_compile

还是看 Dense。

位置：`xtuner/v1/model/dense/dense.py`

```python
class Dense(BaseModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.norm = RMSNorm(...)
        self.lm_head = LMHead(...)
        self.layers = self.build_layers(config)
        self.rotary_emb = self.build_rotary_embedding(config)
        self.embed_tokens = self.build_embeddings(config)

        self._init_load_spec()
        self._maybe_enable_compile(self.compile_cfg)
```

这里才是真正启用 compile 的地方。

顺序很重要：

```text
super().__init__(config)
  只解析 compile_cfg，得到 self._compile_cfg

构建具体模块:
  norm / lm_head / layers / rotary_emb / embeddings

self._maybe_enable_compile(self.compile_cfg)
  遍历 compile_cfg，逐个调用 _compile_overwrite
```

`_maybe_enable_compile` 很简单：

位置：`xtuner/v1/model/base.py`

```python
def _maybe_enable_compile(self, compile_cfg):
    if compile_cfg:
        torch._dynamo.config.cache_size_limit = 256

    for target, option in compile_cfg.items():
        self._compile_overwrite(target, option)
```

也就是：

```text
compile_cfg 为空:
  什么也不做

compile_cfg 非空:
  调高 dynamo cache_size_limit
  对每个目标调用 _compile_overwrite
```

第一部分讲过，`_compile_overwrite` 后面会分两种情况：

```text
目标是 @maybe_compile 顶层函数:
  调 MaybeCompile.enable_compile(...)

目标是类方法:
  setattr(cls, method_name, torch.compile(...))
```



## 13. 用 Dense 串起完整链路

假设用户没有显式设置：

```python
model_cfg = Qwen3Dense8BConfig()
```

默认：

```text
model_cfg.compile_cfg = None
```

构建模型时：

```text
Dense.__init__
        |
        v
BaseModel.__init__
        |
        v
_resolve_compile_cfg(config)
        |
        v
config.compile_cfg 是 None，所以使用 self.default_compile_cfg
        |
        v
Dense.default_compile_cfg 返回 DENSE_COMPILE_CFG
        |
        v
self._compile_cfg = DENSE_COMPILE_CFG
        |
        v
Dense 继续构建 layers / norm / lm_head 等模块
        |
        v
self._maybe_enable_compile(self.compile_cfg)
        |
        v
遍历 DENSE_COMPILE_CFG
        |
        v
DenseDecoderLayer.forward 被替换成 compiled method
        |
        v
DEFAULT_FLOAT8_CFG 里的 @maybe_compile 函数被 enable_compile
```

这个流程里有两个阶段要分清：

```text
解析阶段:
  BaseModel.__init__
  只决定 self._compile_cfg 是什么

启用阶段:
  Dense.__init__ 后半段
  调 _maybe_enable_compile，才真正调用 torch.compile
```



## 14. 关闭 compile 时发生什么

如果用户设置：

```python
model_cfg.compile_cfg = False
```

那么在 `BaseModel.__init__` 里：

```text
_resolve_compile_cfg
        |
        v
发现 custom_cfg is False
        |
        v
调用 _disable_compile_cfg(self.config)
        |
        v
返回 {}
```

最后：

```text
self._compile_cfg = {}
self.compile_cfg = {}
_maybe_enable_compile({})
```

结果就是不会 compile 任何目标。

这里 `_disable_compile_cfg` 是递归的，会把子 config 里的 `compile_cfg` 也关掉。

这对组合模型很重要，因为只设置最外层：

```python
model_cfg.compile_cfg = False
```

就能把里面的 text model / vision model 等子配置一起关掉。

## 15. 自定义 compile_cfg 时发生什么

如果用户传的是 dict：

```python
model_cfg.compile_cfg = {
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": {
        "fullgraph": True,
    }
}
```

那么：

```text
_resolve_compile_cfg 不再使用 default_compile_cfg
而是直接返回用户传入的 dict
```

也就是说，自定义 dict 是覆盖默认策略，不是和默认策略自动合并。

如果用户还想保留 FP8 默认 compile 目标，需要自己把那些目标也放进 dict，或者在代码里基于默认配置拷贝后修改。

## 16. 第二部分的小结

第二部分的最小理解是：

```text
compile_cfg 先在 config 上表达用户意图。

BaseModel.__init__:
  把 config.compile_cfg 解析成 self._compile_cfg。

具体模型 __init__:
  先把 layers 等模块建好。
  然后调用 _maybe_enable_compile(self.compile_cfg)。

_maybe_enable_compile:
  遍历 compile_cfg。
  对每个目标调用 _compile_overwrite。
```

Dense 默认路径可以压缩成一句话：

```text
Qwen3Dense8BConfig(compile_cfg=None)
  -> Dense.default_compile_cfg
  -> DENSE_COMPILE_CFG
  -> _maybe_enable_compile
  -> DenseDecoderLayer.forward 和 FP8 helper 被 compile
```

到这里为止，我们已经理解了两层：

```text
第一层:
  单个目标怎么被 compile。

第二层:
  compile_cfg 怎么从模型配置进入模型构建流程。
```



## 17. 第三部分：recompute 之后为什么还会手动 compile wrapper.forward

前两部分讲的是模型构建期的默认 compile 机制：

```text
compile_cfg
  -> _maybe_enable_compile
  -> _compile_overwrite
```

但 Dense 里还有一段看起来像“绕开 compile_cfg”的代码。

位置：`xtuner/v1/model/dense/dense.py`

```python
if layer_idx < num_recompute_layers:
    layer = checkpoint_wrapper(
        layer,
        preserve_rng_state=checkpoint_preserve_rng_state,
        checkpoint_impl=CheckpointImpl.REENTRANT,
    )

    if self.compile_cfg:
        fullgraph = self.config.layers_type[layer_idx] != "linear_attention"
        layer.forward = torch.compile(layer.forward, fullgraph=fullgraph)
```

这段代码解决的是另一个问题：

```text
checkpoint_wrapper 会在运行时把原始 layer 包成一个新的 module。

compile_cfg 能 compile 原始 DenseDecoderLayer.forward，
但它不会自动 compile checkpoint_wrapper 之后新产生的 wrapper.forward。
```

所以 Dense 在 recompute 分支里做了额外处理：

```text
原始 layer
        |
        v
checkpoint_wrapper(layer)
        |
        v
新的 wrapped layer
        |
        v
torch.compile(wrapped_layer.forward)
```

也就是说，在 recompute 这条路径里，compile 是包在 checkpoint wrapper 外面的：

```text
compiled checkpoint wrapper forward
```

而不是简单地只 compile 原始 `DenseDecoderLayer.forward`。

## 18. 为什么这段不能只靠 compile_cfg

`compile_cfg` 适合处理稳定、可定位的目标，比如：

```text
xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward
xtuner.v1.float8.float8_linear_tensor_wise.per_tensor_fp8_quant
```

这些目标可以通过字符串全名定位。

但 `checkpoint_wrapper(layer)` 返回的是运行时新对象。是否产生这个 wrapper 取决于：

```text
recompute_ratio
layer_idx
```

而且这里只需要处理被 recompute 的 layer，不是所有 layer。

所以这类 per-instance、recompute 后才出现的 wrapper.forward，不适合直接写进全局 `compile_cfg`。

这里的 `if self.compile_cfg:` 只是把 compile_cfg 当成总开关：

```text
用户关闭 compile:
  不 compile checkpoint wrapper.forward

用户开启 compile:
  对 recompute 后的 wrapper.forward 再补一层 compile
```



## 19. fullgraph 为什么还要按 layer 类型判断

Dense 里还有这个判断：

```python
fullgraph = self.config.layers_type[layer_idx] != "linear_attention"
```

注意，这个判断只发生在 recompute 分支里：

```python
if layer_idx < num_recompute_layers:
    layer = checkpoint_wrapper(...)
    layer.forward = torch.compile(layer.forward, fullgraph=fullgraph)
```

也就是说，它控制的是：

```text
checkpoint wrapper 之后的 wrapper.forward
```

不是模型构建期默认的：

```text
DenseDecoderLayer.forward
```

普通默认 compile 仍然是：

```text
DenseDecoderLayer.forward fullgraph=True
```

只有被 checkpoint wrapper 包住的 layer，才按 layer 类型重新设置 wrapper 外层 compile 的 `fullgraph`。

被 checkpoint wrapper 包住的 full attention layer：

```text
fullgraph=True
```

被 checkpoint wrapper 包住的 linear attention layer：

```text
fullgraph=False
```

原因是 linear attention / GatedDeltaNet 相关层会写 `seq_ctx.seq_idx` 这类状态。

当 checkpoint wrapper 的 forward 被 `torch.compile(fullgraph=True)` 包住时，checkpoint 可能被 Dynamo 表示成 HigherOrderOperator。这个场景下，fullgraph 对副作用更敏感，写 `seq_ctx.seq_idx` 这种 side effect 会出问题。

所以 linear attention layer 仍然 compile checkpoint wrapper 后的 `wrapper.forward`，但用：

```text
fullgraph=False
```

允许必要的 graph break。

## 20. Dense 为什么有这段，MoE 为什么没有

这里有一个很重要的经验判断：

```text
Dense 测试发现：
  compile 包在 checkpoint wrapper 外面有更好的加速效果。

MoE 测试发现：
  这个收益不明显，所以没有写类似逻辑。
```

所以 Dense 里额外加了：

```python
layer.forward = torch.compile(layer.forward, fullgraph=fullgraph)
```

但 MoE 的 recompute 分支没有这段。

位置：`xtuner/v1/model/moe/moe.py`

```python
for layer_idx, layer in self.layers.items():
    if self._should_recompute(layer_idx=layer_idx, mtp_idx=None):
        layer = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.REENTRANT)

    self.layers[str(layer_idx)] = layer
    self._fully_shard(..., module=layer)
```

MoE 当前只做：

```text
layer = checkpoint_wrapper(layer)
```

没有做：

```text
layer.forward = torch.compile(layer.forward)
```

这和 MoE 的结构也一致。MoE forward 里有更多复杂成分：

```text
router
dispatcher / all-to-all
expert parallel
aux loss
可能的 offload
```

所以 MoE 默认更倾向于 compile 内部稳定片段，而不是把 checkpoint wrapper 后的整层 forward 再包一层 compile。

## 21. Qwen3 compose 模型里的情况

多模态 compose 要分模块看。

Qwen3VL 的 vision tower 有类似 Dense 的写法。

位置：`xtuner/v1/model/compose/qwen3_vl/modeling_vision.py`

```python
if layer_idx < num_recompute_layers:
    layer = checkpoint_wrapper(
        layer,
        preserve_rng_state=checkpoint_preserve_rng_state,
        checkpoint_impl=CheckpointImpl.REENTRANT,
    )
    if self.compile_cfg:
        layer.forward = torch.compile(layer.forward, fullgraph=True)
```

所以 Qwen3VL vision 是：

```text
vision layer
  -> checkpoint_wrapper(layer)
  -> compile wrapper.forward
```

InternS1 vision 也有类似逻辑，但多了 `drop_path_rate == 0.0` 条件：

位置：`xtuner/v1/model/compose/intern_s1/modeling_vision.py`

```python
if layer_idx < num_recompute_layers:
    layer = checkpoint_wrapper(...)
    if self.config.drop_path_rate == 0.0 and self.compile_cfg:
        layer.forward = torch.compile(layer.forward, fullgraph=True)
```

而 compose 里的 language model 要看具体 text model：

```text
text model 是 Dense:
  走 Dense 逻辑，有 checkpoint wrapper 后再 compile wrapper.forward

text model 是 MoE:
  走 MoE 逻辑，没有额外 compile wrapper.forward
```



## 22. 第三部分的小结

这一部分要补充到前面的理解里：

```text
compile_cfg 负责静态、可定位的 compile 目标。

checkpoint_wrapper 后的 wrapper.forward 是运行时新对象，
所以 Dense / vision tower 里会手动补一层 torch.compile。
```

Dense 的 recompute 路径是：

```text
原始 Dense layer
  -> 可能已经通过 compile_cfg compile 原始 class method
  -> checkpoint_wrapper(layer)
  -> 再 compile wrapper.forward
```

MoE 的 recompute 路径是：

```text
原始 MoE layer / 内部函数
  -> 通过 compile_cfg compile 稳定片段
  -> checkpoint_wrapper(layer)
  -> 不再额外 compile wrapper.forward
```

这里不是设计不一致，而是性能收益和模块复杂度不同：

```text
Dense:
  wrapper 外层 compile 有明确加速收益，所以加。

MoE:
  收益不明显，且整层 forward 更复杂，所以不加。
```



## 23. 补充：外层 compile 和双层 compile 不是一回事

这里要特别区分两个概念：

```text
外层 compile:
  torch.compile(checkpoint_wrapper.forward)

双层 compile:
  torch.compile(checkpoint_wrapper.forward)
    -> 内部又调用已经 compiled 的原始 layer.forward
```

Dense 里真正被测试验证更快的点，是：

```text
compile 包在 checkpoint wrapper 外面，
比只 compile checkpoint wrapper 里面的原始 layer 更快。
```

也就是外层 compile 有收益。

但当前代码里，因为模型构建期已经通过 `compile_cfg` 把 `DenseDecoderLayer.forward` 这个 class method compile 了，到了 FSDP/recompute 阶段又对 checkpoint wrapper 后的 `layer.forward` 做 compile，于是实际调用链可能变成：

```text
compiled(checkpoint_wrapper.forward)
    -> checkpoint 逻辑
        -> compiled(DenseDecoderLayer.forward)
```

这个“双层 compile”更像是当前架构下自然产生的结果，而不是明确为了追求双层收益而设计的。

更准确地说：

```text
外层 wrapper.forward compile:
  是 Dense 里保留这段手写 torch.compile 的主要原因。

内层 DenseDecoderLayer.forward 也已经 compile:
  是 compile_cfg class-level 全局策略带来的结果。
```



## 24. 双层 compile 可能有什么风险

当前判断是：

```text
双层 compile 不是语义上天然非法。
```

`torch.compile(torch.compile(fn))` 这种形式在当前 PyTorch 里可以运行。`CheckpointWrapper.forward` 内部会调用 `_checkpoint_wrapped_module`，如果原始 module 的 `forward` 已经被 class-level compile，那么外层 compiled wrapper 确实会调用到内层 compiled module。

但这不代表完全没有成本或风险。

主要风险有：

```text
1. 首次编译成本更高
   外层 wrapper.forward 有自己的 Dynamo/Inductor 编译。
   内层 layer.forward 也有自己的编译。

2. cache 和 guard 更多
   双层 compiled callable 会增加编译缓存和 guard 管理压力。

3. 外层图可能把内层 compiled callable 当成不透明调用
   这样不一定能获得一个完全融合的大图。

4. checkpoint + fullgraph + side effect 更敏感
   Dense 里 linear_attention 要用 fullgraph=False，
   就是因为 checkpoint wrapper 被 compile 后，
   fullgraph 对写 seq_ctx.seq_idx 这类副作用更敏感。

5. 调试和版本兼容风险更高
   如果遇到 graph break、重复 recompile、checkpoint backward 异常，
   双层 compile 是需要优先排查的因素。
```

所以目前更稳妥的结论是：

```text
内外都套 compile 不一定有问题。
但它也不是完全“无所谓”。

已知收益点是外层 wrapper.forward compile。
内层原始 layer.forward compile 是否仍然必要，需要单独 benchmark / ablation。
```

理想上，如果只想验证外层 compile 的收益，可以尝试一种更干净的对照：

```text
recompute layer:
  不提前 compile 原始 DenseDecoderLayer.forward
  只在 checkpoint_wrapper 之后 compile wrapper.forward

non-recompute layer:
  仍然按 compile_cfg compile 原始 DenseDecoderLayer.forward
```

但当前 XTuner 的 `compile_cfg` 是 class-level 全局覆盖：

```text
DenseDecoderLayer.forward
```

它不是按 layer instance、也不是按 recompute 状态精细控制。所以要去掉 recompute layer 的内层 compile，不是简单改一行配置就能做到。

这个点后续需要再和专家确认：

```text
当前双层 compile 是否只是可接受的实现折中？
有没有必要引入 per-layer / per-instance 的 compile 控制，
让 recompute layer 只 compile wrapper.forward？
```



## 25. 当前设计边界：类方法 compile 是 class-level，不是 instance-level

还有一个需要特别注意的点：

```text
compile_cfg 里的类方法目标是 class-level 覆盖。
```

也就是说，如果配置里有：

```python
"xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": {
    "fullgraph": True,
}
```

最终 `_compile_overwrite` 做的是：

```python
setattr(cls, method_name, torch.compile(compiled_function, **compile_options))
```

等价于：

```python
DenseDecoderLayer.forward = torch.compile(DenseDecoderLayer.forward, fullgraph=True)
```

这不是只改某一个 layer instance。

它会影响当前进程里所有 `DenseDecoderLayer` 实例。

所以当前 compile 控制粒度是：

```text
顶层函数:
  function-level，全局 wrapper 生效

类方法:
  class-level，全 class 生效
```

不是：

```text
module instance-level
```



## 26. 如果 ViT 和 LLM 用不同 class，问题不大

如果视觉部分和语言模型部分用的是不同类，比如：

```text
vision:
  Qwen3VLVisionLayer.forward

text:
  DenseDecoderLayer.forward
  或 MoEDecoderLayer.forward
```

那么它们是不同 compile target，可以分别控制。

例如想让 LLM 开 compile、vision 不开，思路是：

```python
model_cfg.compile_cfg = {}
model_cfg.vision_config.compile_cfg = False
model_cfg.text_config.compile_cfg = None
```

这里的含义是：

```text
model_cfg.compile_cfg = {}:
  不使用 compose 外层默认 compile_cfg，避免外层把 vision/projector 目标加回来。

vision_config.compile_cfg = False:
  vision 自己关闭 compile。

text_config.compile_cfg = None:
  text model 仍然使用自己的 default_compile_cfg。
```

不过这需要结合具体 compose 模型确认，因为有些 compose 的默认 compile 配置本身就包含 vision/projector 目标。

## 27. 如果 ViT 和 LLM 复用同一个 class，当前设计不够完善

真正麻烦的是这种情况：

```text
vision 和 text 都复用了同一个模块类:

SomeSharedBlock.forward
```

如果 LLM 侧希望 compile：

```python
"xxx.SomeSharedBlock.forward": {"fullgraph": True}
```

那么实际发生的是：

```python
SomeSharedBlock.forward = torch.compile(SomeSharedBlock.forward)
```

这会让 vision 里的 `SomeSharedBlock` 实例也一起变成 compiled forward。

也就是说，当前设计无法干净表达：

```text
LLM 里的 SomeSharedBlock:
  开 compile

ViT 里的 SomeSharedBlock:
  不开 compile
```

因为它们共享同一个 class method。

这不是配置写法的问题，而是当前 compile 机制的粒度问题。

## 28. @maybe_compile 顶层函数也有类似边界

`@maybe_compile` 顶层函数也是全局 wrapper。

如果 vision 和 text 都调用同一个被 `@maybe_compile` 包住的顶层函数：

```python
@maybe_compile
def shared_op(...):
    ...
```

一旦 `compile_cfg` 里启用了：

```python
"some.module.shared_op": {"fullgraph": True}
```

那么 wrapper 内部会变成：

```python
shared_op.func = torch.compile(shared_op.origin_func, ...)
```

所有调用这个 wrapper 的路径都会走 compiled function。

所以顶层函数也是：

```text
function-level 全局控制
```

不是按调用方、按模块实例分别控制。

## 29. 这个边界可能怎么改

如果未来要支持更细粒度控制，可能需要引入 instance-level 或 module-path-level 的 compile 机制。

几个可能方向：

```text
1. 不再只写 class method target
   而是支持具体模块路径，比如:
   language_model.layers.0.forward
   vision_tower.blocks.0.forward

2. 在 build 完成后，对具体 instance 的 bound method 做 compile
   例如只替换某个 layer.forward，
   而不是替换 SomeLayer.forward 这个 class attribute。

3. 对共享模块类拆 subclass / wrapper class
   让 vision 和 text 拥有不同 class target。

4. 对 recompute layer 单独控制
   让 recompute layer 可以只 compile checkpoint wrapper.forward，
   而不提前 class-level compile 原始 layer.forward。
```

这些方向都比当前实现复杂，因为需要处理：

```text
模块构建顺序
FSDP wrapping 顺序
checkpoint_wrapper 后的新 module
配置如何稳定描述具体 instance
保存 / 加载 / 日志打印时如何表达 compile 状态
```

所以当前 XTuner 的实现可以理解为一个务实取舍：

```text
优点:
  简单，配置稳定，容易通过字符串定位目标。

缺点:
  粒度较粗，只能按函数或 class method 控制。
  对“同一个 class 的不同实例使用不同 compile 策略”支持不完善。
```



## 30. 例子：Qwen3.5 Dense 和 MoE 的默认 compile 设置

这一节把前面的机制落到两个具体模型上：

```text
Qwen3_5_VLDense4BConfig
Qwen3_5_VLMoE35BA3Config
```

它们都是 Qwen3VL compose 结构：

```text
Qwen3.5 VL model
  vision_tower
  multi_modal_projector
  language_model
```

所以默认 compile 设置要分两层看：

```text
compose 外层:
  vision / projector

text tower:
  Dense 或 MoE 自己的 decoder layer / attention / expert 等
```



## 31. Qwen3.5 Dense / MoE 共同的 vision 和 projector compile

Qwen3.5 的 compose 基类继承自 Qwen3VL compose。

在当前默认使用 `torch >= 2.9.1` 的前提下，Qwen3VL compose 外层默认 compile 配置是：

位置：`xtuner/v1/model/compose/qwen3_vl/modeling_qwen3_vl.py`

```python
QWEN3VL_COMPILE_CFG = {
    "xtuner.v1.model.compose.qwen3_vl.modeling_projector.Qwen3VLProjector.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}
```

所以 Qwen3.5 Dense 和 Qwen3.5 MoE 共同拥有这部分默认 compile：

```text
Qwen3VLProjector.forward:
  fullgraph=True

Qwen3VLVisionLayer.forward:
  fullgraph=True

DEFAULT_FLOAT8_CFG:
  FP8 相关 helper 函数 fullgraph=True
```

## 32. Qwen3.5 Dense 的 text tower compile

Qwen3.5 Dense 配置是：

位置：`xtuner/v1/model/compose/qwen3_5/qwen3_5_config.py`

```python
class Qwen3_5_VLDense4BConfig(Qwen3_5_BaseConfig):
    vision_config = Qwen3_5_VisionConfig(...)
    projector_config = Qwen3_5_ProjectorConfig(...)
    text_config = Qwen3_5_VLTextDense4BConfig()
```

`Qwen3_5_VLTextDense4BConfig` 构建的是 `Qwen3_5_VLTextDense`。

`Qwen3_5_VLTextDense` 没有单独覆盖 `default_compile_cfg`，它继承 Dense 通用配置：

位置：`xtuner/v1/model/dense/dense.py`

```python
DENSE_COMPILE_CFG = {
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}
```

所以 Qwen3.5 Dense 的 text tower 默认 compile 是：

```text
DenseDecoderLayer.forward:
  fullgraph=True

DEFAULT_FLOAT8_CFG:
  FP8 helper fullgraph=True
```

它不会 compile 整个 `Qwen3_5_VLTextDense.forward`。

它选择 compile 每个 decoder layer：

```text
Qwen3_5_VLTextDense.forward
  for decoder_layer in layers:
    decoder_layer(...)
```

其中 `decoder_layer(...)` 最终会走已经被 class-level compile 的：

```text
DenseDecoderLayer.forward
```

Qwen3.5 Dense 的 layer 类型是混合的：

位置：`xtuner/v1/model/dense/qwen3_5_text.py`

```python
return [
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
    for i in range(self.num_hidden_layers)
]
```

也就是：

```text
每 4 层一个 full_attention
其他层是 linear_attention / GatedDeltaNet
```

但默认 `DENSE_COMPILE_CFG` 仍然统一写：

```text
DenseDecoderLayer.forward fullgraph=True
```

如果后续开启 recompute，Dense 的 FSDP 分支还会对 checkpoint wrapper 后的 `layer.forward` 再 compile。

这里要分清两层：

```text
模型构建期默认 compile:
  DenseDecoderLayer.forward fullgraph=True

recompute 后的 wrapper 外层 compile:
  checkpoint_wrapper(layer)
  layer.forward = torch.compile(wrapper.forward, fullgraph=...)
```

只有第二层，也就是 checkpoint wrapper 后的 `wrapper.forward`，才会按 layer 类型改 `fullgraph`：

```text
被 checkpoint wrapper 包住的 full_attention layer:
  wrapper.forward fullgraph=True

被 checkpoint wrapper 包住的 linear_attention layer:
  wrapper.forward fullgraph=False
```

原因前面已经说过：checkpoint wrapper + fullgraph 对写 `seq_ctx.seq_idx` 这类 side effect 更敏感。因此 linear attention 只是在 recompute wrapper 外层 compile 时用 `fullgraph=False`，不是说它的普通 `DenseDecoderLayer.forward` 默认 compile 也是 `fullgraph=False`。

## 33. Qwen3.5 Dense 的完整默认 compile 图

如果不考虑训练配置里手动 pop，也不考虑 recompute 后的 wrapper.forward 额外 compile，Qwen3.5 Dense 默认可以理解为：

```text
Qwen3_5_VLDense4BConfig
  compose 外层:
    Qwen3VLProjector.forward fullgraph=True
    Qwen3VLVisionLayer.forward fullgraph=True
    DEFAULT_FLOAT8_CFG

  text tower:
    DenseDecoderLayer.forward fullgraph=True
    DEFAULT_FLOAT8_CFG
```

最终 `self.compile_cfg` 会把 compose 外层和子 `BaseModel` 的 compile_cfg 合并。

因为 dict 合并时相同 key 会覆盖/去重，所以 `DEFAULT_FLOAT8_CFG` 即使在外层和 text tower 都出现，也只是相同目标的重复合并，不代表同一个 FP8 helper 会被编译两份。

## 34. Qwen3.5 MoE 的 text tower compile

Qwen3.5 MoE 配置是：

位置：`xtuner/v1/model/compose/qwen3_5/qwen3_5_config.py`

```python
class Qwen3_5_VLMoE35BA3Config(Qwen3_5_BaseConfig):
    vision_config = Qwen3_5_VisionConfig()
    projector_config = Qwen3_5_ProjectorConfig()
    text_config = Qwen3_5_VLTextMoE35BA3BConfig()
```

`Qwen3_5_VLTextMoE` 专门覆盖了 `default_compile_cfg`：

位置：`xtuner/v1/model/moe/qwen3_5_text.py`

```python
def default_compile_cfg(self):
    if self.config.ep_size > 1:
        return MOE_EP_COMPILE_CFG
    else:
        return MOE_NON_EP_COMPILE_CFG
```

也就是说，Qwen3.5 MoE 的 text compile 设置会根据 EP 开关变化。

## 35. Qwen3.5 MoE：非 EP 时的 compile 设置

非 EP，也就是：

```text
ep_size == 1
```

默认使用：

```python
MOE_NON_EP_COMPILE_CFG
```

内容是：

```text
MoEBlock.forward:
  fullgraph=True

MoEDecoderLayer.forward:
  fullgraph=False

MoEDecoderLayer._pre_moe_forward:
  fullgraph=False

MultiHeadAttention.forward:
  fullgraph=True

GatedDeltaNet.forward:
  fullgraph=True

MoEDecoderLayer._shared_experts_forward:
  fullgraph=True

MoEDecoderLayer._post_moe_forward:
  fullgraph=True

DenseDecoderLayer.forward:
  fullgraph=True

DEFAULT_FLOAT8_CFG:
  FP8 helper fullgraph=True
```

这里的策略不是把整个 MoE model forward 编译成一个大图，而是把 MoE decoder layer 拆成多个相对稳定的片段。

关键点有两个：

```text
1. MoEDecoderLayer.forward 用 fullgraph=False

2. _pre_moe_forward 也用 fullgraph=False
```

这是因为 Qwen3.5 MoE 的 layer 内部比 Dense 复杂很多：

```text
attention / linear attention
router
expert dispatch / combine
shared experts
aux loss 相关数据
可能的 EP 通信
```

所以整层 forward 不强求 fullgraph。

相对稳定的计算片段则尽量 fullgraph：

```text
MoEBlock.forward
MHA.forward
GatedDeltaNet.forward
shared experts forward
post moe forward
DenseDecoderLayer.forward
```



## 36. Qwen3.5 MoE：EP 时的 compile 设置

EP 开启时：

```text
ep_size > 1
```

配置变成：

```python
MOE_EP_COMPILE_CFG = MOE_NON_EP_COMPILE_CFG.copy()
MOE_EP_COMPILE_CFG.pop(
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward"
)
```

也就是 EP 下会去掉：

```text
MoEDecoderLayer.forward
```

这个 compile target。

保留的仍然是内部片段：

```text
MoEBlock.forward fullgraph=True
MoEDecoderLayer._pre_moe_forward fullgraph=False
MultiHeadAttention.forward fullgraph=True
GatedDeltaNet.forward fullgraph=True
MoEDecoderLayer._shared_experts_forward fullgraph=True
MoEDecoderLayer._post_moe_forward fullgraph=True
DenseDecoderLayer.forward fullgraph=True
DEFAULT_FLOAT8_CFG
```

这样做的直觉是：

```text
EP 下 MoEDecoderLayer.forward 会包含更多通信 / dispatch 编排。
整层 forward 更不适合被 compile 成一个整体。

所以 EP 下不 compile 整层 MoEDecoderLayer.forward，
只 compile 内部相对稳定的计算函数。
```

这和前面 MoE recompute 没有额外 compile checkpoint wrapper.forward 的思路是一致的：

```text
MoE 更倾向于编译稳定片段，而不是包住整层复杂调度。
```



## 37. Qwen3.5 MoE 的完整默认 compile 图

如果不考虑训练配置里的手动覆盖，Qwen3.5 MoE 默认可以理解为：

```text
Qwen3_5_VLMoE35BA3Config
  compose 外层:
    Qwen3VLProjector.forward fullgraph=True
    Qwen3VLVisionLayer.forward fullgraph=True
    DEFAULT_FLOAT8_CFG

  text tower, ep_size == 1:
    MoEBlock.forward fullgraph=True
    MoEDecoderLayer.forward fullgraph=False
    MoEDecoderLayer._pre_moe_forward fullgraph=False
    MultiHeadAttention.forward fullgraph=True
    GatedDeltaNet.forward fullgraph=True
    MoEDecoderLayer._shared_experts_forward fullgraph=True
    MoEDecoderLayer._post_moe_forward fullgraph=True
    DenseDecoderLayer.forward fullgraph=True
    DEFAULT_FLOAT8_CFG

  text tower, ep_size > 1:
    MoEBlock.forward fullgraph=True
    MoEDecoderLayer._pre_moe_forward fullgraph=False
    MultiHeadAttention.forward fullgraph=True
    GatedDeltaNet.forward fullgraph=True
    MoEDecoderLayer._shared_experts_forward fullgraph=True
    MoEDecoderLayer._post_moe_forward fullgraph=True
    DenseDecoderLayer.forward fullgraph=True
    DEFAULT_FLOAT8_CFG
```



## 38. Dense 和 MoE 的差异总结

Dense 的策略：

```text
decoder layer 结构比较规整。
默认直接 compile DenseDecoderLayer.forward fullgraph=True。
recompute 后还会额外 compile checkpoint wrapper.forward。
```

MoE 的策略：

```text
整层 MoEDecoderLayer.forward 复杂。
非 EP 下只用 fullgraph=False。
EP 下直接不 compile 整层 MoEDecoderLayer.forward。
更多依赖内部稳定片段 compile。
recompute 后不额外 compile checkpoint wrapper.forward。
```

vision / projector 的策略：

```text
Qwen3VLProjector.forward fullgraph=True
Qwen3VLVisionLayer.forward fullgraph=True
```

但具体训练配置可能因为稳定性或收益问题手动移除 vision layer compile。

## 39. 补充：_mark_dynamic 是为了提前声明 cu_seq_lens 的动态维度

BaseModel 里还有一个和 compile 相关的小函数：

位置：`xtuner/v1/model/base.py`

```python
def _mark_dynamic(self, seq_ctx: SequenceContext, dim=0):
    """`cu_seq_lens_q` and `cu_seq_lens_k` are dynamic shapes in each
    fwd/bwd pass.

    Mark them as dynamic explicitly to avoid recompilation.
    """
    torch._dynamo.mark_dynamic(seq_ctx.cu_seq_lens_q, dim)
    torch._dynamo.mark_dynamic(seq_ctx.cu_seq_lens_k, dim)
```

它标记的是：

```text
seq_ctx.cu_seq_lens_q.shape[0]
seq_ctx.cu_seq_lens_k.shape[0]
```

`cu_seq_lens` 是 varlen / packed attention 用的累计长度。比如一个 packed batch 里有 3 条样本：

```text
seq_lens = [5, 7, 4]
cu_seq_lens = [0, 5, 12, 16]
```

下一步如果 packed 进去 5 条样本：

```text
seq_lens = [3, 8, 2, 6, 5]
cu_seq_lens = [0, 3, 11, 13, 19, 24]
```

那么 `cu_seq_lens.shape[0]` 会变化。

`mark_dynamic` 的作用是提前告诉 Dynamo：

```text
这个维度会变。
不要把它固定成第一次看到的 shape。
请按动态 shape 处理。
```

## 40. 没有 mark_dynamic 是否一定会反复编译

这里要说准确一点：

```text
不一定。
```

`torch.compile` 的默认参数是：

```python
dynamic=None
```

当前 PyTorch 文档里的语义是：

```text
默认先按静态 shape 编译。
如果后续发现 shape 变化导致 guard failure，
Dynamo 会尝试在 recompile 时生成更动态的 kernel。
```

所以没有 `_mark_dynamic` 时，确实可能出现这种情况：

```text
第 1 次:
  用 cu_seq_lens.shape[0] = 9 编一个静态图

第 2 次:
  发现 cu_seq_lens.shape[0] = 13，guard fail
  Dynamo 重新编译，并尝试把这个维度动态化

第 3 次:
  如果新的动态图覆盖了 shape[0] = 7 / 11 / 15 ...
  就不再因为这个维度变化而重新编译
```

也就是说，你说的情况是可能的：

```text
第二次发现变了之后，Dynamo 自动泛化。
后面不再因为同一个维度变化继续编译。
```

但这不是绝对保证，原因包括：

```text
1. 不是所有动态 shape 都能成功泛化。
2. 某些 op / backend 优化可能强制 specialization。
3. 可能同时有多个动态维度或 Python 分支 guard。
4. fullgraph / checkpoint / custom op 场景下更容易有额外限制。
```

所以 `_mark_dynamic` 的价值不是“没有它就必然每种 shape 编一次”，而是：

```text
提前告诉 Dynamo 这个维度动态。
避免先编一个很快失效的静态图。
减少前几步的 guard failure / recompile。
让 packed varlen batch 更稳定地走动态 shape 路径。
```

可以理解成：

```text
没有 mark_dynamic:
  依赖 Dynamo 自动发现动态性。
  可能先静态编译，再因为 shape 变化重编一次或多次。

有 mark_dynamic:
  从第一次 trace 就声明这个维度动态。
  减少 trial-and-error 式的重编译。
```
