# Torch Compile 学习笔记

本文档用于持续记录 `torch.compile` 学习过程中的问题、实验结论和源码入口。当前内容基于本机 conda 环境里的 PyTorch 源码分析：

- Python: `/mnt/shared-storage-user/huanghaian/miniconda3/envs/pt29_all_env/bin/python`
- PyTorch: `2.9.1+cu128`
- Torch package: `/mnt/shared-storage-user/huanghaian/miniconda3/envs/pt29_all_env/lib/python3.12/site-packages/torch`

## 1. 重编译的核心机制

`torch.compile` 前端主要由 TorchDynamo 负责。Dynamo 会把 Python frame 编译成带 guard 的 cache entry。

关键源码：

- `torch/_dynamo/cache_size.py`
- `torch/_dynamo/convert_frame.py`
- `torch/_dynamo/guards.py`
- `torch/_dynamo/config.py`

核心逻辑：

1. 每个 Python frame 的 compiled cache 是一个链表。
2. 每个 cache entry 包含 `guard_manager` 和编译后的代码。
3. 下次调用同一个 frame 时，Dynamo 会遍历 cache entry，并检查 guard。
4. 如果某个 entry 的 guard 通过，就复用这个 entry。
5. 如果没有任何 entry 的 guard 通过，就触发重新编译，并追加新的 cache entry。

所以，“是否重编译”的判断标准不是“输入有没有变化”，而是：

> 变化后的输入和运行时状态，是否还能通过已有 compiled graph 的 guard。



## 2. PyTorch 2.9.1 默认重编译相关配置

在当前环境的 `torch/_dynamo/config.py` 里，关键默认值是：

```python
recompile_limit = 8
accumulated_recompile_limit = 256
dynamic_shapes = True
assume_static_by_default = True
automatic_dynamic_shapes = True
```

需要注意：`dynamic_shapes=True` 不等于默认第一次就把所有 shape 都当成完全动态。

在 PyTorch 2.9.1 当前默认策略下：

1. 首次编译通常偏静态。
2. 如果后续某些 tensor shape 维度发生变化，shape guard 失败。
3. Dynamo 会触发一次重编译。
4. 因为 `automatic_dynamic_shapes=True`，这次重编译会尝试把“晃动过”的维度提升成动态维度。
5. 后续同类 shape 变化通常可以复用动态 graph。



## 3. Tensor shape 变化会不会一直重编译

不一定。

### 3.1 默认 `dynamic=None`

例子：

```python
@torch.compile
def f(x):
    return x.sin() + 1

f(torch.randn(2, 4))  # compile #1，首个 shape
f(torch.randn(3, 4))  # compile #2，batch 维变化，触发动态化
f(torch.randn(5, 4))  # 通常复用 compile #2
```

结论：

- 同一个 rank 下，某个维度 size 变化，默认通常是“先静态编一次，再因 shape 变化重编一次”。
- 后续同一类 shape 变化，通常不再反复编译。



### 3.2 显式 `dynamic=False`

```python
compiled_f = torch.compile(f, dynamic=False)
```

这种情况下 shape 更容易被固定住：

```python
f(torch.randn(2, 4))  # compile #1
f(torch.randn(3, 4))  # compile #2
f(torch.randn(5, 4))  # compile #3
```

如果 shape 一直变，可能不断产生新的 cache entry，直到碰到重编译限制。

### 3.3 显式 `dynamic=True`

```python
compiled_f = torch.compile(f, dynamic=True)
```

这种情况下 Dynamo 会更积极地使用动态 shape。对于 shape 经常变化的 workload，通常能减少默认模式下“首编静态 + 第二次动态化”的额外编译。

但 `dynamic=True` 不是万能的。rank 变化、dtype 变化、stride/layout 变化、Python 分支变化等仍可能导致不同 graph。

## 4. Tensor rank 从 2D 变 3D 会编译几次

动态 shape 通常动态的是某个维度的 size，不是 tensor 的 rank。

所以：

- `(B, H)` 和 `(B, H, W)` 通常不能共用同一个 compiled graph。
- 2D 和 3D 会分别有自己的 cache entry。

默认 `dynamic=None` 下，一个典型过程是：

```python
f(torch.randn(2, 4))       # compile #1: 2D 静态
f(torch.randn(2, 4, 8))    # compile #2: 3D 静态
f(torch.randn(5, 4))       # compile #3: 2D 动态化
f(torch.randn(3, 4, 8))    # compile #4: 3D 动态化

f(torch.randn(7, 4))       # 复用 2D 动态 graph
f(torch.randn(9, 4, 8))    # 复用 3D 动态 graph
```

大致结论：

- 固定一个 2D shape 和固定一个 3D shape 来回切换：通常 2 次编译。
- 2D 内 shape 也变，3D 内 shape 也变：默认常见是 4 次左右。
- `dynamic=True`：通常每个 rank 一套 graph，也就是 2D 一次、3D 一次。
- `dynamic=False`：每个不同 concrete shape 都可能编一次。



## 5. 除 shape 外，Tensor 还有哪些变化会触发重编译

Tensor 的 guard 不只看 shape，还会看 metadata。常见会导致 guard fail 的变化包括：

- `shape`
- `rank`
- `stride`
- `dtype`
- `device`
- `layout`
- `requires_grad`
- `storage_offset`
- contiguous / non-contiguous 差异

例如：

```python
@torch.compile
def f(x):
    return x + 1

f(torch.randn(2, 4, dtype=torch.float32))  # compile #1
f(torch.randn(2, 4, dtype=torch.float64))  # compile #2
```

`requires_grad` 变化也可能触发：

```python
f(torch.randn(2, 4, requires_grad=False))  # compile #1
f(torch.randn(2, 4, requires_grad=True))   # compile #2
```



## 6. Python 对象和值变化会不会触发重编译

不是“只要 Python 对象值变了就一定重编译”。

更准确的说法是：

> 只要这个 Python 对象的某个值、属性、结构或身份被 Dynamo 用来生成 guard，并且下次调用 guard 不满足，就会触发重编译。



### 6.1 Python bool 分支

```python
@torch.compile
def f(x, flag):
    if flag:
        return x * 2
    return x + 2

f(x, True)   # compile #1
f(x, False)  # compile #2
f(x, True)   # 复用 compile #1
```

这里 `flag` 参与 Python 控制流，所以会被 guard 住。

### 6.2 Python scalar

```python
@torch.compile
def f(x, scale):
    return x * scale
```

如果 `scale` 是 Python int / float，它可能作为 Python 值被 specialization。`scale` 变化时可能触发重编译。

如果想减少这类重编译，可以考虑把频繁变化的 scalar 表达成 tensor，或让它进入更适合动态处理的路径，但要结合具体代码判断。

### 6.3 容器结构

```python
@torch.compile
def f(xs):
    return sum(xs)

f([x, y])     # compile #1
f([x, y, z])  # compile #2
```

list / tuple / dict 常见会 guard：

- 类型
- 长度
- key 集合
- 元素结构

所以容器结构变了很容易触发重编译。

### 6.4 普通 Python 对象属性

```python
class C:
    def __init__(self, mode):
        self.mode = mode

@torch.compile
def f(x, cfg):
    if cfg.mode == "a":
        return x * 2
    return x + 2
```

如果 `cfg.mode` 参与分支，`cfg.mode` 变化通常会触发 guard fail 和重编译。

但如果对象没有被使用，或者它变化的部分没有被 Dynamo 读取，那么不一定重编译：

```python
@torch.compile
def f(x, obj):
    return x + 1
```

这里 `obj` 对输出没有影响，通常不会因为 `obj` 内部状态变化而重编译。

## 7. 如何诊断重编译原因

最直接的方式是打开 recompiles 日志：

```bash
TORCH_LOGS=recompiles python your_script.py
```

典型输出：

```text
Recompiling function f in ...
    triggered by the following guard failure(s):
    - tensor 'x' size mismatch at index 0. expected 2, actual 3
```

Python 值导致的重编译可能看到：

```text
flag == True
len(L['xs']) == 2
L['cfg'].mode == 'a'
```

如果需要更详细的 Dynamo trace，可以使用：

```bash
TORCH_LOGS="+dynamo" python your_script.py
```

不过 `+dynamo` 日志很大，日常排查重编译优先用 `TORCH_LOGS=recompiles`。

## 8. 如何用 custom backend 观察编译次数

可以用一个简单 backend 统计 Dynamo 编译次数，避免默认 Inductor 编译耗时干扰：

```python
import torch
import torch._dynamo as dynamo

dynamo.reset()
counter = {"compile": 0}

def backend(gm, example_inputs):
    counter["compile"] += 1
    print("compile", counter["compile"])
    for x in example_inputs:
        if isinstance(x, torch.Tensor):
            print("  tensor", tuple(x.shape), x.dtype, x.requires_grad, x.stride())
        else:
            print("  non_tensor", type(x).__name__, x)
    return gm.forward

@torch.compile(backend=backend)
def f(x):
    return x.sin() + 1

f(torch.randn(2, 4))
f(torch.randn(3, 4))
f(torch.randn(5, 4))
```

如果第二次编译时看到 `SymInt` 出现在 `example_inputs` 里，通常说明 Dynamo 已经把某个维度动态化了。

## 9. 模块不支持 full 编译时怎么办

如果某段逻辑 Dynamo 不支持 trace，常见有两类处理方式：

1. 允许 graph break，把这段逻辑放到 compiled graph 外面执行。
2. 包装成 custom op，让 `torch.compile` 把它当成图里的黑盒 operator。

这两种方式适用场景不同。

### 9.1 不要求 `fullgraph=True`：优先 graph break / disable

如果目标只是“这段不编译，前后继续编译”，可以用：

```python
@torch.compiler.disable
def unsupported_fn(x):
    # Dynamo 不会 trace 这里
    return x

@torch.compile
def f(x):
    y = x.sin()
    z = unsupported_fn(y)
    return z + 1
```

这种方式简单，但会产生 graph break。

因此它不适合 `torch.compile(fullgraph=True)`，因为 `fullgraph=True` 要求整个函数被捕获成一张完整图，不允许中间断开。

### 9.2 要求 `fullgraph=True`：考虑 custom op

如果某段逻辑内部不支持 Dynamo trace，但希望外层仍然能 `fullgraph=True`，可以把这段逻辑包装成 `torch.library.custom_op`。

示例：

```python
import torch

@torch.library.custom_op("mylib::my_op", mutates_args=())
def my_op(x: torch.Tensor) -> torch.Tensor:
    # 内部可以是 Dynamo 不支持的 Python、第三方库、自定义 kernel 等
    return x

@my_op.register_fake
def _(x):
    # fake/meta 实现：告诉编译器输出 tensor 的 metadata
    return torch.empty_like(x)

@torch.compile(fullgraph=True)
def f(x):
    y = torch.sin(x)
    z = my_op(y)
    return z + 1
```

这里 Dynamo 不会 trace `my_op` 的 Python 内部实现，而是在 FX graph 里放一个 `mylib::my_op` 节点。编译期使用 `register_fake` 里的 fake 实现推导输出 shape / dtype / device 等 metadata；运行期再调用真实实现。

所以 custom op 可以理解为：

> 让 `torch.compile` 看到一个稳定的 operator 边界，而不是继续深入内部 Python 逻辑。

### 9.3 custom op 不是只包一层函数就完事

custom op 需要维护 operator 契约：

- `mutates_args` 必须准确，否则 mutation / alias 行为可能错。
- 通常需要 `register_fake`，否则 FakeTensor 编译阶段不知道输出 metadata。
- 如果参与训练，需要注册 autograd / backward。
- 如果输出 shape 是数据依赖的动态 shape，fake 实现也要正确表达这种动态关系。
- 如果内部只是 Python 实现，它可以作为黑盒进入图，但内部不会被 Inductor 融合优化。
- 如果要性能，custom op 背后最好是真正的 C++ / CUDA / Triton kernel，或稳定的 dispatcher op。

### 9.4 `allow_in_graph` 和 custom op 的区别

还有一个相关工具是：

```python
torch.compiler.allow_in_graph(fn)
```

它也可以让 Dynamo 不 trace 某个 Python 函数内部，而是把函数调用放进图里。

但它更像“告诉 Dynamo 这个函数可以作为 graph 节点出现”，后续 FX / AOTAutograd / Inductor 仍然需要能处理它。长期维护的工程代码里，如果这是一个稳定边界，通常更推荐正式注册 `torch.library.custom_op`，因为 schema、fake、mutation、autograd 契约更清晰。

### 9.5 经验判断

- 不要求完整图：优先 `torch.compiler.disable` / graph break，成本最低。
- 要 `fullgraph=True`：优先考虑 `torch.library.custom_op`。
- 要训练：custom op 还要补 backward / autograd。
- 要性能：custom op 只是黑盒边界，黑盒内部不会被 compile 自动优化。
- 不要把 custom op 当成万能逃生口；它解决的是“编译器如何看待边界”，不是自动优化内部实现。

## 10. 目前形成的经验规则

1. shape 一直变，不一定一直重编译；默认 2.9.1 通常会先静态再动态化。
2. rank 变化通常需要不同 graph。
3. `dynamic=True` 能减少 shape 抖动造成的额外编译，但不能消除所有 specialization。
4. `dynamic=False` 会显著增加 shape 变化导致的重编译概率。
5. Python bool / 分支 / 容器结构 / 对象属性变化，只要进入 guard，就可能触发重编译。
6. 不要猜重编译原因，优先看 `TORCH_LOGS=recompiles`。
7. 模块不支持 full 编译时，先判断是否允许 graph break；如果必须 fullgraph，再考虑 custom op。
8. custom op 会让 compile 把内部当黑盒，但必须提供正确的 fake/meta、mutation、autograd 等契约。



## 11. 后续待补充问题

- Inductor 层面的 kernel cache 和 Dynamo 重编译之间的区别。
- `torch._dynamo.mark_dynamic` / `maybe_mark_dynamic` 的实际用法。
- `torch.compile` 在训练场景中与 optimizer、autograd、DDP/FSDP 的交互。
- XTuner 代码中哪些输入或状态最容易造成重编译。
