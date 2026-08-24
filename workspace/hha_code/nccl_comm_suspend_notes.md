# NCCL Communicator Suspend/Resume 备忘

## 背景

NCCL 2.29.7 开始提供 communicator 级别的内存 offload 能力：

- `ncclCommSuspend(comm, NCCL_SUSPEND_MEM)`
- `ncclCommResume(comm)`
- 相关内存统计接口：`ncclCommMemStats`

这个能力不是全局释放进程里所有 NCCL 内存，而是作用在某一个具体的
`ncclComm_t` 上。因此谁创建并持有 communicator，谁就应该负责调用
suspend/resume。

## PyTorch 支持情况

PyTorch 从 2.12 开始支持 NCCL communicator suspend/resume。相关 PR：

- pytorch/pytorch#176300: Add NCCL comm suspend, resume and memory stats
- pytorch/pytorch#180547: 对 NCCL < 2.29.7 的环境跳过相关测试

PyTorch 侧暴露的是 backend 级别接口：

```python
backend = pg._get_backend(torch.device("cuda:0"))
backend.suspend()
backend.resume()
stats = backend.memory_stats()
```

底层实现是在 `ProcessGroupNCCL` / `NCCLComm` 中调用：

```cpp
ncclCommSuspend(comm, NCCL_SUSPEND_MEM);
ncclCommResume(comm);
```

要求：

- PyTorch 版本需要支持该接口，已知从 torch 2.12 开始支持。
- 编译和运行时 NCCL 都需要满足 2.29.7+。
- 如果运行时加载到旧 NCCL，可能出现 `undefined symbol: ncclCommResume`
  这类动态链接问题。

## PyNCCL 和 torch.distributed 不能混用管理

PyNCCL 和 PyTorch 即使用的是同一个 `libnccl.so`，它们创建的
communicator 也是不同的 `ncclComm_t`。

因此：

- `pynccl_comm.suspend()` 只能释放 PyNCCL 自己创建的 communicator 内存。
- `torch.distributed.all_reduce()` 使用的是 PyTorch `ProcessGroupNCCL`
  内部的 communicator，不能靠 PyNCCL 去 suspend。
- 对 torch 训练、DDP、FSDP 等路径，应使用 PyTorch 自己的 backend
  suspend/resume。
- 对 vLLM 自己创建的 PyNCCL communicator，可以由 vLLM 调
  `pynccl_comm.suspend()` / `resume()`。

核心原则：communicator 的 owner 负责生命周期和状态转换，不要跨框架
偷偷 suspend/resume 另一个框架内部持有的 `ncclComm_t`。

## 使用注意

- `ncclCommSuspend` / `ncclCommResume` 是 collective 语义，相关 rank 必须
  按一致顺序调用。
- suspend 前必须确保 communicator idle，没有未完成 collective。
- resume 后再继续执行 collective。
- 如果一个 process group 内部缓存了多个 device/communicator，需要确认
  PyTorch API 实际 suspend 的范围是否覆盖目标 communicator。

## 和 vLLM PR #46234 的关系

vLLM PR #46234 的目标是 sleep mode 时释放 vLLM 自己的 PyNCCL
communicator 内存。它不负责释放 PyTorch `ProcessGroupNCCL` 内部的
communicator。

如果后续要在 vLLM 里释放 torch.distributed communicator，需要显式接入
PyTorch backend 的 `suspend()` / `resume()`，不能依赖 PyNCCL 混用。
