# 中心化内存

以共卡为例，在一次完整训练 step 中

- replay buffer 本身就是一个中心化内存，所有数据除了超大对象，都存在内存中，不释放。train_batch = produce_result.rollout_states 中是全量的
- 在 data_batches, —— = self._prepare_train_data() 后又会额外多一份稍微小一点的全局内存
- 在 train controller 中 packed_data_batches = self._packing() 这个对象又是额外一份全局内存
- 然后将这些对象通过序列化传给每个 worker，这个地方可能也存在一份全量的 packed_data_batches 全局内存

因此从夸张角度估计，会同时存在 4 份全局内存在 head 节点上。虽然内部没有超大对象，但是如果序列非常长，logprob+input id+其他对象，也可能占比较多。

# 序列化次数

目前 rollout controller 还是一个中心化 ray，虽然内存问题可能不是很严重，但是序列化开销也是需要考虑的，特别是后续改成 http 服务则问题比较多。

- async def generate(self, rollout_state: RolloutState) 这个不合理。虽然内部的 worker 是 n 个，但是所有数据始终要经过他转发。最合理应该单独写一个 router，不是一个 ray 对象即可。
而且 agentloop 是 m 个时候，agentloop 调用这个方法会涉及到跨节点的，明显可以节省这一层开销。 后面切换为 http 模式则更严重。
- _apply_output_parsers 这个后处理时间不知道咋样？目前是纯同步调用。

# rollout controller 或者其他 ray actor 没有指定 cpu

```python
return (
            ray.remote(RolloutController)
            .options(max_concurrency=int(os.environ.get("RAY_MAX_CONCURRENCY", 1000)))
            .remote(self, placement_group)
        )
```
虽然这没有啥大问题，但是不符合 ray 规范，建议写 num_cpu=1

# slime 设计

他在 ray actor 的 rolloutmanager 里面做了太多事情，一旦调用 generate 生成，adv 计算，后内部会拉取数据，调用 asycn def 生成，然后拿到中心化所有数据后，先进行 batch 级别的
balance 进行打乱，然后按照 dp size 个数切分为 dp 份，然后转为 ray 发送给 train，所有并没有中间的这层中心化。也就是说 slime 只有一层全局中心化。

# agentloop 开启 ray actor 时候并没有强制分散，因为目前没有用 pg


