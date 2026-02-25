# Distributed Transforms

## `shard_map`

```python
def shard_map(func: 'Callable[..., Any]', mesh: 'DeviceMesh', in_specs: 'dict[int, ShardingSpec]', out_specs: 'dict[int, ShardingSpec] | None' = None, auto_sharding: 'bool' = False, debug: 'bool' = False) -> 'Callable[..., Any]':
```
Execute a function with explicit or automatic SPMD sharding.

Applies *in_specs* to shard the inputs across the *mesh*, traces
*func* to capture the computation graph, executes the sharded
forward pass, and optionally reshards the outputs according to
*out_specs*.

**Parameters**

- **`func`** – Function to distribute. It is traced once and executed with
sharded tensors.
- **`mesh`** – Device mesh describing the multi-device topology.
- **`in_specs`** – Mapping from argument index to :class:`ShardingSpec`.
Specifies how each input should be sharded.
- **`out_specs`** – Optional mapping from output index to
:class:`ShardingSpec`. If ``None``, output sharding is
inferred from the computation graph.
- **`auto_sharding`** – If ``True``, run the ILP auto-sharding solver to
determine optimal intermediate shardings.
- **`debug`** – If ``True``, print sharding decisions at each op.

**Returns**

A wrapped function with the same signature as *func*.


---
