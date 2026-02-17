# Distributed

## `shard_map`

```python
def shard_map(func: 'Callable[..., Any]', mesh: 'DeviceMesh', in_specs: 'dict[int, ShardingSpec]', out_specs: 'dict[int, ShardingSpec] | None' = None, auto_sharding: 'bool' = False, debug: 'bool' = False) -> 'Callable[..., Any]':
```
Execute a function with automatic sharding propagation and execution.


---
