# Communication

## `shard`

```python
def shard(x, mesh: 'DeviceMesh', dim_specs: 'list[DimSpec]', replicated_axes: 'set[str] | None' = None, **kwargs):
```
Shard a tensor across a device mesh according to the given dimension specs.

This is the primary API for distributing a tensor across devices. It
handles transitions between different sharding layouts transparently:

1. If the input is already sharded in a different way, collects via
   ``all_gather`` or ``all_reduce`` as necessary before reslicing.
2. Applies physical slicing (``ShardOp``) to reach the target distribution.

**Parameters**

- **`x`** – Input tensor or pytree of tensors to shard.
- **`mesh`** – Target :class:`DeviceMesh`.
- **`dim_specs`** – Per-dimension sharding specification, one :class:`DimSpec`
per logical dimension.
- **`replicated_axes`** – Set of mesh axis names that are replicated (not
sharded). Default: ``None`` (no replicated axes).
- **`**kwargs`** – Additional keyword arguments forwarded to the underlying op.

**Returns**

Sharded tensor distributed as specified by *dim_specs* on *mesh*.


---
## `reshard`

```python
def reshard(*args, **kwargs):
```
Deprecated: Use ops.shard instead.

This function delegates strictly to ops.shard, which now handles
smart resharding transitions (AllGather/AllReduce) automatically.


---
## `all_reduce`

```python
def all_reduce(sharded_tensor, **kwargs):
```
All-reduce a sharded tensor across all shards.

Each shard applies a commutative reduction (default: sum) over the
values held on all participating devices and replaces its local value
with the global result, so every shard ends up with the same value.

**Parameters**

- **`sharded_tensor`** – Sharded input tensor.
- **`**kwargs`** – Optional keyword args forwarded to the backend, including
``reduce_op`` (``'sum'``, ``'max'``, ``'min'``, ``'prod'``) and
``reduce_axes`` to restrict reduction to a subset of mesh axes.

**Returns**

Tensor with shard-local values replaced by the global reduction result.


---
## `all_gather`

```python
def all_gather(sharded_tensor, axis: 'int' = None, **kwargs):
```
Gather shards along *axis* to produce a locally-replicated full tensor.

Each device receives a copy of the full concatenated tensor.

**Parameters**

- **`sharded_tensor`** – Sharded input tensor, distributed along *axis*.
- **`axis`** – Logical axis along which to gather. If ``None``, the sharding
metadata is used to determine the gather dimension.
- **`**kwargs`** – Additional keyword arguments forwarded to the backend
(e.g., ``physical_axis``).

**Returns**

Tensor with the same shape as the global (unsharded) tensor,
replicated on every device.


---
## `all_to_all`

```python
def all_to_all(sharded_tensor, split_axis: 'int', concat_axis: 'int', tiled: 'bool' = True):
```
All-to-all collective (distributed transpose).


---
## `reduce_scatter`

```python
def reduce_scatter(sharded_tensor, axis: 'int', **kwargs):
```
Sum-reduce then scatter result across shards.

Note: MAX only supports sum reduction natively.


---
## `distributed_broadcast`

```python
def distributed_broadcast(x, mesh=None):
```
Broadcast a tensor across a distributed mesh.


---
## `ppermute`

```python
def ppermute(sharded_tensor, permutation: 'list[tuple]'):
```
Point-to-point permutation collective.


---
## `to_device`

```python
def to_device(x: 'Tensor', device: 'Device | Any' = None, *, sharding: 'Any' = None) -> 'Tensor':
```
Transfer tensor to specified device or sharding.

Like JAX's device_put, this function supports both:
- Single device transfer: `to_device(x, CPU())`
- Multi-device sharding: `to_device(x, sharding=my_sharding_spec)`

This operation is differentiable - gradients flow through the transfer.
On the forward pass, data is moved/distributed.
On the backward pass, gradients are transferred back to the input's layout.

**Parameters**

- **`x`** – Input tensor
- **`device`** – Target device (Device object). If None and sharding is None,
returns x as-is if already on a device, otherwise moves to default device.
- **`sharding`** – Target ShardingSpec for multi-device distribution (mutually exclusive with device)

**Returns**

Tensor on the target device or with target sharding

**Examples**

*Warning: Could not parse examples correctly.*

---
## `transfer_to`

```python
def transfer_to(x: 'Tensor', device: 'Device | Any' = None, *, sharding: 'Any' = None) -> 'Tensor':
```
Transfer tensor to specified device or sharding.

Like JAX's device_put, this function supports both:
- Single device transfer: `to_device(x, CPU())`
- Multi-device sharding: `to_device(x, sharding=my_sharding_spec)`

This operation is differentiable - gradients flow through the transfer.
On the forward pass, data is moved/distributed.
On the backward pass, gradients are transferred back to the input's layout.

**Parameters**

- **`x`** – Input tensor
- **`device`** – Target device (Device object). If None and sharding is None,
returns x as-is if already on a device, otherwise moves to default device.
- **`sharding`** – Target ShardingSpec for multi-device distribution (mutually exclusive with device)

**Returns**

Tensor on the target device or with target sharding

**Examples**

*Warning: Could not parse examples correctly.*

---
## `cpu`

```python
def cpu(x: 'Tensor') -> 'Tensor':
```
Transfer tensor to CPU.

Convenience function for `to_device(x, CPU())`.

**Parameters**

- **`x`** – Input tensor

**Returns**

Tensor on CPU


---
## `gpu`

```python
def gpu(x: 'Tensor') -> 'Tensor':
```
Transfer tensor to GPU/Accelerator.

Convenience function for `to_device(x, Accelerator())`.

**Parameters**

- **`x`** – Input tensor

**Returns**

Tensor on GPU/Accelerator


---
## `accelerator`

```python
def accelerator(x: 'Tensor', device_id: 'int' = 0) -> 'Tensor':
```
Transfer tensor to specific accelerator device.

**Parameters**

- **`x`** – Input tensor
- **`device_id`** – Accelerator device ID (default: 0)

**Returns**

Tensor on specified accelerator


---
