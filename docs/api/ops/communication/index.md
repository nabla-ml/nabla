# Distributed Communication

## `shard`

```python
def shard(x, mesh: 'DeviceMesh', dim_specs: 'list[DimSpec]', replicated_axes: 'set[str] | None' = None, **kwargs):
```
Shard a tensor according to the given mesh and dimension specs.

This operation is "smart":
1. If the input is already sharded differently, it inserts necessary
   communication (AllGather, AllReduce) to transition to the valid state.
2. Then it applies the physical slicing (ShardOp) to reach the target distribution.


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
Sum-reduce across all shards.

Note: MAX only supports sum reduction natively.


---
## `all_gather`

```python
def all_gather(sharded_tensor, axis: 'int' = None, **kwargs):
```
Gather all shards to produce a replicated tensor.


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

 – Tensor on the target device or with target sharding

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

 – Tensor on the target device or with target sharding

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

 – Tensor on CPU


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

 – Tensor on GPU/Accelerator


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

 – Tensor on specified accelerator


---
