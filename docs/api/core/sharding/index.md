# Sharding

## `DeviceMesh`

```python
class DeviceMesh(name: str, shape: tuple[int, ...], axis_names: tuple[str, ...], devices: list[int] | None = None, device_refs: list['DeviceRef'] | None = None, bandwidth: float = 1.0) -> None:
```
Logical multi-dimensional view of devices: @name = <["axis1"=size1, ...]>.

**Parameters**

- **`name`** – Name of the mesh.
- **`shape`** – Shape of the mesh (e.g., (2, 4)).
- **`axis_names`** – Names for each axis (e.g., ("x", "y")).
- **`devices`** – Logical device IDs.
- **`device_refs`** – Physical device references.


---
## `PartitionSpec`

```python
class PartitionSpec(*args):
```
JAX-compatible PartitionSpec.


---
## `P`

```python
class P(*args):
```
JAX-compatible PartitionSpec.


---
## `DimSpec`

```python
class DimSpec(axes: list[str] = <factory>, is_open: bool = False, priority: int = 0, partial: bool = False) -> None:
```
Per-dimension sharding specification.


---
## `ShardingSpec`

```python
class ShardingSpec(mesh: nabla.core.sharding.spec.DeviceMesh, dim_specs: list[nabla.core.sharding.spec.DimSpec] = <factory>, replicated_axes: set[str] = <factory>, partial_sum_axes: set[str] = <factory>) -> None:
```
Complete tensor sharding: sharding<@mesh, [dim_shardings], replicated={axes}>.


---
## `compute_local_shape`

```python
def compute_local_shape(global_shape: tuple[int, ...], sharding: nabla.core.sharding.spec.ShardingSpec, device_id: int) -> tuple[int, ...]:
```
Compute the local shard shape for a device.


---
## `get_num_shards`

```python
def get_num_shards(sharding: nabla.core.sharding.spec.ShardingSpec) -> int:
```
Get the total number of shards for this sharding spec.


---
