# Tensor

## `Tensor`

```python
class Tensor(*, buffers: 'driver.Buffer | None' = None, value: 'graph.BufferValue | graph.TensorValue | None' = None, impl: 'TensorImpl | None' = None, is_traced: 'bool' = False) -> 'None':
```
Multi-dimensional array with eager execution and automatic compilation.


### Methods

#### `abs`
```python
def abs(self):
```

#### `acos`
```python
def acos(self):
```

#### `argmax`
```python
def argmax(self, axis: 'int | None' = None, keepdims: 'bool' = False):
```

#### `argmin`
```python
def argmin(self, axis: 'int | None' = None, keepdims: 'bool' = False):
```

#### `atanh`
```python
def atanh(self):
```

#### `backward`
```python
def backward(self, gradient: 'Tensor | None' = None, retain_graph: 'bool' = False, create_graph: 'bool' = False) -> 'None':
```
Compute gradients of this tensor w.r.t. graph leaves (PyTorch style).

Populates .grad on all tensors with requires_grad=True that this tensor
depends on. All gradients are batch-realized for efficiency.

**Parameters**

- **`gradient`** – Gradient w.r.t. this tensor. Required for non-scalar tensors.
- **`retain_graph`** – Unused (maintained for PyTorch API compatibility).
- **`create_graph`** – If True, graph of the derivatives will be constructed,
allowing to compute higher order derivatives.


#### `broadcast_to`
```python
def broadcast_to(self, shape: 'ShapeLike') -> 'Tensor':
```

#### `cast`
```python
def cast(self, dtype: 'DType'):
```

#### `cos`
```python
def cos(self):
```

#### `cpu`
```python
def cpu(self) -> 'Tensor':
```
Move tensor to CPU, gathering shards if needed.

For sharded tensors, this first gathers all shards to a single device,
then transfers to CPU. For unsharded tensors, it returns self if already
on CPU, otherwise creates a new tensor on CPU.

**Returns**

 – Tensor on CPU with all data gathered.


#### `cuda`
```python
def cuda(self, device: 'int | str' = 0) -> 'Tensor':
```
Move tensor to GPU (shortcut for PyTorch users).


#### `cumsum`
```python
def cumsum(self, axis: 'int'):
```

#### `detach`
```python
def detach(self) -> 'Tensor':
```
Returns a new Tensor, detached from the current graph (PyTorch style).


#### `dim`
```python
def dim(self) -> 'int':
```
Alias for rank (PyTorch style).


#### `erf`
```python
def erf(self):
```

#### `exp`
```python
def exp(self):
```

#### `expand`
```python
def expand(self, *shape: 'int') -> 'Tensor':
```
Alias for broadcast_to (PyTorch style).


#### `flatten`
```python
def flatten(self, start_dim: 'int' = 0, end_dim: 'int' = -1) -> 'Tensor':
```

#### `flip`
```python
def flip(self, axis: 'int | tuple[int, ...]') -> 'Tensor':
```

#### `floor`
```python
def floor(self):
```

#### `gather`
```python
def gather(self) -> 'Tensor':
```
Gather shards into a single global tensor if needed (lazy).


#### `gelu`
```python
def gelu(self, approximate: 'str | bool' = 'none'):
```

#### `hydrate`
```python
def hydrate(self) -> 'Tensor':
```
Populate graph values from buffers for realized tensors.

If the tensor is already registered as a graph input, uses that.
In EAGER_MAX_GRAPH mode, adds buffer data as a constant for intermediate
tensors accessed during eager graph building.


#### `is_inf`
```python
def is_inf(self):
```

#### `is_nan`
```python
def is_nan(self):
```

#### `item`
```python
def item(self) -> 'float | int | bool':
```

#### `log`
```python
def log(self):
```

#### `log1p`
```python
def log1p(self):
```

#### `logical_local_shape`
```python
def logical_local_shape(self, shard_idx: 'int' = 0) -> 'graph.Shape | None':
```

#### `logsoftmax`
```python
def logsoftmax(self, axis: 'int' = -1):
```

#### `max`
```python
def max(self, axis: 'int | tuple[int, ...] | None' = None, keepdims: 'bool' = False):
```

#### `mean`
```python
def mean(self, axis: 'int | tuple[int, ...] | None' = None, keepdims: 'bool' = False):
```

#### `min`
```python
def min(self, axis: 'int | tuple[int, ...] | None' = None, keepdims: 'bool' = False):
```

#### `new_empty`
```python
def new_empty(self, shape: 'ShapeLike', *, dtype: 'DType | None' = None, device: 'Device | None' = None) -> 'Tensor':
```
Create a new uninitialized tensor (defaults to zeros in Nabla).


#### `new_full`
```python
def new_full(self, shape: 'ShapeLike', fill_value: 'Number', *, dtype: 'DType | None' = None, device: 'Device | None' = None) -> 'Tensor':
```
Create a new tensor filled with value with same device/dtype as self by default.


#### `new_ones`
```python
def new_ones(self, shape: 'ShapeLike', *, dtype: 'DType | None' = None, device: 'Device | None' = None) -> 'Tensor':
```
Create a new tensor of ones with same device/dtype as self by default.


#### `new_zeros`
```python
def new_zeros(self, shape: 'ShapeLike', *, dtype: 'DType | None' = None, device: 'Device | None' = None) -> 'Tensor':
```
Create a new tensor of zeros with same device/dtype as self by default.


#### `num_elements`
```python
def num_elements(self) -> 'int':
```

#### `numel`
```python
def numel(self) -> 'int':
```
Alias for num_elements() (PyTorch style).


#### `numpy`
```python
def numpy(self) -> 'np.ndarray':
```
Convert tensor to numpy array.


#### `permute`
```python
def permute(self, *order: 'int') -> 'Tensor':
```

#### `physical_local_shape`
```python
def physical_local_shape(self, shard_idx: 'int' = 0) -> 'graph.Shape | None':
```

#### `physical_local_shape_ints`
```python
def physical_local_shape_ints(self, shard_idx: 'int' = 0) -> 'tuple[int, ...] | None':
```
Int-tuple shape for a specific shard (avoids creating Shape/Dim objects).


#### `realize`
```python
def realize(self) -> 'Tensor':
```
Force immediate realization (blocking).


#### `relu`
```python
def relu(self):
```

#### `requires_grad_`
```python
def requires_grad_(self, value: 'bool' = True) -> 'Tensor':
```
In-place style alias for setting requires_grad (PyTorch style).


#### `reshape`
```python
def reshape(self, shape: 'ShapeLike') -> 'Tensor':
```

#### `round`
```python
def round(self):
```

#### `rsqrt`
```python
def rsqrt(self):
```

#### `shard`
```python
def shard(self, mesh: 'DeviceMesh', dim_specs: 'list[ShardingSpec | str | list[str] | None]', replicated_axes: 'set[str] | None' = None) -> 'Tensor':
```
Shard this tensor across a device mesh, handling resharding and vmap batch dims.


#### `shard_shape`
```python
def shard_shape(self, shard_idx: 'int' = 0) -> 'graph.Shape':
```
Returns the shape of a specific shard.


#### `sigmoid`
```python
def sigmoid(self):
```

#### `silu`
```python
def silu(self):
```

#### `sin`
```python
def sin(self):
```

#### `size`
```python
def size(self, dim: 'int | None' = None) -> 'graph.Shape | int':
```
Returns the shape or size of a specific dimension (PyTorch style).


#### `softmax`
```python
def softmax(self, axis: 'int' = -1):
```

#### `sqrt`
```python
def sqrt(self):
```

#### `squeeze`
```python
def squeeze(self, axis: 'int | tuple[int, ...] | None' = None) -> 'Tensor':
```

#### `sum`
```python
def sum(self, axis: 'int | tuple[int, ...] | None' = None, keepdims: 'bool' = False):
```

#### `swap_axes`
```python
def swap_axes(self, axis1: 'int', axis2: 'int') -> 'Tensor':
```

#### `tanh`
```python
def tanh(self):
```

#### `to`
```python
def to(self, target: 'Device | str | DType') -> 'Tensor':
```
Move tensor to a device or cast to a dtype.

**Parameters**

- **`target`** – Target Device object, device string (e.g. 'cpu', 'gpu:0'), or DType.


#### `to_numpy`
```python
def to_numpy(self) -> 'np.ndarray':
```
Convert tensor to numpy array.


#### `to_numpy_all`
```python
def to_numpy_all(*tensors: 'Tensor') -> 'tuple[np.ndarray, ...]':
```
Convert multiple tensors to numpy arrays in a single batched compilation.

This is more efficient than calling `.to_numpy()` on each tensor individually,
as it combines all gather and realize operations into a single compilation.

**Parameters**

- **`*tensors`** – Variable number of tensors to convert.

**Returns**

 – Tuple of numpy arrays, one per input tensor.


#### `tolist`
```python
def tolist(self) -> 'list[Any]':
```
Convert tensor to a Python list (PyTorch style).


#### `trace`
```python
def trace(self) -> 'Tensor':
```
Enable tracing on this tensor for autograd.


#### `transpose`
```python
def transpose(self, axis1: 'int', axis2: 'int') -> 'Tensor':
```

#### `trunc`
```python
def trunc(self):
```

#### `type_as`
```python
def type_as(self, other: 'Tensor') -> 'Tensor':
```
Cast this tensor to the same dtype as `other`.


#### `unsqueeze`
```python
def unsqueeze(self, axis: 'int') -> 'Tensor':
```

#### `view`
```python
def view(self, *shape: 'int | ShapeLike') -> 'Tensor':
```
Alias for reshape() (PyTorch style).


#### `with_sharding`
```python
def with_sharding(self, mesh: 'DeviceMesh', dim_specs: 'list[ShardingSpec | str | list[str] | None]', replicated_axes: 'set[str] | None' = None) -> 'Tensor':
```
Apply sharding constraint, resharding if needed.


#### `with_sharding_constraint`
```python
def with_sharding_constraint(self, mesh: 'DeviceMesh', dim_specs: 'list[Any]', replicated_axes: 'set[str] | None' = None) -> 'Tensor':
```
Apply sharding constraint for global optimization; no immediate resharding.


---
## `realize_all`

```python
def realize_all(*tensors: 'Tensor') -> 'tuple[Tensor, ...]':
```
Realize multiple tensors in a single batched compilation.

This is more efficient than calling `.realize()` on each tensor individually,
as it combines all pending computations into a single graph compilation.

**Parameters**

- **`*tensors`** – Variable number of tensors to realize.

**Returns**

 – Tuple of realized tensors (same tensors, now with computed values).


---
## `is_tensor`

```python
def is_tensor(obj: 'Any') -> 'bool':
```

---
