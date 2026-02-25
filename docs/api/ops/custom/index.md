# Custom Op Utilities

Base classes and helpers for implementing custom differentiable operations with automatic VJP/JVP support.

## `Operation`

```python
class Operation():
```
Base class for all operations.

Operations are stateless singletons used purely for dispatch. All context
needed by a method is passed as parameters — no mutable state on self.

Classes serve as namespace + inheritance for shared logic; instances carry
only class-level constants (name, flags, axis_arg_names, etc.).


### Methods

#### `adapt_kwargs`
```python
def adapt_kwargs(self, args: 'OpArgs', kwargs: 'OpKwargs', batch_dims: 'int') -> 'OpKwargs':
```

#### `compute_cost`
```python
def compute_cost(self, input_shapes: 'list[tuple[int, ...]]', output_shapes: 'list[tuple[int, ...]]') -> 'float':
```
Estimate compute cost (FLOPs).


#### `compute_physical_shape`
```python
def compute_physical_shape(self, args: 'OpArgs', kwargs: 'OpKwargs', output_sharding: 'ShardingSpec | None' = None) -> 'tuple[list[tuple[int, ...]] | None, list[DType] | None, list[Any] | None]':
```
Infer per-shard physical shapes for outputs.

Subclasses must override this when used with physical execution.


#### `execute`
```python
def execute(self, args: 'OpArgs', kwargs: 'OpKwargs') -> 'tuple[list[Any], ShardingSpec | None, DeviceMesh | None]':
```
Default physical execution: execute kernel on each shard independently.

Operations with specialized execution logic (e.g. CreationOperation)
can override. Uses adapt_kwargs for batch_dims offset and conditionally
infers output sharding based on _infer_output_sharding class flag.

**Parameters**

- **`args`** – Flat list of Tensor inputs.
- **`kwargs`** – Static metadata dictionary.

**Returns**

**`tuple`** – (shard_results, output_sharding, mesh)


#### `get_sharding_rule_template`
```python
def get_sharding_rule_template(self) -> 'Any':
```

#### `infer_output_rank`
```python
def infer_output_rank(self, input_shapes: 'list[tuple[int, ...]]', **kwargs) -> 'int':
```
Infer output rank from input shapes.


#### `jvp_rule`
```python
def jvp_rule(self, primals: 'list[Tensor]', tangents: 'list[Tensor]', outputs: 'list[Tensor]', kwargs: 'OpKwargs') -> 'list[Tensor | None]':
```

#### `kernel`
```python
def kernel(self, args: 'OpTensorValues', kwargs: 'OpKwargs') -> 'OpTensorValues':
```
Execute the low-level computation.

**Parameters**

- **`args`** – Flat list of TensorValue inputs.
- **`kwargs`** – Static metadata dictionary.

**Returns**

Flat list of TensorValue outputs (even for single-output ops).


#### `memory_cost`
```python
def memory_cost(self, input_shapes: 'list[tuple[int, ...]]', output_shapes: 'list[tuple[int, ...]]', dtype_bytes: 'int' = 4) -> 'int':
```
Estimate memory usage (bytes) for output tensors.


#### `sharding_rule`
```python
def sharding_rule(self, input_shapes: 'list[tuple[int, ...]]', output_shapes: 'list[tuple[int, ...]]', **kwargs: 'Any') -> 'Any':
```
Default sharding rule: elementwise for same-rank ops.


#### `vjp_rule`
```python
def vjp_rule(self, primals: 'list[Tensor]', cotangents: 'list[Tensor]', outputs: 'list[Tensor]', kwargs: 'OpKwargs') -> 'list[Tensor | None]':
```

---
## `UnaryOperation`

```python
class UnaryOperation():
```
Base for unary element-wise operations.


---
## `BinaryOperation`

```python
class BinaryOperation():
```
Base for binary element-wise ops with batch_dims-aware broadcasting.


---
## `ReduceOperation`

```python
class ReduceOperation():
```
Base for reduction operations (sum, mean, max, min, etc.).


---
## `AxisOp`

```python
class AxisOp():
```
Base for ops that take LOGICAL axis/axes kwargs.

Translates integer kwargs by batch_dims offset.


---
## `ShapeOp`

```python
class ShapeOp():
```
Base for ops that take LOGICAL shape kwargs.


---
## `CreationOperation`

```python
class CreationOperation():
```
Base for operations that create new tensors (e.g. zeros, ones, random).

Creation operations don't follow the standard element-wise sharding;
instead, they generate data directly on each shard.


---
## `CollectiveOperation`

```python
class CollectiveOperation():
```
Base class for collective communication operations.

Handles value hydration, graph execution (kernel), and output wrapping/sharding update.


---
## `ensure_tensor`

```python
def ensure_tensor(x: 'Any') -> 'Tensor':
```
Convert a scalar or array-like value to a :class:`Tensor`.

Useful inside custom operation implementations that need to accept
Python scalars, NumPy arrays, or existing tensors uniformly.

**Parameters**

- **`x`** – Value to wrap. If already a :class:`Tensor`, returned unchanged.
Otherwise, a new constant :class:`Tensor` is created via
:meth:`Tensor.constant`.

**Returns**

A :class:`Tensor` wrapping *x*.


---
