# Custom Op Utilities

Base classes and helpers for implementing custom differentiable operations with automatic VJP/JVP support.

## `Operation`

```python
class Operation():
```
Base class for all differentiable operations.

``Operation`` is the fundamental scaffolding for all tensor routines in Nabla.
It manages the entire execution lifecycle: validating and adapting arguments,
handling SPMD sharding rules, maintaining structural hash caches, and
dispatching to either Eager or Trace tracking contexts.

Operations are stateless singletons used purely for dispatch. All necessary
execution context is passed as parameters — subclasses must not maintain
mutable states on ``self``.

**What this automates:**
 - Argument validation and metadata collection.
 - Interaction with ``SPMD`` infrastructure (sharding propagation and resharding).
 - Construction of computational graph ``OpNode``s for Tracing/Autograd.
 - Caching for repeated eager calls.

**What you must implement:**
 - ``name``: A property returning the string name of the op.
 - ``kernel(args, kwargs)``: The low-level execution logic (e.g., calling MAX ops).
 - ``compute_physical_shape(args, kwargs, output_sharding)``: Determines the physical per-shard shape, dtype, and device.
 - ``vjp_rule(...)`` and ``jvp_rule(...)``: Reverse- and Forward-mode autodiff rules.


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
Base class for unary element-wise operations (e.g., exp, sin, relu).

**What this automates:**
 - **Shape Inference:** Automates ``compute_physical_shape`` under the assumption that the output precisely matches the input shape, dtype, and device.
 - **Autodiff Simplification:** Automatically provides ``vjp_rule`` and ``jvp_rule`` by relying on a single analytical derivative method, handling the chain rule multiplication for you.

**What you must implement:**
 - ``name``
 - ``kernel(args, kwargs)``: The element-wise application logic.
 - ``_derivative(primal_tensor, output_tensor)``: Must return a tensor representing the element-wise partial derivative ``d(output) / d(primal)``. (Alternatively, return ``NotImplemented`` and override ``vjp_rule``/``jvp_rule`` manually).


---
## `BinaryOperation`

```python
class BinaryOperation():
```
Base class for binary element-wise operations (e.g. add, mul).

**What this automates:**
 - **Broadcasting:** Automatically resolves logical broadcasting of mismatched input shapes before dispatching to the kernel.
 - **Batch dimension alignment:** Within a ``vmap``, ensures that both operands are broadcasted to share identical batch dimensions.
 - **Physical shape inference:** Automatically implements ``compute_physical_shape`` by assuming outputs share the physical shape of the broadcasted inputs.

**What you must implement:**
 - ``name``
 - ``kernel(args, kwargs)``: Execute the element-wise operation.
 - ``vjp_rule(...)`` and ``jvp_rule(...)``


---
## `ReduceOperation`

```python
class ReduceOperation():
```
Base class for reduction operations (e.g., sum, mean, max).

**What this automates:**
 - **Axis Offsetting:** Inherits from ``AxisOp`` to manage ``vmap`` batch dimensions.
 - **Shape Inference:** Automates ``compute_physical_shape``, safely stripping or preserving dimensions based on the ``keepdims`` kwarg.
 - **Cross-Shard Coordination:** Interacts with SPMD propagation to automatically apply secondary distributed reductions (like ``all_reduce``) if the tensor is sharded across the reduction axis.

**What you must implement:**
 - ``name``
 - ``kernel(args, kwargs)``: Execute the local intra-shard reduction.
 - ``vjp_rule(...)`` and ``jvp_rule(...)``
 - (Optional) ``collective_reduce_type``: Defaults to `"sum"`. Set to `"max"` or `"min"` if necessary.


---
## `AxisOp`

```python
class AxisOp():
```
Base class for operations accepting logical axis/dimension keyword arguments.

**What this automates:**
 - **Batch dimension translation:** When an operation is executed inside ``vmap``, tensors gain implicit batch dimensions (always placed at the front/axis 0). ``AxisOp`` conceptually intercepts any ``axis`` or ``dim`` kwarg and automatically offsets it by the tensor's ``batch_dims`` count before the low-level ``kernel`` sees it.
 - **Sharding Rules:** Provides default SPMD rules that track axes being preserved or removed.

*Note: Subclass this if your op uses an axis but is NOT a reduction (e.g., concatenate, slice).*


---
## `ShapeOp`

```python
class ShapeOp():
```
Base class for pure structural view operations (e.g., reshape, broadcast_to).

**What this automates:**
 - **Batch Dimension Integration:** Intercepts the logical ``shape`` kwarg and automatically prepends implicit ``vmap`` batch dimensions, presenting the correct *physical* target shape to the ``kernel``.
 - **Shape inference:** Automatically deduces the output's local runtime physical shape by cross-referencing the modified global shape with the output sharding spec.

**What you must implement:**
 - ``name``
 - ``kernel(args, kwargs)``: Applying the shape/stride transformations.
 - ``vjp_rule(...)`` and ``jvp_rule(...)``


---
## `CreationOperation`

```python
class CreationOperation():
```
Base class for ops creating independent new tensors (e.g., zeros, uniform).

**What this automates:**
 - **Physical Allocation:** Automates ``compute_physical_shape``, deciphering output shapes, dtypes, and devices from the provided kwargs cleanly.
 - **Per-Shard Replication:** Automatically invokes the kernel across multiple distributed shards without requiring input dependency traversal.
 - **Autodiff Termination:** Provides a default ``vjp_rule`` that safely returns ``None`` for all gradients, as creation operations act as leaves/sources in the autodiff graph.

**What you must implement:**
 - ``name``
 - ``kernel(args, kwargs)``: The initialization allocation routine.


---
## `CollectiveOperation`

```python
class CollectiveOperation():
```
Base class for distributed cross-mesh communication operations.

**What this automates:**
 - **Physical Shape Tracking:** Automates ``compute_physical_shape``, distinguishing between operations that preserve the global shape (like ``all_reduce``) and operations that slice/gather it.
 - **Cost Analysis:** Provides specialized ``communication_cost`` estimation methods based on mesh bandwidth and ring-reduction models.

**What you must implement:**
 - ``name``
 - ``kernel(args, kwargs)``: The low-level distributed routine (e.g. hooking into MAX's distributed ops).
 - ``vjp_rule(...)`` and ``jvp_rule(...)``


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
