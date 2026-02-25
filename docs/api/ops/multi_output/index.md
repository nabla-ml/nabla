# Multi-Output

## `split`

```python
def split(x: 'Tensor', num_splits: 'int', axis: 'int' = 0) -> 'list':
```
Split a tensor into *num_splits* equal chunks along *axis*.

**Parameters**

- **`x`** – Input tensor. The size along *axis* must be divisible by *num_splits*.
- **`num_splits`** – Number of equal parts to split into.
- **`axis`** – Axis along which to split. Default: ``0``.

**Returns**

List of *num_splits* tensors each with size ``x.shape[axis] // num_splits``
along *axis* and the same size as *x* in all other dimensions.


---
## `chunk`

```python
def chunk(x: 'Tensor', chunks: 'int', axis: 'int' = 0) -> 'list':
```
Split a tensor into *chunks* chunks along *axis*.

The last chunk may be smaller if the axis size is not divisible by *chunks*.

**Parameters**

- **`x`** – Input tensor.
- **`chunks`** – Number of chunks to split into.
- **`axis`** – Axis along which to split. Default: ``0``.

**Returns**

List of tensors. All chunks except possibly the last have size
``ceil(x.shape[axis] / chunks)`` along *axis*.


---
## `unbind`

```python
def unbind(x: 'Tensor', axis: 'int' = 0) -> 'list':
```
Remove *axis* and return a list of slices along that dimension.

Analogous to Python's ``list(x)`` applied along *axis*.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Axis to remove. Default: ``0``.

**Returns**

List of ``x.shape[axis]`` tensors, each with rank one less than *x*.


---
## `minmax`

```python
def minmax(x: 'Tensor') -> 'dict[str, Tensor]':
```
Compute both the global minimum and maximum of *x*.

**Parameters**

- **`x`** – Input tensor of any shape.

**Returns**

**```{'min'`** – scalar_min, 'max': scalar_max}`` — both are scalar tensors.


---
