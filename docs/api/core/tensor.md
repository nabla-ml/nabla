# Tensor

## `Tensor`

```python
class Tensor(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), materialize: 'bool' = False, name: 'str' = '', batch_dims: 'Shape' = ()) -> 'None':
```
Core tensor-like tensor class with automatic differentiation support.


### Methods

#### `add_arguments`
```python
def add_arguments(self, *arg_nodes: 'Tensor') -> 'None':
```
Add an arguments to this Tensor's computation graph if traced.


#### `astype`
```python
def astype(self, dtype: 'DType') -> 'Tensor':
```
Convert tensor to a different data type.

**Parameters**

- **`dtype`** – Target data type

**Returns**

 – New Tensor with the specified data type


#### `at`
```python
def at(self, key, value):
```
Update tensor at specified indices/slices, returning new tensor.


#### `backward`
```python
def backward(self, grad: 'Tensor | None' = None, retain_graph: 'bool' = False) -> 'None':
```
Compute gradients flowing into traced leaf inputs that influence this Tensor.

**Parameters**

- **`grad`** – Optional cotangent tensor; defaults to ones for scalar outputs
- **`retain_graph`** – If False (default), frees the computation graph after backward pass


#### `copy_from`
```python
def copy_from(self, other: 'Tensor') -> 'None':
```
Copy data from another Tensor.


#### `get_arguments`
```python
def get_arguments(self) -> 'list[Tensor]':
```
Get list of argument Tensors.


#### `impl_`
```python
def impl_(self, value: 'Union[np.ndarray, MAXTensor] | None') -> 'None':
```
Set the implementation of this Tensor to a Numpy tensor or Tensor.


#### `permute`
```python
def permute(self, axes: 'tuple[int, ...]') -> 'Tensor':
```
Permute the dimensions of the tensor.

**Parameters**

- **`axes`** – List of integers specifying the new order of dimensions

**Returns**

 – Tensor with dimensions permuted according to the specified axes


#### `realize`
```python
def realize(self) -> 'None':
```
Force computation of this Tensor.


#### `requires_grad_`
```python
def requires_grad_(self, val: 'bool' = True) -> 'Tensor':
```
Opt into or out of gradient tracking for imperative workflows.

This is an in-place operation that returns self for method chaining.
Similar to PyTorch's requires_grad_() method.


#### `reshape`
```python
def reshape(self, shape: 'Shape') -> 'Tensor':
```
Change the shape of an tensor without changing its data.

**Parameters**

- **`shape`** – New shape for the tensor

**Returns**

 – Tensor with the new shape


#### `set`
```python
def set(self, key, value) -> 'Tensor':
```
Set values at specified indices/slices, returning a new tensor.

This is a functional operation that returns a new Tensor with the specified
values updated, leaving the original Tensor unchanged.

**Parameters**

- **`key`** – Index specification (int, slice, tuple of indices/slices, ellipsis)
- **`value`** – Value(s) to set at the specified location

**Returns**

 – New Tensor with updated values

**Examples**

new_arr = arr.set(1, 99.0)              # Set single element
new_arr = arr.set((1, 2), 99.0)         # Set element at (1,2)
new_arr = arr.set(slice(1, 3), 99.0)    # Set slice
new_arr = arr.set(..., 99.0)            # Set with ellipsis


#### `set_maxpr`
```python
def set_maxpr(self, fn: 'MaxprCallable') -> 'None':
```
Set the MAX PR function for this operation.


#### `sum`
```python
def sum(self, axes=None, keep_dims=False) -> 'Tensor':
```
Sum tensor elements over given axes.

**Parameters**

- **`axes`** – Axis or axes along which to sum. Can be int, list of ints, or None (sum all)
- **`keep_dims`** – If True, reduced axes are left as dimensions with size 1

**Returns**

 – Tensor with the sum along the specified axes


#### `to`
```python
def to(self, device: 'Device') -> 'Tensor':
```
Move Tensor to specified device.


#### `to_numpy`
```python
def to_numpy(self) -> 'np.ndarray':
```
Get NumPy representation.


#### `transpose`
```python
def transpose(self, axes: 'tuple[int, ...]') -> 'Tensor':
```
Permute the dimensions of the tensor.

**Parameters**

- **`axes`** – List of integers specifying the new order of dimensions

**Returns**

 – Tensor with dimensions permuted according to the specified axes


---
