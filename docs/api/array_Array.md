# Array

## Signature

```python
nabla.Array(shape: 'Shape', dtype: 'DType' = float32, device: 'Device' = Device(type=cpu,id=0), materialize: 'bool' = False, name: 'str' = '', batch_dims: 'Shape' = ()) -> 'None'
```

## Description

Core tensor-like array class with automatic differentiation support. The Array class is the fundamental data structure in Nabla, providing efficient computation with automatic gradient tracking.

## Properties

### `shape`

- **Type:** `tuple[int, ...]`
- **Description:** The shape of the array (excluding batch dimensions)

### `batch_dims`

- **Type:** `tuple[int, ...]`
- **Description:** Batch dimensions for vectorized operations

### `dtype`

- **Type:** `DType`
- **Description:** Data type of the array elements

### `device`

- **Type:** `Device`
- **Description:** Device where the array is stored (CPU/GPU)

### `name`

- **Type:** `str`
- **Description:** Optional name for debugging and identification

## Methods

### Array Creation and Conversion

#### `from_impl(impl: Tensor, name: str = "") -> Array`

**Class method.** Create Array from existing MAX Tensor implementation.

#### `from_numpy(np_array: np.ndarray) -> Array`

**Class method.** Create a new Array from a NumPy array.

#### `to_numpy() -> np.ndarray`

Get NumPy representation with caching. Forces realization if needed.

#### `copy_from(other: Array) -> None`

Copy data from another Array. Arrays must have matching shape and dtype.

### Device Operations

#### `to(device: Device) -> Array`

Move Array to specified device (CPU/GPU).

### Computation and Realization

#### `realize() -> None`

Force computation of this Array. Executes the computation graph if needed.

#### `add_arguments(*arg_nodes: Array) -> None`

Add arguments to this Array's computation graph for gradient tracking.

#### `get_arguments() -> list[Array]`

Get list of argument Arrays used in this Array's computation.

### Arithmetic Operators

#### `__add__(self, other) -> Array` / `+`

Element-wise addition. Supports broadcasting.

#### `__mul__(self, other) -> Array` / `*`

Element-wise multiplication. Supports broadcasting.

#### `__sub__(self, other) -> Array` / `-`

Element-wise subtraction. Supports broadcasting.

#### `__truediv__(self, other) -> Array` / `/`

Element-wise division. Supports broadcasting.

#### `__pow__(self, power) -> Array` / `**`

Element-wise exponentiation. Supports broadcasting.

#### `__matmul__(self, other) -> Array` / `@`

Matrix multiplication (dot product for 2D arrays).

#### `__neg__(self) -> Array` / `-self`

Element-wise negation.

### Reverse Operators

#### `__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`

Reverse arithmetic operators for when Array is on the right-hand side.

### Array Indexing and Slicing

#### `__getitem__(self, key) -> Array` / `arr[...]`

Array slicing using standard Python syntax. Supports:

- Single indices: `arr[2]`
- Slices: `arr[1:5]`
- Multiple dimensions: `arr[1:3, 2:5]`
- Negative indices: `arr[-2:]`
- Ellipsis: `arr[..., :2]`

### Reduction Operations

#### `sum(axes=None, keep_dims=False) -> Array`

Sum array elements over given axes.

- **Args:**
  - `axes`: Axis or axes to sum along (int, list of ints, or None for all)
  - `keep_dims`: If True, keep reduced axes as size-1 dimensions
- **Returns:** Array with summed values

## Examples

### Basic Array Creation

```python
import nabla as nb

# Create array from shape
arr = nb.Array(shape=(3, 4), dtype=nb.float32)

# Create from NumPy
import numpy as np
np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
nb_arr = nb.Array.from_numpy(np_arr)
```

### Arithmetic Operations

```python
# Element-wise operations
a = nb.Array.from_numpy(np.array([1, 2, 3]))
b = nb.Array.from_numpy(np.array([4, 5, 6]))

c = a + b      # Addition
d = a * b      # Multiplication
e = a @ b.T    # Matrix multiplication
f = -a         # Negation
```

### Array Slicing

```python
arr = nb.Array.from_numpy(np.arange(12).reshape(3, 4))

sliced = arr[1:3, :2]     # Slice rows 1-2, columns 0-1
row = arr[0]              # Get first row
col = arr[:, -1]          # Get last column
```

### Reduction Operations

```python
arr = nb.Array.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))

total = arr.sum()                    # Sum all elements
row_sums = arr.sum(axis=1)          # Sum along rows
col_sums = arr.sum(axis=0)          # Sum along columns
```

### Device Operations

```python
# Move to different device
cpu_arr = nb.Array.from_numpy(np.array([1, 2, 3]))
gpu_arr = cpu_arr.to(nb.GPU(0))  # Move to GPU
back_to_cpu = gpu_arr.to(nb.CPU())  # Move back to CPU
```

## See Also

- Array creation functions - Functions for creating Arrays
- Binary operations - Element-wise binary operations  
- Transformations - Automatic differentiation
- Device management - GPU/CPU operations

