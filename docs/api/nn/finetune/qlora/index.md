# QLoRA

## `NF4_CODEBOOK`

```python
def NF4_CODEBOOK():
```
ndarray(shape, dtype=None, buffer=None, offset=0, strides=None, order=None)

--

ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using `array`, `zeros` or `empty` (refer
to the See Also section below).  The parameters given here refer to
a low-level method (`ndarray(...)`) for instantiating an array.

For more information, refer to the `numpy` module and examine the
methods and attributes of an array.

**Parameters**

- **`(for the __new__ method; see Notes below)`** – None
- **`shape`** : `tuple of ints` – Shape of created array.
- **`dtype`** : `data-type`, optional, default: `is` – Any object that can be interpreted as a numpy data type.
Default is `numpy.float64`.
- **`buffer`** : `object exposing buffer interface`, optional – Used to fill the array with data.
- **`offset`** : `int`, optional – Offset of array data in buffer.
- **`strides`** : `tuple of ints`, optional – Strides of data in memory.
- **`order`** : `{'C', 'F'}`, optional – Row-major (C-style) or column-major (Fortran-style) order.
- **`T`** : `ndarray` – Transpose of the array.
- **`data`** : `buffer` – The array's elements, in memory.
- **`dtype`** : `dtype object` – Describes the format of the elements in the array.
- **`flags`** : `dict` – Dictionary containing information related to memory use, e.g.,
'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
- **`flat`** : `numpy.flatiter object` – Flattened version of the array as an iterator.  The iterator
allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
assignment examples; TODO).
- **`imag`** : `ndarray` – Imaginary part of the array.
- **`real`** : `ndarray` – Real part of the array.
- **`size`** : `int` – Number of elements in the array.
- **`itemsize`** : `int` – The memory use of each array element in bytes.
- **`nbytes`** : `int` – The total number of bytes required to store the array data,
i.e., ``itemsize * size``.
- **`ndim`** : `int` – The array's number of dimensions.
- **`shape`** : `tuple of ints` – Shape of the array.
- **`strides`** : `tuple of ints` – The step-size required to move from one element to the next in
memory. For example, a contiguous ``(3, 4)`` array of type
``int16`` in C-order has strides ``(8, 2)``.  This implies that
to move from element to element in memory requires jumps of 2 bytes.
To move from row-to-row, one needs to jump 8 bytes at a time
(``2 * 4``).
- **`ctypes`** : `ctypes object` – Class containing properties of the array needed for interaction
with ctypes.
- **`base`** : `ndarray` – If the array is a view into another array, that array is its `base`
(unless that array is also a view).  The `base` array is where the
array data is actually stored.

**Examples**

These examples illustrate the low-level `ndarray` constructor.  Refer
to the `See Also` section above for easier ways of constructing an
ndarray.

First mode, `buffer` is None:

```python
>>> import numpy as np
>>> np.ndarray(shape=(2,2), dtype=float, order='F')
array([[0.0e+000, 0.0e+000], # random
       [     nan, 2.5e-323]])
```

Second mode:

```python
>>> np.ndarray((2,), buffer=np.array([1,2,3]),
...            offset=np.int_().itemsize,
...            dtype=int) # offset = 1*itemsize, i.e. skip first element
array([2, 3])
```

---
## `quantize_nf4`

```python
def quantize_nf4(weight: 'Tensor', block_size: 'int' = 64) -> 'dict[str, Any]':
```
Quantize a 2D weight to NF4 indices + per-block scales.


---
## `dequantize_nf4`

```python
def dequantize_nf4(qweight: 'dict[str, Any]', *, dtype: 'DType' = float32) -> 'Tensor':
```
Dequantize NF4 weight dict back to dense tensor using Nabla ops.


---
## `qlora_linear`

```python
def qlora_linear(x: 'Tensor', qweight: 'dict[str, Any]', adapter: 'dict[str, Tensor]', *, alpha: 'float' = 1.0, compute_dtype: 'DType' = float32) -> 'Tensor':
```
QLoRA-style linear layer using frozen NF4 weight + LoRA adapter.


---
