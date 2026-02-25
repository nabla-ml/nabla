# Creation

## `zeros`

```python
def zeros(shape: 'ShapeLike', *, dtype: 'DType | None' = None, device: 'Device | None' = None, is_traced: 'bool' = False):
```
Return a tensor of *shape* filled with zeros.

**Parameters**

- **`shape`** – Output shape.
- **`dtype`** – Element dtype. Uses the current default if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

Zero-valued tensor of the given shape and dtype.


---
## `ones`

```python
def ones(shape: 'ShapeLike', *, dtype: 'DType | None' = None, device: 'Device | None' = None, is_traced: 'bool' = False):
```
Return a tensor of *shape* filled with ones.

**Parameters**

- **`shape`** – Output shape.
- **`dtype`** – Element dtype. Uses the current default if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

One-valued tensor of the given shape and dtype.


---
## `full`

```python
def full(shape: 'ShapeLike', value: 'Number', *, dtype: 'DType | None' = None, device: 'Device | None' = None, is_traced: 'bool' = False):
```
Return a tensor of *shape* filled with *value*.

**Parameters**

- **`shape`** – Output shape.
- **`value`** – Fill value (scalar).
- **`dtype`** – Element dtype. Uses the current default if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

Tensor with all elements equal to *value*.


---
## `constant`

```python
def constant(value: 'NestedArray | Number', *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Create a tensor from a Python scalar, list, or NumPy array.

Scalars and 0-d arrays are wrapped with the default dtype unless
*dtype* is specified. Multi-dimensional arrays/lists are converted
via DLPack without copying.

**Parameters**

- **`value`** – Python int, float, bool, complex, list, or ``np.ndarray``.
- **`dtype`** – Target element dtype. Inferred from *value* if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

A realized ``Tensor`` wrapping *value*.


---
## `arange`

```python
def arange(start: 'int' = 0, stop: 'int | None' = None, step: 'int' = 1, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Return a 1-D tensor with evenly spaced values in ``[start, stop)``.

When called with a single positional argument, it is treated as *stop*
and *start* defaults to ``0``, matching NumPy / PyTorch semantics.

**Parameters**

- **`start`** – Start of the interval (inclusive). Default: ``0``.
- **`stop`** – End of the interval (exclusive). If ``None``, *start* is
used as *stop* and start becomes ``0``.
- **`step`** – Spacing between values. Default: ``1``.
- **`dtype`** – Element dtype. Uses the current default if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

1-D tensor of length ``ceil((stop - start) / step)``.


---
## `uniform`

```python
def uniform(shape: 'ShapeLike' = (), low: 'float' = 0.0, high: 'float' = 1.0, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Return a tensor of *shape* with values sampled from U(*low*, *high*).

**Parameters**

- **`shape`** – Output shape.
- **`low`** – Lower bound of the uniform distribution.
- **`high`** – Upper bound of the uniform distribution.
- **`dtype`** – Element dtype. Uses the current default if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

Tensor with elements drawn uniformly at random from [*low*, *high*).


---
## `normal`

```python
def normal(shape: 'ShapeLike' = (), mean: 'float' = 0.0, std: 'float' = 1.0, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Return a tensor of *shape* with values sampled from N(*mean*, *std*²).

Also accessible as ``nabla.normal``.

**Parameters**

- **`shape`** – Output shape.
- **`mean`** – Mean of the Gaussian distribution. Default: ``0.0``.
- **`std`** – Standard deviation. Default: ``1.0``.
- **`dtype`** – Element dtype. Uses the current default if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

Tensor with elements drawn from a normal distribution.


---
## `gaussian`

```python
def gaussian(shape: 'ShapeLike' = (), mean: 'float' = 0.0, std: 'float' = 1.0, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Return a tensor of *shape* with values sampled from N(*mean*, *std*²).

Also accessible as ``nabla.normal``.

**Parameters**

- **`shape`** – Output shape.
- **`mean`** – Mean of the Gaussian distribution. Default: ``0.0``.
- **`std`** – Standard deviation. Default: ``1.0``.
- **`dtype`** – Element dtype. Uses the current default if ``None``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

Tensor with elements drawn from a normal distribution.


---
## `hann_window`

```python
def hann_window(window_length: 'int', *, periodic: 'bool' = True, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Return a 1-D Hann (raised cosine) window of length *window_length*.

The window follows the convention used in signal processing:
``w[n] = 0.5 * (1 - cos(2π n / N))`` where *N* is the window size.

**Parameters**

- **`window_length`** – Number of points in the window.
- **`periodic`** – If ``True`` (default), generates a periodic window for
use in spectral analysis. If ``False``, generates a symmetric
window.
- **`dtype`**, default: ```float32``` – Element dtype. Defaults to ``float32``.
- **`device`** – Target device. Uses the current default if ``None``.

**Returns**

1-D Tensor of shape ``(window_length,)``.


---
## `triu`

```python
def triu(x: 'Tensor', k: 'int' = 0) -> 'Tensor':
```
Return the upper triangular part of a matrix (or batch of matrices).

Elements below the *k*-th diagonal are zeroed out.

**Parameters**

- **`x`** – Input tensor of shape ``(*, M, N)``.
- **`k`** – Diagonal offset. ``k=0`` (default) is the main diagonal,
``k>0`` is above it, ``k<0`` is below.

**Returns**

Tensor of the same shape as *x* with the lower triangle zeroed.


---
## `tril`

```python
def tril(x: 'Tensor', k: 'int' = 0) -> 'Tensor':
```
Return the lower triangular part of a matrix (or batch of matrices).

Elements above the *k*-th diagonal are zeroed out.

**Parameters**

- **`x`** – Input tensor of shape ``(*, M, N)``.
- **`k`** – Diagonal offset. ``k=0`` (default) is the main diagonal,
``k>0`` is above it, ``k<0`` is below.

**Returns**

Tensor of the same shape as *x* with the upper triangle zeroed.


---
## `zeros_like`

```python
def zeros_like(x: 'Tensor') -> 'Tensor':
```
Return a zero tensor with the same shape, dtype, device, and sharding as *x*.

**Parameters**

- **`x`** – Reference tensor.

**Returns**

Tensor of zeros matching all metadata of *x*.


---
## `ones_like`

```python
def ones_like(x: 'Tensor') -> 'Tensor':
```
Return a ones tensor with the same shape, dtype, device, and sharding as *x*.

**Parameters**

- **`x`** – Reference tensor.

**Returns**

Tensor of ones matching all metadata of *x*.


---
## `full_like`

```python
def full_like(x: 'Tensor', value: 'Number') -> 'Tensor':
```
Return a tensor filled with *value*, matching *x*'s shape, dtype, device, and sharding.

**Parameters**

- **`x`** – Reference tensor.
- **`value`** – Scalar fill value.

**Returns**

Tensor filled with *value*, with the same metadata as *x*.


---
