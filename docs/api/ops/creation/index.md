# Creation

## `zeros`

```python
def zeros(shape: 'ShapeLike', *, dtype: 'DType | None' = None, device: 'Device | None' = None, is_traced: 'bool' = False):
```
Create a tensor filled with zeros.


---
## `ones`

```python
def ones(shape: 'ShapeLike', *, dtype: 'DType | None' = None, device: 'Device | None' = None, is_traced: 'bool' = False):
```
Create a tensor filled with ones.


---
## `full`

```python
def full(shape: 'ShapeLike', value: 'Number', *, dtype: 'DType | None' = None, device: 'Device | None' = None, is_traced: 'bool' = False):
```
Create a tensor filled with a constant value.


---
## `constant`

```python
def constant(value: 'NestedArray | Number', *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Create a tensor from a constant value.


---
## `arange`

```python
def arange(start: 'int' = 0, stop: 'int | None' = None, step: 'int' = 1, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Create a tensor with evenly spaced values.


---
## `uniform`

```python
def uniform(shape: 'ShapeLike' = (), low: 'float' = 0.0, high: 'float' = 1.0, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Create a tensor with uniform random values.


---
## `normal`

```python
def normal(shape: 'ShapeLike' = (), mean: 'float' = 0.0, std: 'float' = 1.0, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Create a tensor with Gaussian (normal) random values.


---
## `gaussian`

```python
def gaussian(shape: 'ShapeLike' = (), mean: 'float' = 0.0, std: 'float' = 1.0, *, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Create a tensor with Gaussian (normal) random values.


---
## `hann_window`

```python
def hann_window(window_length: 'int', *, periodic: 'bool' = True, dtype: 'DType | None' = None, device: 'Device | None' = None):
```
Create a 1D Hann window tensor.


---
## `triu`

```python
def triu(x: 'Tensor', k: 'int' = 0) -> 'Tensor':
```
Upper triangular part of a matrix.


---
## `tril`

```python
def tril(x: 'Tensor', k: 'int' = 0) -> 'Tensor':
```
Lower triangular part of a matrix.


---
## `zeros_like`

```python
def zeros_like(x: 'Tensor') -> 'Tensor':
```
Create a tensor of zeros with the same shape/dtype/device/sharding as x.


---
## `ones_like`

```python
def ones_like(x: 'Tensor') -> 'Tensor':
```
Create a tensor of ones with the same shape/dtype/device/sharding as x.


---
## `full_like`

```python
def full_like(x: 'Tensor', value: 'Number') -> 'Tensor':
```
Create a tensor filled with value, matching x's properties.


---
