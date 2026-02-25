# Unary & Activations

## `abs`

```python
def abs(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise absolute value ``|x|``.


---
## `neg`

```python
def neg(x: 'Tensor') -> 'Tensor':
```
Negate each element: ``-x``.


---
## `exp`

```python
def exp(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise natural exponential ``e^x``.


---
## `log`

```python
def log(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise natural logarithm ``log(x)``.

Returns ``-inf`` for zero and ``nan`` for negative inputs.


---
## `log1p`

```python
def log1p(x: 'Tensor') -> 'Tensor':
```
Compute ``log(1 + x)`` element-wise, numerically stable near ``x = 0``.


---
## `sqrt`

```python
def sqrt(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise square root ``sqrt(x)``.


---
## `rsqrt`

```python
def rsqrt(x: 'Tensor') -> 'Tensor':
```
Compute the reciprocal square root ``1 / sqrt(x)`` element-wise.


---
## `sin`

```python
def sin(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise sine ``sin(x)`` (radians).


---
## `cos`

```python
def cos(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise cosine ``cos(x)`` (radians).


---
## `acos`

```python
def acos(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise arccosine ``acos(x)``, returning ``nan`` for ``|x| > 1``.


---
## `atanh`

```python
def atanh(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise inverse hyperbolic tangent ``atanh(x)``.


---
## `erf`

```python
def erf(x: 'Tensor') -> 'Tensor':
```
Compute the element-wise Gauss error function ``erf(x)``.


---
## `floor`

```python
def floor(x: 'Tensor') -> 'Tensor':
```
Round each element down to the nearest integer (towards ``-inf``).


---
## `round`

```python
def round(x: 'Tensor') -> 'Tensor':
```
Round each element to the nearest integer (half-to-even / banker's rounding).


---
## `trunc`

```python
def trunc(x: 'Tensor') -> 'Tensor':
```
Round each element towards zero (truncation).


---
## `cast`

```python
def cast(x: 'Tensor', dtype: 'DType | None' = None) -> 'Tensor':
```
Cast *x* to a different data type.

**Parameters**

- **`x`** – Input tensor.
- **`dtype`** – Target :class:`DType`. If ``None``, the tensor is returned unchanged.

**Returns**

 – Tensor with elements reinterpreted as *dtype*.


---
## `is_inf`

```python
def is_inf(x: 'Tensor') -> 'Tensor':
```
Return a boolean tensor that is ``True`` where *x* is infinite.


---
## `is_nan`

```python
def is_nan(x: 'Tensor') -> 'Tensor':
```
Return a boolean tensor that is ``True`` where *x* is NaN.


---
## `relu`

```python
def relu(x: 'Tensor') -> 'Tensor':
```
Apply the Rectified Linear Unit (ReLU) activation element-wise: ``max(0, x)``.


---
## `sigmoid`

```python
def sigmoid(x: 'Tensor') -> 'Tensor':
```
Apply the sigmoid activation element-wise: ``1 / (1 + exp(-x))``.


---
## `tanh`

```python
def tanh(x: 'Tensor') -> 'Tensor':
```
Apply the hyperbolic tangent activation element-wise.


---
## `gelu`

```python
def gelu(x: 'Tensor') -> 'Tensor':
```
Apply the Gaussian Error Linear Unit (GELU) activation element-wise.

Uses the exact formulation: ``x * Φ(x)`` where ``Φ`` is the standard
normal CDF.


---
## `silu`

```python
def silu(x: 'Tensor') -> 'Tensor':
```
Apply the SiLU (Swish) activation element-wise: ``x * sigmoid(x)``.


---
## `softmax`

```python
def softmax(x: 'Tensor', axis: 'int' = -1) -> 'Tensor':
```
Compute softmax probabilities along *axis*: ``exp(x) / sum(exp(x))``.

Numerically stable. When the reduction axis is sharded across devices,
an all-reduce is automatically inserted.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Dimension along which to compute. Default: ``-1``.

**Returns**

 – Tensor of the same shape as *x* with non-negative values summing to 1.


---
## `logsoftmax`

```python
def logsoftmax(x: 'Tensor', axis: 'int' = -1) -> 'Tensor':
```
Compute log-softmax along *axis*: ``log(exp(x) / sum(exp(x)))``.

Numerically stable via the log-sum-exp trick. When the reduction axis
is sharded across devices, an all-reduce is automatically inserted.

**Parameters**

- **`x`** – Input tensor.
- **`axis`** – Dimension along which to compute. Default: ``-1``.

**Returns**

 – Tensor of the same shape as *x* with log-probabilities.


---
