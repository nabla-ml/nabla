# Unary & Activations

## `abs`

```python
def abs(x: 'Tensor') -> 'Tensor':
```

---
## `neg`

```python
def neg(x: 'Tensor') -> 'Tensor':
```

---
## `exp`

```python
def exp(x: 'Tensor') -> 'Tensor':
```

---
## `log`

```python
def log(x: 'Tensor') -> 'Tensor':
```

---
## `log1p`

```python
def log1p(x: 'Tensor') -> 'Tensor':
```

---
## `sqrt`

```python
def sqrt(x: 'Tensor') -> 'Tensor':
```

---
## `rsqrt`

```python
def rsqrt(x: 'Tensor') -> 'Tensor':
```

---
## `sin`

```python
def sin(x: 'Tensor') -> 'Tensor':
```

---
## `cos`

```python
def cos(x: 'Tensor') -> 'Tensor':
```

---
## `acos`

```python
def acos(x: 'Tensor') -> 'Tensor':
```

---
## `atanh`

```python
def atanh(x: 'Tensor') -> 'Tensor':
```

---
## `erf`

```python
def erf(x: 'Tensor') -> 'Tensor':
```

---
## `floor`

```python
def floor(x: 'Tensor') -> 'Tensor':
```

---
## `round`

```python
def round(x: 'Tensor') -> 'Tensor':
```

---
## `trunc`

```python
def trunc(x: 'Tensor') -> 'Tensor':
```

---
## `cast`

```python
def cast(x: 'Tensor', dtype: 'DType | None' = None) -> 'Tensor':
```

---
## `is_inf`

```python
def is_inf(x: 'Tensor') -> 'Tensor':
```

---
## `is_nan`

```python
def is_nan(x: 'Tensor') -> 'Tensor':
```

---
## `relu`

```python
def relu(x: 'Tensor') -> 'Tensor':
```

---
## `sigmoid`

```python
def sigmoid(x: 'Tensor') -> 'Tensor':
```

---
## `tanh`

```python
def tanh(x: 'Tensor') -> 'Tensor':
```

---
## `gelu`

```python
def gelu(x: 'Tensor') -> 'Tensor':
```

---
## `silu`

```python
def silu(x: 'Tensor') -> 'Tensor':
```

---
## `softmax`

```python
def softmax(x: 'Tensor', axis: 'int' = -1) -> 'Tensor':
```
A composition of existing nabla ops


---
## `logsoftmax`

```python
def logsoftmax(x: 'Tensor', axis: 'int' = -1) -> 'Tensor':
```
LogSoftmax implementation with sharding support.


---
