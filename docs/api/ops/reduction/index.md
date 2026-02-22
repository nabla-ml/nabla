# Reduction

## `reduce_sum`

```python
def reduce_sum(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `reduce_max`

```python
def reduce_max(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `reduce_min`

```python
def reduce_min(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `sum`

```python
def sum(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `max`

```python
def max(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `min`

```python
def min(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `mean`

```python
def mean(x: 'Tensor', *, axis: 'int | tuple[int, ...] | list[int] | None' = None, keepdims: 'bool' = False) -> 'Tensor':
```
Compute arithmetic mean along specified axis/axes.

Implemented as sum(x) / product(shape[axes]) to correctly handle distributed sharding.


---
## `argmax`

```python
def argmax(x: 'Tensor', axis: 'int' = -1, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `argmin`

```python
def argmin(x: 'Tensor', axis: 'int' = -1, keepdims: 'bool' = False) -> 'Tensor':
```

---
## `cumsum`

```python
def cumsum(x: 'Tensor', axis: 'int' = -1, exclusive: 'bool' = False, reverse: 'bool' = False) -> 'Tensor':
```

---
## `reduce_sum_physical`

```python
def reduce_sum_physical(x: 'Tensor', axis: 'int', keepdims: 'bool' = False) -> 'Tensor':
```

---
## `mean_physical`

```python
def mean_physical(x: 'Tensor', axis: 'int', keepdims: 'bool' = False) -> 'Tensor':
```

---
