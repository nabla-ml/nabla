# Control Flow

## `where`

```python
def where(condition: 'Tensor', x: 'Tensor', y: 'Tensor') -> 'Tensor':
```

---
## `cond`

```python
def cond(pred: 'Tensor', true_fn: 'Callable[..., Any]', false_fn: 'Callable[..., Any]', *operands: 'Any') -> 'Any':
```

---
## `while_loop`

```python
def while_loop(cond_fn: 'Callable[..., bool]', body_fn: 'Callable[..., Any]', init_val: 'Any') -> 'Any':
```

---
## `scan`

```python
def scan(f: 'Callable[[Any, Any], tuple[Any, Any]]', init: 'Any', xs: 'Any', length: 'int | None' = None, reverse: 'bool' = False) -> 'tuple[Any, Any]':
```

---
