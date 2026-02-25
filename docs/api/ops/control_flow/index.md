# Control Flow

## `where`

```python
def where(condition: 'Tensor', x: 'Tensor', y: 'Tensor') -> 'Tensor':
```
Select elements from *x* or *y* based on *condition* (element-wise).

Equivalent to ``condition ? x : y`` applied element-wise with
NumPy-style broadcasting across all three operands.

**Parameters**

- **`condition`** – Boolean tensor. ``True`` selects from *x*, ``False`` from *y*.
- **`x`** – Tensor to select when *condition* is ``True``.
- **`y`** – Tensor to select when *condition* is ``False``.

**Returns**

Tensor with the same shape as the broadcast of *condition*, *x*, *y*.


---
## `cond`

```python
def cond(pred: 'Tensor', true_fn: 'Callable[..., Any]', false_fn: 'Callable[..., Any]', *operands: 'Any') -> 'Any':
```
Conditionally execute one of two branches based on a scalar predicate.

Both branches must return outputs with the same shapes and dtypes.
Only the selected branch is evaluated at runtime.

**Parameters**

- **`pred`** – Scalar boolean tensor. If ``True``, *true_fn* is called;
otherwise *false_fn*.
- **`true_fn`** – Callable invoked when *pred* is ``True``.
- **`false_fn`** – Callable invoked when *pred* is ``False``.
- **`*operands`** – Arguments passed to whichever branch is selected.

**Returns**

Output of the selected branch.


---
## `while_loop`

```python
def while_loop(cond_fn: 'Callable[..., bool]', body_fn: 'Callable[..., Any]', init_val: 'Any') -> 'Any':
```
Execute *body_fn* repeatedly while *cond_fn* returns ``True``.

**Parameters**

- **`cond_fn`** – Takes the current loop state and returns a scalar boolean.
- **`body_fn`** – Takes the current loop state and returns the next state.
Must have the same output structure and shapes as *init_val*.
- **`init_val`** – Initial loop state (can be a tensor or pytree of tensors).

**Returns**

The final loop state after *cond_fn* first returns ``False``.


---
## `scan`

```python
def scan(f: 'Callable[[Any, Any], tuple[Any, Any]]', init: 'Any', xs: 'Any', length: 'int | None' = None, reverse: 'bool' = False) -> 'tuple[Any, Any]':
```
Apply *f* while carrying state, scanning over *xs* along axis 0.

Analogous to JAX's ``jax.lax.scan``. Unrolls the loop at trace time.

**Parameters**

- **`f`** – Function with signature ``(carry, x) -> (carry, y)``.
- **`init`** – Initial carry value.
- **`xs`** – Sequence to scan over. Each element ``xs[i]`` is passed
as the second argument to *f*.
- **`length`** – Number of iterations. Inferred from ``xs[0].shape[0]``
when ``None``.
- **`reverse`** – If ``True``, scan from right to left (not yet supported).

**Returns**

``(final_carry, stacked_ys)`` where *stacked_ys* has an extra
leading dimension of size *length*.


---
