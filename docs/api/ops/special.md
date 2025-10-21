# Special Operations

## `softmax`

```python
def softmax(arg: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor:
```
Computes the softmax function for an tensor.

The softmax function transforms a vector of real numbers into a probability
distribution. Each element in the output is in the range (0, 1), and the
elements along the specified axis sum to 1. It is calculated in a
numerically stable way as `exp(x - logsumexp(x))`.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.
- **`axis`** : `int`, optional, default: `is` – The axis along which the softmax computation is performed. The default
is -1, which is the last axis.

**Returns**

`Tensor` – An tensor of the same shape as the input, containing the softmax
probabilities.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1.0, 2.0, 3.0])
>>> nb.softmax(x)
Tensor([0.09003057, 0.24472848, 0.66524094], dtype=float32)
```

```python
>>> logits = nb.tensor([[1, 2, 3], [1, 1, 1]])
>>> nb.softmax(logits, axis=1)
Tensor([[0.09003057, 0.24472848, 0.66524094],
       [0.33333334, 0.33333334, 0.33333334]], dtype=float32)
```

---
## `logsumexp`

```python
def logsumexp(arg: nabla.core.tensor.Tensor, axis: int | None = None, keep_dims: bool = False) -> nabla.core.tensor.Tensor:
```
Computes the log of the sum of exponentials of input elements.

This function computes `log(sum(exp(x)))` in a numerically stable way by using
the identity: `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`. This
avoids overflow errors that can occur when `exp(x)` is very large.

**Parameters**

- **`arg`** : `Tensor` – The input tensor.
- **`axis`** : `int | None`, optional – The axis or axes along which to compute the `logsumexp`. If None (the
default), the operation is performed over all elements of the tensor.
- **`keep_dims`** : `bool`, optional, default: `False` – If True, the axes which are reduced are left in the result as
dimensions with size one. With this option, the result will broadcast
correctly against the input tensor. Defaults to False.

**Returns**

`Tensor` – An tensor containing the result of the `logsumexp` operation.

**Examples**

```python
>>> import nabla as nb
>>> x = nb.tensor([1.0, 2.0, 3.0])
>>> nb.logsumexp(x)
Tensor([3.407606], dtype=float32)
```

```python
>>> data = nb.tensor([[1, 2, 3], [4, 5, 6]])
>>> nb.logsumexp(data, axis=1)
Tensor([3.407606, 6.407606], dtype=float32)
```

---
## `where`

```python
def where(condition: nabla.core.tensor.Tensor, x: nabla.core.tensor.Tensor, y: nabla.core.tensor.Tensor) -> nabla.core.tensor.Tensor:
```
Selects elements from two tensors based on a condition.

This function returns an tensor with elements chosen from `x` where the
corresponding element in `condition` is True, and from `y` otherwise.
The function supports broadcasting among the three input tensors.

**Parameters**

- **`condition`** : `Tensor` – A boolean tensor. Where True, yield `x`, otherwise yield `y`.
- **`x`** : `Tensor` – The tensor from which to take values when `condition` is True.
- **`y`** : `Tensor` – The tensor from which to take values when `condition` is False.

**Returns**

`Tensor` – An tensor with elements from `x` and `y`, depending on `condition`.

**Examples**

```python
>>> import nabla as nb
>>> condition = nb.tensor([True, False, True])
>>> x = nb.tensor([1, 2, 3])
>>> y = nb.tensor([10, 20, 30])
>>> nb.where(condition, x, y)
Tensor([1, 20, 3], dtype=int32)
```

Broadcasting example:
```python
>>> nb.where(nb.tensor([True, False]), nb.tensor(5), nb.tensor([10, 20]))
Tensor([5, 20], dtype=int32)
```

---
## `cond`

```python
def cond(condition: nabla.core.tensor.Tensor, true_fn: collections.abc.Callable, false_fn: collections.abc.Callable, *args, **kwargs) -> nabla.core.tensor.Tensor:
```
Conditionally executes one of two functions.

If `condition` is True, `true_fn` is called; otherwise, `false_fn` is
called. This is a control-flow primitive that allows for conditional
execution within a computational graph. Unlike `nabla.where`, which
evaluates both branches, `cond` only executes the selected function.

**Parameters**

- **`condition`** : `Tensor` – A scalar boolean tensor that determines which function to execute.
- **`true_fn`** : `Callable` – The function to be called if `condition` is True.
- **`false_fn`** : `Callable` – The function to be called if `condition` is False.
- **`*args`** – Positional arguments to be passed to the selected function.
- **`**kwargs`** – Keyword arguments to be passed to the selected function.

**Returns**

`Tensor` – The result of calling either `true_fn` or `false_fn`.

**Examples**

```python
>>> import nabla as nb
>>> def f(x):
...     return x * 2
...
>>> def g(x):
...     return x + 10
...
>>> x = nb.tensor(5)
>>> # Executes g(x) because the condition is False
>>> nb.cond(nb.tensor(False), f, g, x)
Tensor([15], dtype=int32)
```

---
