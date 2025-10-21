# softmax

## Signature

```python
nabla.softmax(arg: nabla.core.tensor.Tensor, axis: int = -1) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.special`

Computes the softmax function for an tensor.

The softmax function transforms a vector of real numbers into a probability
distribution. Each element in the output is in the range (0, 1), and the
elements along the specified axis sum to 1. It is calculated in a
numerically stable way as `exp(x - logsumexp(x))`.

Parameters
----------
arg : Tensor
    The input tensor.
axis : int, optional
    The axis along which the softmax computation is performed. The default
    is -1, which is the last axis.

Returns
-------
Tensor
    An tensor of the same shape as the input, containing the softmax
    probabilities.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([1.0, 2.0, 3.0])
>>> nb.softmax(x)
Tensor([0.09003057, 0.24472848, 0.66524094], dtype=float32)

>>> logits = nb.tensor([[1, 2, 3], [1, 1, 1]])
>>> nb.softmax(logits, axis=1)
Tensor([[0.09003057, 0.24472848, 0.66524094],
       [0.33333334, 0.33333334, 0.33333334]], dtype=float32)

