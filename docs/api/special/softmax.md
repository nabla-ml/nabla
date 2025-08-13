# softmax

## Signature

```python
nabla.softmax(arg: 'Array', axis: 'int') -> 'Array'
```

## Description

Computes the softmax function for an array.

The softmax function transforms a vector of real numbers into a probability
distribution. Each element in the output is in the range (0, 1), and the
elements along the specified axis sum to 1. It is calculated in a
numerically stable way as `exp(x - logsumexp(x))`.

## Parameters

- **`arg`** (`Array`): The input array.

- **`axis`** (`int, optional`): The axis along which the softmax computation is performed. The default is -1, which is the last axis.

## Returns

- `Array`: An array of the same shape as the input, containing the softmax probabilities.

## Examples

```python
import nabla as nb
x = nb.array([1.0, 2.0, 3.0])
nb.softmax(x)
Array([0.09003057, 0.24472848, 0.66524094], dtype=float32)

logits = nb.array([[1, 2, 3], [1, 1, 1]])
nb.softmax(logits, axis=1)
Array([[0.09003057, 0.24472848, 0.66524094],
       [0.33333334, 0.33333334, 0.33333334]], dtype=float32)
```
