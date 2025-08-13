# tanh

## Signature

```python
nabla.tanh(arg: 'Array') -> 'Array'
```

## Description

Computes the element-wise hyperbolic tangent of an array.

The tanh function is a common activation function in neural networks,
squashing values to the range `[-1, 1]`.

## Parameters

- **`arg`** (`Array`): The input array.

## Returns

- `Array`: An array containing the hyperbolic tangent of each element.

## Examples

```python
>>> import nabla as nb
>>> x = nb.array([-1.0, 0.0, 1.0, 20.0])
>>> nb.tanh(x)
Array([-0.7615942,  0.       ,  0.7615942,  1.       ], dtype=float32)
```
