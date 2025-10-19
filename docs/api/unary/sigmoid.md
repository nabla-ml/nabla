# sigmoid

## Signature

```python
nabla.sigmoid(arg: 'Tensor') -> 'Tensor'
```

## Description

Computes the element-wise sigmoid function.

The sigmoid function, defined as `1 / (1 + exp(-x))`, is a common
activation function that squashes values to the range `(0, 1)`.

## Parameters

- **`arg`** (`Tensor`): The input tensor.

## Returns

- `Tensor`: An tensor containing the sigmoid of each element.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([-1.0, 0.0, 1.0, 20.0])
>>> nb.sigmoid(x)
Tensor([0.26894143, 0.5       , 0.7310586 , 1.        ], dtype=float32)
```
