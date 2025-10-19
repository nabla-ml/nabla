# abs

## Signature

```python
nabla.abs(arg: 'Tensor') -> 'Tensor'
```

## Description

Computes the element-wise absolute value of an tensor.

## Parameters

- **`arg`** (`Tensor`): The input tensor.

## Returns

- `Tensor`: An tensor containing the absolute value of each element.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([-1.5, 0.0, 2.5])
>>> nb.abs(x)
Tensor([1.5, 0. , 2.5], dtype=float32)
```
