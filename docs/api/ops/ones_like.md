# ones_like

## Signature

```python
nabla.ones_like(template: 'Tensor') -> 'Tensor'
```

**Source**: `nabla.ops.creation`

## Description

Creates an tensor of ones with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

## Parameters

- **`template`** (`Tensor`): The template tensor to match properties from.

## Returns

- `Tensor`: A new tensor of ones with the same properties as the template.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.tensor([[1., 2.], [3., 4.]])
>>> nb.ones_like(x)
Tensor([[1., 1.],
       [1., 1.]], dtype=float32)
```
