# full_like

## Signature

```python
nabla.full_like(template: 'Tensor', fill_value: 'float') -> 'Tensor'
```

**Source**: `nabla.ops.creation`

## Description

Creates a filled tensor with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor, filled with `fill_value`.

## Parameters

- **`template`** (`Tensor`): The template tensor to match properties from.

- **`fill_value`** (`float`): The value to fill the new tensor with.

## Returns

- `Tensor`: A new tensor filled with `fill_value` and with the same properties as the template.

## Examples

```pycon
>>> import nabla as nb
>>> x = nb.zeros((2, 2))
>>> nb.full_like(x, 7.0)
Tensor([[7., 7.],
       [7., 7.]], dtype=float32)
```
