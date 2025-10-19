# ndarange_like

## Signature

```python
nabla.ndarange_like(template: 'Tensor') -> 'Tensor'
```

## Description

Creates an tensor with sequential values like a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor. It is filled with values from 0 to
N-1, where N is the total number of elements.

## Parameters

- **`template`** (`Tensor`): The template tensor to match properties from.

## Returns

- `Tensor`: A new tensor with the same properties as the template, filled with sequential values.

## Examples

```pycon
>>> import nabla as nb
>>> template = nb.zeros((2, 2), dtype=nb.DType.int32)
>>> nb.ndarange_like(template)
Tensor([[0, 1],
       [2, 3]], dtype=int32)
```
