# zeros_like

## Signature

```python
nabla.zeros_like(template: 'Tensor') -> 'Tensor'
```

**Source**: `nabla.ops.creation`

Creates an tensor of zeros with the same properties as a template tensor.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

Parameters
----------
template : Tensor
    The template tensor to match properties from.

Returns
-------
Tensor
    A new tensor of zeros with the same properties as the template.

Examples
--------
>>> import nabla as nb
>>> x = nb.tensor([[1, 2], [3, 4]], dtype=nb.DType.int32)
>>> nb.zeros_like(x)
Tensor([[0, 0],
       [0, 0]], dtype=int32)

