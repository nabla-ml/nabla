# rand_like

## Signature

```python
nabla.rand_like(template: 'Tensor', lower: 'float' = 0.0, upper: 'float' = 1.0, seed: 'int' = 0) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

Creates an tensor with uniformly distributed random values like a template.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

Parameters
----------
template : Tensor
    The template tensor to match properties from.
lower : float, optional
    The lower boundary of the output interval. Defaults to 0.0.
upper : float, optional
    The upper boundary of the output interval. Defaults to 1.0.
seed : int, optional
    The seed for the random number generator. Defaults to 0.

Returns
-------
Tensor
    A new tensor with the same properties as the template, filled with
    uniformly distributed random values.

