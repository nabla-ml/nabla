# ones_like

## Signature

```python
nabla.ones_like(template: 'Array') -> 'Array'
```

## Description

Creates an array of ones with the same properties as a template array.

The new array will have the same shape, dtype, device, and batch
dimensions as the template array.

## Parameters

- **`template`** (`Array`): The template array to match properties from.

## Returns

- `Array`: A new array of ones with the same properties as the template.

## Examples

```python
>>> import nabla as nb
>>> x = nb.array([[1., 2.], [3., 4.]])
>>> nb.ones_like(x)
Array([[1., 1.],
       [1., 1.]], dtype=float32)
```
