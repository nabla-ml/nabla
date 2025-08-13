# zeros_like

## Signature

```python
nabla.zeros_like(template: 'Array') -> 'Array'
```

## Description

Creates an array of zeros with the same properties as a template array.

The new array will have the same shape, dtype, device, and batch
dimensions as the template array.

## Parameters

- **`template`** (`Array`): The template array to match properties from.

## Returns

- `Array`: A new array of zeros with the same properties as the template.

## Examples

```python
import nabla as nb
x = nb.array([[1, 2], [3, 4]], dtype=nb.DType.int32)
nb.zeros_like(x)
Array([[0, 0],
       [0, 0]], dtype=int32)
```
