# ndarange_like

## Signature

```python
nabla.ndarange_like(template: 'Array') -> 'Array'
```

## Description

Creates an array with sequential values like a template array.

The new array will have the same shape, dtype, device, and batch
dimensions as the template array. It is filled with values from 0 to
N-1, where N is the total number of elements.

## Parameters

- **`template`** (`Array`): The template array to match properties from.

## Returns

- `Array`: A new array with the same properties as the template, filled with sequential values.

## Examples

```python
import nabla as nb
template = nb.zeros((2, 2), dtype=nb.DType.int32)
nb.ndarange_like(template)
Array([[0, 1],
       [2, 3]], dtype=int32)
```
