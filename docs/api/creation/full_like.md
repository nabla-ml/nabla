# full_like

## Signature

```python
nabla.full_like(template: 'Array', fill_value: 'float') -> 'Array'
```

## Description

Creates a filled array with the same properties as a template array.

The new array will have the same shape, dtype, device, and batch
dimensions as the template array, filled with `fill_value`.

## Parameters

- **`template`** (`Array`): The template array to match properties from.

- **`fill_value`** (`float`): The value to fill the new array with.

## Returns

- `Array`: A new array filled with `fill_value` and with the same properties as the template.

## Examples

```python
import nabla as nb
x = nb.zeros((2, 2))
nb.full_like(x, 7.0)
Array([[7., 7.],
       [7., 7.]], dtype=float32)
```
