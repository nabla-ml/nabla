# randn_like

## Signature

```python
nabla.randn_like(template: 'Array', mean: 'float', std: 'float', seed: 'int') -> 'Array'
```

## Description

Creates an array with normally distributed random values like a template.

The new array will have the same shape, dtype, device, and batch
dimensions as the template array.

## Parameters

- **`template`** (`Array`): The template array to match properties from.

- **`mean`** (`float, optional`): The mean of the normal distribution. Defaults to 0.0.

- **`std`** (`float, optional`): The standard deviation of the normal distribution. Defaults to 1.0.

- **`seed`** (`int, optional`): The seed for the random number generator. Defaults to 0.

## Returns

- `Array`: A new array with the same properties as the template, filled with normally distributed random values.
