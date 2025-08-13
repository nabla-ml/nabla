# rand_like

## Signature

```python
nabla.rand_like(template: 'Array', lower: 'float', upper: 'float', seed: 'int') -> 'Array'
```

## Description

Creates an array with uniformly distributed random values like a template.

The new array will have the same shape, dtype, device, and batch
dimensions as the template array.

## Parameters

- **`template`** (`Array`): The template array to match properties from.

- **`lower`** (`float, optional`): The lower boundary of the output interval. Defaults to 0.0.

- **`upper`** (`float, optional`): The upper boundary of the output interval. Defaults to 1.0.

- **`seed`** (`int, optional`): The seed for the random number generator. Defaults to 0.

## Returns

- `Array`: A new array with the same properties as the template, filled with uniformly distributed random values.
