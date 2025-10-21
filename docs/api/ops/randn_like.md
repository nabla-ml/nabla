# randn_like

## Signature

```python
nabla.randn_like(template: 'Tensor', mean: 'float' = 0.0, std: 'float' = 1.0, seed: 'int' = 0) -> 'Tensor'
```

**Source**: `nabla.ops.creation`

## Description

Creates an tensor with normally distributed random values like a template.

The new tensor will have the same shape, dtype, device, and batch
dimensions as the template tensor.

## Parameters

- **`template`** (`Tensor`): The template tensor to match properties from.

- **`mean`** (`float, optional`): The mean of the normal distribution. Defaults to 0.0.

- **`std`** (`float, optional`): The standard deviation of the normal distribution. Defaults to 1.0.

- **`seed`** (`int, optional`): The seed for the random number generator. Defaults to 0.

## Returns

Tensor
    A new tensor with the same properties as the template, filled with
    normally distributed random values.
