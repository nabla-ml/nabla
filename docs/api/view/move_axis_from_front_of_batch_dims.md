# move_axis_from_front_of_batch_dims

## Signature

```python
nabla.move_axis_from_front_of_batch_dims(input_array: 'Array', target_axis: 'int') -> 'Array'
```

## Description

Move front batch dimension (position 0) to specified target position.

## Parameters

- **`input_array`** (`Input tensor with batch dimensions (assumes front batch dim is the one to move)`): target_axis: Target position for the front batch dimension (negative index)

## Returns

- `Tensor with front batch dimension moved to target position`: 

## Examples

```python
x = nb.ones((4, 2, 3))  # shape (4, 2, 3)
    >>> x.batch_dims = (0, 1)  # Simulated for example
    >>> y = move_axis_from_front_of_batch_dims(x, -1)  # Move front batch dim to last position
    >>> # Result has batch_dims=(1, 0) and shape=(4, 2, 3)
```
