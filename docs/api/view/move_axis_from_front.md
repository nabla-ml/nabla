# move_axis_from_front

## Signature

```python
nabla.move_axis_from_front(input_array: 'Array', target_axis: 'int') -> 'Array'
```

## Description

Move front axis (position 0) to specified target position.

## Parameters

- **`input_array`** (`Input tensor (assumes front axis is the one to move)`): target_axis: Target position for the front axis

## Returns

- `Tensor with front axis moved to target position`: 

## Examples

```pycon
>>> x = nb.ones((4, 2, 3))  # front axis has size 4
    >>> y = move_axis_from_front(x, 2)  # shape (2, 3, 4)
    >>> # front axis moved to position 2: [1, 2, 0]
```
