# move_axis_to_front

## Signature

```python
nabla.move_axis_to_front(input_tensor: nabla.core.tensor.Tensor, axis: int) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.view`

Move specified axis to the front (position 0), shifting others right.

Parameters
----------
    input_tensor: Input tensor
    axis: Axis to move to front

Returns
-------
    Tensor with specified axis moved to front

Examples
--------
    >>> x = nb.ones((2, 3, 4))  # shape (2, 3, 4)
    >>> y = move_axis_to_front(x, 2)  # shape (4, 2, 3)
    >>> # axis 2 moved to front, others shifted: [2, 0, 1]

