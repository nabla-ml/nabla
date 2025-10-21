# move_axis_to_front_of_batch_dims

## Signature

```python
nabla.move_axis_to_front_of_batch_dims(input_tensor: nabla.core.tensor.Tensor, axis: int) -> nabla.core.tensor.Tensor
```

**Source**: `nabla.ops.view`

Move specified batch dimension to the front (position 0), shifting others right.

Parameters
----------
    input_tensor: Input tensor with batch dimensions
    axis: Batch dimension to move to front (negative index)

Returns
-------
    Tensor with specified batch dimension moved to front

Examples
--------

.. code-block:: python

    >>> x = nb.ones((2, 3, 4))  # shape (2, 3, 4)
    >>> x.batch_dims = (1, 0)  # Simulated for example
    >>> y = move_axis_to_fron_of_batch_dims(x, -1)  # Move last batch dim to front
    >>> # Result has batch_dims=(0, 1) and shape=(2, 3, 4)

