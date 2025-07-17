# unsqueeze_batch_dims

## Signature

```python
nabla.unsqueeze_batch_dims(arg: nabla.core.array.Array, axes: list[int] | None = None) -> nabla.core.array.Array
```

## Description

Unsqueeze array by adding batch dimensions of size 1.


## Parameters

arg: Input array
axes: List of positions where to insert batch dimensions of size 1.
If None, returns array unchanged.


## Returns

Array with batch dimensions of size 1 added at specified positions

