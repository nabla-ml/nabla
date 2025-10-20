# vmap

## Signature

```python
nabla.vmap(func: collections.abc.Callable | None = None, in_axes: Union[int, NoneType, list, tuple] = 0, out_axes: Union[int, NoneType, list, tuple] = 0) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.vmap`

## Description

Creates a function that maps a function over axes of pytrees.
