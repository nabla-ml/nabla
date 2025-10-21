# vmap

## Signature

```python
nabla.vmap(func: collections.abc.Callable | None = None, in_axes: Union[int, NoneType, list, tuple] = 0, out_axes: Union[int, NoneType, list, tuple] = 0) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.vmap`

Creates a function that maps a function over axes of pytrees.

Parameters
----------
func : Callable or None
    Function to vectorize
in_axes : int or None or list or tuple, optional
    Specifies which axes to map over for inputs. Can be:
    - int: axis to map over (default 0)
    - None: broadcast (don't map)
    - list/tuple: per-input axis specification
out_axes : int or None or list or tuple, optional
    Specifies which axes to map over for outputs (default 0)

Returns
-------
Callable
    Vectorized function that maps func over the specified axes

Examples
--------
>>> import nabla as nb
>>> def square(x):
...     return x ** 2
>>> x = nb.tensor([[1.0, 2.0], [3.0, 4.0]])
>>> vmap_square = nb.vmap(square)
>>> result = vmap_square(x)

Multiple inputs with different axes:

>>> def multiply(x, y):
...     return x * y
>>> x = nb.tensor([[1.0, 2.0], [3.0, 4.0]])
>>> y = nb.tensor([10.0, 20.0])
>>> result = nb.vmap(multiply, in_axes=(0, None))(x, y)

As a decorator:

>>> @nb.vmap
... def process_batch(x):
...     return x ** 2 + 1
>>> batch = nb.tensor([1.0, 2.0, 3.0, 4.0])
>>> result = process_batch(batch)

