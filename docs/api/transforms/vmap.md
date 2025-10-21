# vmap

## Signature

```python
nabla.vmap(func: collections.abc.Callable | None = None, in_axes: Union[int, NoneType, list, tuple] = 0, out_axes: Union[int, NoneType, list, tuple] = 0) -> collections.abc.Callable[..., typing.Any]
```

**Source**: `nabla.transforms.vmap`

Creates a function that maps a function over axes of pytrees.

Args:
    func: Function to vectorize
    in_axes: Specifies which axes to map over for inputs. Can be:
        - int: axis to map over (default 0)
        - None: broadcast (don't map)
        - list/tuple: per-input axis specification
    out_axes: Specifies which axes to map over for outputs (default 0)

Returns:
    Vectorized function that maps func over the specified axes

Examples:
    Basic vectorization:

    ```python
    import nabla as nb

    def square(x):
        return x ** 2

    # Map over first axis (batch dimension)
    x = nb.tensor([[1.0, 2.0], [3.0, 4.0]])
    vmap_square = nb.vmap(square)
    result = vmap_square(x)
    # result = [[1.0, 4.0], [9.0, 16.0]]
    ```

    Multiple inputs with different axes:

    ```python
    import nabla as nb

    def multiply(x, y):
        return x * y

    x = nb.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    y = nb.tensor([10.0, 20.0])              # shape (2,)
    
    # Map x over axis 0, broadcast y
    result = nb.vmap(multiply, in_axes=(0, None))(x, y)
    ```

    Batch matrix multiplication:

    ```python
    import nabla as nb

    def matvec(matrix, vector):
        return matrix @ vector

    matrices = nb.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    vectors = nb.tensor([[1, 1], [2, 2]])                        # (2, 2)
    
    # Batch process both inputs
    results = nb.vmap(matvec)(matrices, vectors)
    ```

    As a decorator:

    ```python
    import nabla as nb

    @nb.vmap
    def process_batch(x):
        return x ** 2 + 1

    batch = nb.tensor([1.0, 2.0, 3.0, 4.0])
    result = process_batch(batch)
    ```

