# jit

## Signature

```python
nabla.jit(func: 'Callable[..., Any]' = None, static: 'bool' = False, show_graph: 'bool' = False) -> 'Callable[..., Any]'
```

## Description

Just-in-time compile a function for performance optimization.
This can be used as a function call like `jit(func)` or as a decorator `@jit`.


## Parameters

func: Function to optimize with JIT compilation (should take positional arguments)


## Returns

JIT-compiled function with optimized execution


## Examples

### Basic JIT compilation

```python
import nabla as nb

def expensive_computation(x):
    result = x
    for _ in range(100):
        result = nb.sin(result) + nb.cos(result)
    return result

# JIT compile the function
fast_func = nb.jit(expensive_computation)
x = nb.randn((1000,))
result = fast_func(x)
```

### Using as a decorator

```python
@nb.jit
def my_func(x):
    return x * 2 + nb.sin(x)

result = my_func(nb.array([1.0, 2.0, 3.0]))
```

### Static compilation for maximum performance

```python
@nb.jit(static=True)
def neural_network_layer(x, weights, bias):
    return nb.tanh(nb.matmul(x, weights) + bias)

# First call compiles, subsequent calls reuse compilation
output = neural_network_layer(x, w, b)
```

## Notes

This follows JAX's jit API:

* Only accepts positional arguments
* For functions requiring keyword arguments, use functools.partial or lambda
* Supports both list-style (legacy) and unpacked arguments style (JAX-like)


## See Also

- {doc}`vmap <trafos_vmap>` - Vectorization
- {doc}`grad <trafos_grad>` - Automatic differentiation

