# sjit

## Signature

```python
nabla.sjit(func: 'Callable[..., Any]' = None, show_graph: 'bool' = False) -> 'Callable[..., Any]'
```

## Description

Just-in-time compile a function for performance optimization with static=True.
This is a simplified alias for jit with static=True, which is useful for
functions that do not require dynamic shapes or inputs.

## Parameters

func: Function to optimize with JIT compilation (should take positional arguments)

## Returns

JIT-compiled function with optimized execution

## Examples

### Basic static JIT compilation

```python
import nabla as nb

def matrix_computation(x, y):
    return nb.matmul(x, y) + nb.sin(x)

# Static JIT compile the function
fast_func = nb.sjit(matrix_computation)
x = nb.randn((100, 100))
y = nb.randn((100, 100))
result = fast_func(x, y)
```

### Using as a decorator

```python
@nb.sjit
def neural_network_forward(params, x):
    """Forward pass through a neural network."""
    w1, b1, w2, b2 = params
    h1 = nb.tanh(nb.matmul(x, w1) + b1)
    output = nb.matmul(h1, w2) + b2
    return output

# First call compiles, subsequent calls reuse compilation
params = [nb.randn((10, 20)), nb.randn((20,)), nb.randn((20, 1)), nb.randn((1,))]
x = nb.randn((5, 10))
result = neural_network_forward(params, x)
```

## Notes

This follows JAX's jit API with static=True:
- Only accepts positional arguments
- For functions requiring keyword arguments, use functools.partial or lambda
- Supports both list-style (legacy) and unpacked arguments style (JAX-like)

