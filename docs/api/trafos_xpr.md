# xpr

## Signature

```python
nabla.xpr(fn: 'Callable[..., Any]', *primals) -> 'str'
```

## Description

Get a JAX-like string representation of the function's computation graph.


## Parameters

fn: Function to trace (should take positional arguments)
*primals: Positional arguments to the function (can be arbitrary pytrees)


## Returns

JAX-like string representation of the computation graph


## Examples

### Basic computation graph visualization

```python
import nabla as nb

def simple_function(x):
    return x ** 2 + nb.sin(x)

x = nb.array([1.0, 2.0, 3.0])
graph_str = nb.xpr(simple_function, x)
print(graph_str)
```

### Complex function with multiple operations

```python
def neural_network_layer(x, weights, bias):
    return nb.tanh(nb.matmul(x, weights) + bias)

x = nb.randn((5, 10))
weights = nb.randn((10, 20))
bias = nb.randn((20,))

# Visualize the computation graph
graph = nb.xpr(neural_network_layer, x, weights, bias)
print("Neural network computation graph:")
print(graph)
```

### Understanding transformations

```python
def loss_function(params, data, targets):
    predictions = nb.sum(params * data, axis=1)
    return nb.mean((predictions - targets) ** 2)

params = nb.randn((10,))
data = nb.randn((32, 10))
targets = nb.randn((32,))

# See what the computation graph looks like
computation_graph = nb.xpr(loss_function, params, data, targets)
print("Loss function graph:")
print(computation_graph)
```

## Notes

This follows the same flexible API as vjp, jvp, and vmap:
- Accepts functions with any number of positional arguments
- For functions requiring keyword arguments, use functools.partial or lambda
- Useful for debugging and understanding computation graphs
- Provides JAX-compatible string representation

## See Also

- {doc}`jit <trafos_jit>` - Just-in-time compilation
- {doc}`vjp <trafos_vjp>` - Vector-Jacobian product
- {doc}`grad <trafos_grad>` - Gradient computation

