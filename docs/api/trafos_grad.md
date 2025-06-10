# grad

## Signature

```python
nabla.grad(fun: 'Callable' = None, argnums: 'int | Sequence[int]' = 0, has_aux: 'bool' = False, holomorphic: 'bool' = False, allow_int: 'bool' = False, reduce_axes: 'Sequence' = (), mode: 'str' = 'reverse') -> 'Callable[..., Any]'
```

## Description

Creates a function that evaluates the gradient of fun.

This is a convenience wrapper around jacrev/jacfwd that matches JAX's grad API.
By default uses reverse-mode autodiff (jacrev) but can be configured to use
forward-mode (jacfwd) via the mode parameter.


## Parameters

fun: Function to be differentiated. Should return a scalar.
argnums: Which positional argument(s) to differentiate with respect to (default 0).
has_aux: Whether fun returns (output, aux) pair (default False).
holomorphic: Whether fun is holomorphic - currently ignored (default False).
allow_int: Whether to allow integer inputs - currently ignored (default False).
reduce_axes: Axes to reduce over - currently ignored (default ()).
mode: "reverse" (default) for jacrev or "forward" for jacfwd.


## Returns

A function that computes the gradient of fun.


## Examples

### Basic gradient computation

```python
import nabla as nb

def loss_function(x):
    return nb.sum(x ** 2)

# Compute gradient function
grad_loss = nb.grad(loss_function)
x = nb.array([1.0, 2.0, 3.0])
gradient = grad_loss(x)
print(f"Gradient: {gradient}")  # [2.0, 4.0, 6.0]
```

### Using as a decorator

```python
@nb.grad
def quadratic_loss(x):
    return nb.sum(x ** 2)

x = nb.array([1.0, 2.0])
gradient = quadratic_loss(x)  # Returns gradient, not function value
```

### Forward-mode gradient

```python
# Use forward-mode for functions with few inputs, many outputs
grad_fn = nb.grad(my_function, mode="forward")
```

