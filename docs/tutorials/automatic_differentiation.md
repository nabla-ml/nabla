# Automatic Differentiation

Nabla provides powerful automatic differentiation capabilities through several transformation functions. This tutorial covers the core AD functions: `grad`, `vjp`, and `jvp`.

## The `grad` Function

The `grad` function computes gradients using reverse-mode automatic differentiation:

```python
import nabla as nb

def simple_function(x):
    return nb.sum(x ** 2)

# Create gradient function
grad_fn = nb.grad(simple_function)

# Compute gradient
x = nb.array([1.0, 2.0, 3.0])
gradient = grad_fn(x)
print(f"Gradient: {gradient}")  # [2.0, 4.0, 6.0]
```

### Multi-argument Functions

```python
def multi_arg_function(x, y):
    return nb.sum(x * y + x ** 2)

# Gradient with respect to first argument (default)
grad_wrt_x = nb.grad(multi_arg_function)
x, y = nb.array([1.0, 2.0]), nb.array([3.0, 4.0])
dx = grad_wrt_x(x, y)

# Gradient with respect to specific argument
grad_wrt_y = nb.grad(multi_arg_function, argnums=1)
dy = grad_wrt_y(x, y)

print(f"Gradient w.r.t x: {dx}")
print(f"Gradient w.r.t y: {dy}")
```

### Higher-order Derivatives

```python
def polynomial(x):
    return x ** 4 + 2 * x ** 3 - 3 * x ** 2 + x

# First derivative
first_derivative = nb.grad(polynomial)

# Second derivative (gradient of gradient)
second_derivative = nb.grad(first_derivative)

# Third derivative
third_derivative = nb.grad(second_derivative)

x = nb.array([2.0])
print(f"f'(2) = {first_derivative(x)}")
print(f"f''(2) = {second_derivative(x)}")
print(f"f'''(2) = {third_derivative(x)}")
```

## Vector-Jacobian Products (VJP)

The `vjp` function computes both the function value and returns a function to compute vector-Jacobian products:

```python
def vector_function(x):
    return nb.array([x[0] ** 2 + x[1], x[0] * x[1] ** 2])

x = nb.array([2.0, 3.0])

# VJP returns (output, vjp_function)
output, vjp_fn = nb.vjp(vector_function, x)

# Compute VJP with a cotangent vector
cotangent = nb.array([1.0, 1.0])
gradient = vjp_fn(cotangent)

print(f"Function output: {output}")
print(f"VJP result: {gradient}")
```

### Practical VJP Example: Neural Network Layer

```python
def linear_layer(x, weights, bias):
    return nb.matmul(x, weights) + bias

def relu_layer(x, weights, bias):
    return nb.relu(linear_layer(x, weights, bias))

# Sample data
x = nb.randn((32, 10))  # Batch of 32, input dim 10
W = nb.randn((10, 5))   # Weights
b = nb.zeros((5,))      # Bias

# Forward pass with VJP
output, vjp_fn = nb.vjp(lambda W, b: relu_layer(x, W, b), W, b)

# Backward pass (assuming unit cotangent)
cotangent = nb.ones_like(output)
dW, db = vjp_fn(cotangent)

print(f"Output shape: {output.shape}")
print(f"Weight gradient shape: {dW.shape}")
print(f"Bias gradient shape: {db.shape}")
```

## Jacobian-Vector Products (JVP)

The `jvp` function implements forward-mode automatic differentiation:

```python
def vector_function(x):
    return nb.array([x[0] ** 2 + x[1], x[0] * x[1] ** 2])

x = nb.array([2.0, 3.0])
tangent = nb.array([1.0, 0.0])  # Direction vector

# JVP returns (output, jvp_result)
output, jvp_result = nb.jvp(vector_function, (x,), (tangent,))

print(f"Function output: {output}")
print(f"JVP result: {jvp_result}")
```

### When to Use JVP vs VJP

**Use VJP (reverse-mode) when:**
- You have many inputs, few outputs (common in ML)
- Computing gradients for optimization
- Backpropagation in neural networks

**Use JVP (forward-mode) when:**
- You have few inputs, many outputs
- Computing directional derivatives
- Sensitivity analysis

```python
# Example: Many inputs → few outputs (use VJP)
def loss_function(params):  # params could be millions of parameters
    # ... complex computation ...
    return scalar_loss  # Single output

grad_loss = nb.grad(loss_function)  # Efficient with VJP

# Example: Few inputs → many outputs (use JVP)  
def physics_simulation(initial_conditions):  # Few parameters
    # ... simulation ...
    return state_vector  # Many outputs (position, velocity, etc.)

# Sensitivity to initial conditions
output, sensitivity = nb.jvp(physics_simulation, (initial_conditions,), (perturbation,))
```

## Combining Transformations

You can compose different AD transformations:

```python
def quadratic_form(x, A):
    return nb.sum(x * nb.matmul(A, x))

# Gradient of gradient (Hessian-vector product)
def hvp(f, x, v):
    """Hessian-vector product using grad composition"""
    return nb.jvp(nb.grad(f), (x,), (v,))[1]

# Example usage
x = nb.array([1.0, 2.0, 3.0])
A = nb.array([[2.0, -1.0, 0.0],
              [-1.0, 2.0, -1.0], 
              [0.0, -1.0, 2.0]])
v = nb.array([1.0, 0.0, 0.0])

hessian_vector_product = hvp(lambda x: quadratic_form(x, A), x, v)
print(f"Hessian-vector product: {hessian_vector_product}")
```

## Practical Example: Gradient Descent

```python
def rosenbrock(x):
    """The Rosenbrock function - a classic optimization test case"""
    return nb.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def gradient_descent(f, x0, learning_rate=0.01, num_steps=1000):
    """Simple gradient descent optimizer"""
    grad_f = nb.grad(f)
    x = x0
    
    for i in range(num_steps):
        g = grad_f(x)
        x = x - learning_rate * g
        
        if i % 100 == 0:
            loss = f(x)
            print(f"Step {i}: loss = {loss:.6f}")
    
    return x

# Optimize the Rosenbrock function
x0 = nb.array([0.0, 0.0])
x_opt = gradient_descent(rosenbrock, x0, learning_rate=0.001, num_steps=5000)
print(f"Optimized point: {x_opt}")
print(f"Function value: {rosenbrock(x_opt)}")
```

## Memory and Performance Considerations

### Gradient Checkpointing

For memory-efficient training of large models:

```python
def expensive_function(x):
    # Simulate a memory-intensive computation
    for _ in range(10):
        x = nb.sin(x) + nb.cos(x)
    return nb.sum(x ** 2)

# Standard gradient computation (high memory)
standard_grad = nb.grad(expensive_function)

# Memory-efficient gradient with checkpointing
# (Note: Implementation details may vary)
def checkpointed_grad(f):
    def grad_fn(x):
        # This would implement gradient checkpointing
        # Trading computation for memory
        return nb.grad(f)(x)
    return grad_fn

efficient_grad = checkpointed_grad(expensive_function)
```

## Common Patterns and Best Practices

### 1. Loss Function Design

```python
def mse_loss(predictions, targets):
    """Mean squared error loss"""
    return nb.mean((predictions - targets) ** 2)

def cross_entropy_loss(logits, labels):
    """Cross-entropy loss with numerical stability"""
    log_softmax = logits - nb.log(nb.sum(nb.exp(logits), axis=-1, keepdims=True))
    return -nb.mean(nb.sum(labels * log_softmax, axis=-1))
```

### 2. Parameter Updates

```python
def sgd_update(params, gradients, learning_rate):
    """Stochastic gradient descent update"""
    return [p - learning_rate * g for p, g in zip(params, gradients)]

def adam_update(params, gradients, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer update"""
    new_m = [beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m, gradients)]
    new_v = [beta2 * vi + (1 - beta2) * gi ** 2 for vi, gi in zip(v, gradients)]
    
    # Bias correction
    m_hat = [mi / (1 - beta1 ** t) for mi in new_m]
    v_hat = [vi / (1 - beta2 ** t) for vi in new_v]
    
    # Parameter update
    new_params = [p - lr * mh / (nb.sqrt(vh) + eps) 
                  for p, mh, vh in zip(params, m_hat, v_hat)]
    
    return new_params, new_m, new_v
```

## Next Steps

- Learn about {doc}`vectorization_and_jit` for performance optimization
- Check out the {doc}`../examples/index` for practical examples
