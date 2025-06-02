---
title: Advanced Topics
---

This tutorial covers advanced topics in the Nabla library that extend beyond the basics.

## Higher-order Derivatives

Nabla supports computing higher-order derivatives through repeated application of differentiation operators:

```python
import nabla as nb
import numpy as np

def f(x):
    return nb.sin(x)

# First derivative
df_dx = nb.grad(f)

# Second derivative
d2f_dx2 = nb.grad(df_dx)

# Evaluate at a specific point
x = nb.array(np.pi/4)
print(f"First derivative at pi/4: {df_dx(x)}")
print(f"Second derivative at pi/4: {d2f_dx2(x)}")
```

## Custom Operations

You can extend Nabla with custom operations:

```python
import nabla as nb
from nabla.ops import Operation

class MyCustomOp(Operation):
    def forward(self, x):
        # Implementation of forward pass
        return x * x
        
    def backward(self, x, grad_output):
        # Implementation of backward pass (gradient)
        return 2 * x * grad_output

# Use the custom operation
my_op = MyCustomOp()
result = my_op(nb.array([1.0, 2.0, 3.0]))
```

## Optimization Techniques

Nabla can be used for various optimization problems:

```python
import nabla as nb
import numpy as np

# Define a simple loss function
def loss_fn(params):
    x, y = params
    return (x - 2)**2 + (y - 3)**2

# Gradient function
grad_fn = nb.grad(loss_fn)

# Simple gradient descent
params = nb.array([0.0, 0.0])
learning_rate = 0.1

for _ in range(100):
    gradients = grad_fn(params)
    params = params - learning_rate * gradients
    
print(f"Optimized parameters: {params}")
```
