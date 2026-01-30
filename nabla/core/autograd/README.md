# Autograd: Automatic Differentiation

[← Back to Core](../README.md)

## Philosophy

The autograd system implements reverse-mode automatic differentiation (backpropagation) through trace-based gradient computation. It operates on captured computation graphs rather than maintaining persistent per-tensor gradient buffers.

## Architecture

### Trace-Based Differentiation

Gradients are computed by transforming a forward trace into a backward pass:

1. **Forward Pass**: User function executes, recording operations in a `Trace` object
2. **Gradient Initialization**: Output cotangents (∂L/∂output) provided by user or defaulted to 1.0
3. **Backward Traversal**: Operations processed in reverse topological order
4. **VJP Application**: Each operation's `vjp` method computes input cotangents from output cotangents
5. **Accumulation**: Cotangents for multi-use tensors are summed across consumer operations

### Core Functions

**`grad(func)`**: Returns a function computing gradients of `func`'s outputs w.r.t. its inputs.

```python
def grad(func):
    def grad_fn(*args, **kwargs):
        trace_obj = trace(func)(*args, **kwargs)
        cotangents = backward_on_trace(trace_obj, ...)
        return cotangents
    return grad_fn
```

**`value_and_grad(func)`**: Returns both function value and gradients.

**`backward_on_trace(t, output_grads)`**: Core engine that walks the trace backward, invoking each operation's VJP.

### Gradient Reduction

The `_reduce_to_shape` utility handles broadcasting in reverse:

1. **Rank Reduction**: Sum over leading dimensions added by broadcasting
2. **Dimension Reduction**: Sum over internal dimensions where primal had size 1

This ensures cotangents match primal shapes despite forward-pass broadcasting.

### Operation Requirements

Each `Operation` subclass must implement:

**`vjp(cotangents, *primals, **kwargs)`**: Vector-Jacobian product computing input gradients from output gradients. Returns tuple of cotangents matching primal inputs, or `None` for non-differentiable inputs.

## Implementation Details

**Module Structure**:
- `api.py`: Public API (`grad`, `value_and_grad`, `backward`)
- `utils.py`: Core backward engine (`backward_on_trace`, `_reduce_to_shape`)

**Integration with Execution**:
- Autograd operates on traced graphs, not live tensors
- No persistent `.grad` attributes; gradients computed on-demand
- Compatible with sharding: each shard's gradients computed independently via SPMD

**Limitations**:
- Higher-order gradients require nested trace transformations
- Dynamic control flow (Python `if`/`while`) not differentiable without tracing
- Mutations not supported in differentiable code paths
