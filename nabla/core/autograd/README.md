# Autograd: Automatic Differentiation

[← Back to Core](../README.md)

## Philosophy

Reverse-mode automatic differentiation via trace-based gradient computation. Operates on captured computation graphs, not persistent per-tensor gradient buffers.

## Key Mechanisms

### Gradient Computation

1. **Forward Trace**: Function execution records operations in `Trace` object
2. **Initialize Cotangents**: Output gradients (∂L/∂output) provided by user or defaulted to 1.0
3. **Backward Pass**: Operations processed in reverse topological order
4. **VJP Application**: Each operation's `vjp()` computes input cotangents from output cotangents
5. **Accumulation**: Multi-use tensor cotangents summed across consumers

### API

- `grad(func)`: Returns function computing gradients w.r.t. inputs
- `value_and_grad(func)`: Returns both value and gradients
- `backward_on_trace(trace, output_grads)`: Core backward engine

### Operation Requirements

Each `Operation` implements `vjp(cotangents, *primals, **kwargs)` returning tuple of input cotangents (or `None` for non-differentiable inputs).

### Broadcasting Gradients

`_reduce_to_shape` handles reverse broadcasting: sum over leading dimensions and internal size-1 dimensions to match primal shapes.
