# Autograd: Automatic Differentiation

[← Back to Core](../README.md)

> **Purpose**: Implements reverse-mode automatic differentiation via trace-based gradient computation.

## How Gradients Are Computed

Unlike PyTorch (which stores gradient tape per-tensor), Nabla uses **trace-based** autodiff:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Gradient Computation Flow                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  grad_fn = nabla.grad(loss_fn)                                              │
│  grads = grad_fn(params, x, y)                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: TRACE THE FORWARD PASS  (api.py → tracing.py)               │    │
│  │                                                                     │    │
│  │   t = trace(loss_fn, params, x, y)                                  │    │
│  │                                                                     │    │
│  │   What happens:                                                     │    │
│  │   • Mark input tensors as traced=True                               │    │
│  │   • Execute loss_fn normally                                        │    │
│  │   • Each operation records OpNode via _setup_output_refs        │    │
│  │   • trace.compute() walks backward from outputs via DFS             │    │
│  │   • Result: t.nodes = list of OpNode in topological order       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: REHYDRATE  (tracing.py → Trace.refresh_graph_values)                   │    │
│  │                                                                     │    │
│  │   t.refresh_graph_values()                                                     │    │
│  │                                                                     │    │
│  │   Why needed: Graph _values are epoch-scoped. After evaluate(),     │    │
│  │   they become stale. Rehydration restores them for current epoch.   │    │
│  │                                                                     │    │
│  │   How it works:                                                     │    │
│  │   1. Find all leaf tensors (no output_refs) → ensure realized       │    │
│  │   2. Add leaves to current graph epoch                              │    │
│  │   3. For each node in topological order:                            │    │
│  │      • Wrap TensorImpls as Tensors                                  │    │
│  │      • Call op.execute(args, ORIGINAL_kwargs)              │    │
│  │      • Map fresh _values back to original TensorImpls               │    │
│  │                                                                     │    │
│  │   Critical: execute receives original kwargs because       │    │
│  │   that's all we stored in OpNode. It adapts internally.         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: BACKWARD PASS  (utils.py → BackwardEngine)                  │    │
│  │                                                                     │    │
│  │   cotangent = ones_like(output)  # Initialize with 1.0              │    │
│  │   grads_map = backward_on_trace(t, cotangent)                       │    │
│  │                                                                     │    │
│  │   For each node in reversed(t.nodes):                               │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │ 3a. Skip if no vjp_rule or no output has cotangent         │    │    │
│  │   │                                                            │    │    │
│  │   │ 3b. Prepare VJP inputs:                                    │    │    │
│  │   │     primals = wrap(output_refs.op_args)                    │    │    │
│  │   │     outputs = wrap(output_refs.get_alive_outputs())        │    │    │
│  │   │     cotangents = [cotangent_map[id(out)] for out]          │    │    │
│  │   │                                                            │    │    │
│  │   │ 3c. Call VJP rule:                                         │    │    │
│  │   │     input_cotangents = op.vjp_rule(primals, cotangent, out)│    │    │
│  │   │     # Returns d(loss)/d(input) given d(loss)/d(output)     │    │    │
│  │   │                                                            │    │    │
│  │   │ 3d. Shape alignment (broadcasting):                        │    │    │
│  │   │     cot = _reduce_to_shape(cot, primal.shape)              │    │    │
│  │   │     # Sum over broadcasted dims to match primal shape      │    │    │
│  │   │                                                            │    │    │
│  │   │ 3e. Accumulate into cotangent_map:                         │    │    │
│  │   │     _accumulate_cotangent(cotangent_map, input_impl, cot)  │    │    │
│  │   │     # Handles sharding + addition for multi-use tensors    │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │   Result: cotangent_map[id(input_impl)] → gradient TensorImpl       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: FINALIZE  (utils.py → BackwardEngine._finalize)             │    │
│  │                                                                     │    │
│  │   gradients = {}                                                    │    │
│  │   for inp in trace.inputs:                                          │    │
│  │       grad = cotangent_map[id(inp._impl)]                           │    │
│  │                                                                     │    │
│  │       # Resolve partial sums (from sharded contracting dims)        │    │
│  │       if grad.sharding.partial_sum_axes:                            │    │
│  │           grad = all_reduce(grad, ...)                              │    │
│  │                                                                     │    │
│  │       # Reshard to match input sharding                             │    │
│  │       if needs_reshard(grad.sharding, inp.sharding):                │    │
│  │           grad = reshard(grad, inp.sharding)                        │    │
│  │                                                                     │    │
│  │       gradients[inp] = grad                                         │    │
│  │                                                                     │    │
│  │   return gradients                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## VJP Rules

Each operation implements `vjp_rule(primals, cotangent, output)`:

```python
# Example: Addition
def vjp_rule(self, primals, cotangent, output):
    x, y = primals
    # d(x+y)/dx = 1, d(x+y)/dy = 1
    # So d(loss)/dx = cotangent * 1 = cotangent
    return cotangent, cotangent

# Example: Multiplication
def vjp_rule(self, primals, cotangent, output):
    x, y = primals
    # d(x*y)/dx = y, d(x*y)/dy = x
    return cotangent * y, cotangent * x

# Example: Matmul (A @ B = C)
def vjp_rule(self, primals, cotangent, output):
    A, B = primals
    # d(A@B)/dA = cotangent @ B.T
    # d(A@B)/dB = A.T @ cotangent
    return cotangent @ transpose(B), transpose(A) @ cotangent
```

**Return convention**: Return tuple of cotangents matching input structure. Return `None` for non-differentiable inputs.

## Broadcasting Gradient Reduction

When inputs were broadcast, cotangent shape > primal shape. `_reduce_to_shape` handles this:

```python
def _reduce_to_shape(cot, target_shape):
    # 1. Reduce leading dims (rank mismatch)
    if len(cot.shape) > len(target_shape):
        cot = reduce_sum(cot, axis=list(range(diff)))
    
    # 2. Reduce internal broadcasted dims (size 1 in target)
    reduce_axes = [i for i, (c, t) in enumerate(zip(cot.shape, target_shape))
                   if t == 1 and c > 1]
    if reduce_axes:
        cot = reduce_sum(cot, axis=reduce_axes, keepdims=True)
    
    return cot
```

## Sharding-Aware Cotangent Accumulation

`_accumulate_cotangent` handles distributed gradients:

```python
def _accumulate_cotangent(cotangent_map, target_impl, cot_tensor):
    # 1. Resolve partial sums if target doesn't share them
    if cot_tensor.sharding.partial_sum_axes:
        if not target_impl.sharding.partial_sum_axes:
            cot_tensor = all_reduce(cot_tensor, ...)
    
    # 2. Reshard if target has different sharding
    if needs_reshard(cot_tensor.sharding, target_impl.sharding):
        cot_tensor = reshard(cot_tensor, target_impl.sharding)
    
    # 3. Add to existing or initialize
    if id(target_impl) in cotangent_map:
        existing = Tensor(impl=cotangent_map[id(target_impl)])
        cotangent_map[id(target_impl)] = add(existing, cot_tensor)._impl
    else:
        cotangent_map[id(target_impl)] = cot_tensor._impl
```

## API Reference

| Function                               | Purpose                                                    |
| :------------------------------------- | :--------------------------------------------------------- |
| `grad(fn, argnums=0)`                  | Returns function computing gradients w.r.t. specified args |
| `value_and_grad(fn, argnums=0)`        | Returns function computing both value and gradients        |
| `backward_on_trace(trace, cotangents)` | Core backward engine (used internally)                     |

## File Map

| File | Purpose |
| :--- | :--- |
| [api.py](api.py) | User-facing `grad`, `value_and_grad` |
| [utils.py](utils.py) | `BackwardEngine`, `backward_on_trace`, `_reduce_to_shape`, `_accumulate_cotangent` |

## Maintenance Guide

> **AI Agents - Critical Rules**:
>
> 1. **VJP signature**: `vjp_rule(primals, cotangent, output)` - note the order!
> 2. **Rehydration**: Always call `trace.refresh_graph_values()` before backward pass
> 3. **Shape alignment**: Always call `_reduce_to_shape` on VJP outputs
> 4. **Sharding**: Use `_accumulate_cotangent` which handles partial sums and resharding
