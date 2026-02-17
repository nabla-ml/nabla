# Automatic Differentiation

## `backward`

```python
def backward(outputs: 'Any', cotangents: 'Any' = None, *, create_graph: 'bool' = False) -> 'None':
```
PyTorch-style backward pass that populates .grad on requires_grad tensors.

This function:
1. Builds a Trace from outputs using compute_for_backward()
   - Traverses through all OpNodes back to true leaves.
   - Collects all tensors with requires_grad=True as gradient leaves.
2. Runs VJP on the trace.
3. Populates .grad attributes on the collected gradient leaves.
4. Batch-realizes all gradients for efficiency.


---
