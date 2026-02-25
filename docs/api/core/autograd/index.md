# Autograd Runtime

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
## `forward`

```python
def forward(trace: 'Trace', tangents: 'Any', *, create_graph: 'bool' = True) -> 'TangentMap':
```
Pure-function forward-mode AD on a Trace.

Analogous to backward_on_trace but propagates tangents forward.

**Parameters**

- **`trace`** – Captured computation trace.
- **`tangents`** – Tangent vectors for trace inputs (same pytree structure).
- **`create_graph`** – If True, tangent ops are traced for higher-order AD.

**Returns**

**`TangentMap`** – dict mapping output TensorImpl → tangent TensorImpl.


---
