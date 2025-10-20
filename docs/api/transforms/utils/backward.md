# backward

## Signature

```python
nabla.backward(outputs: 'Any', cotangents: 'Any', retain_graph: 'bool' = False) -> 'None'
```

**Source**: `nabla.transforms.utils`

## Description

Accumulate gradients on traced leaf inputs for the given traced outputs.

Args:
    outputs: Output tensors to backpropagate from
    cotangents: Cotangent vectors for outputs
    retain_graph: If False (default), frees the computation graph after backward pass
