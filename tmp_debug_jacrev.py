"""Diagnostic: find the simplest function where jacrev produces wrong batch_dims.

We test increasingly simple functions and trace which op's VJP rule
produces outputs with incorrect batch_dims.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import nabla as nb
from nabla.core.tensor.api import Tensor

def T(arr):
    return nb.Tensor.from_dlpack(np.array(arr, dtype=np.float32))


# ── Test the simplest R^2 -> R case step by step ──
from nabla.transforms.vjp import vjp
from nabla.transforms.vmap import vmap
from nabla.transforms.utils import std_basis, lift_basis_to_batch_prefix
from nabla.core.common.pytree import tree_flatten, tree_unflatten

def f(x):
    return x[0] + x[1]

x = T([2.0, 3.0])
output, pullback = vjp(f, x)

flat_out, out_td = tree_flatten(output, is_leaf=lambda t: isinstance(t, Tensor))
sizes, basis = std_basis(flat_out)
basis = lift_basis_to_batch_prefix(basis, flat_out)

print(f"output: shape={output.shape} bd={output.batch_dims} phys={output._impl.physical_global_shape}")
for i, b in enumerate(basis):
    print(f"basis[{i}]: shape={b.shape} bd={b.batch_dims} phys={b._impl.physical_global_shape}")
print(f"sizes={sizes} total={sum(sizes)}")

# The jacrev code determines in_axes_spec based on basis shapes
# For scalar basis, all shapes are () or len=0 → in_axes_spec = None
in_axes_spec = None if all(b.shape == () or len(b.shape) == 0 for b in basis) else 0
print(f"in_axes_spec = {in_axes_spec}")

def pullback_flat(*cot_flat):
    cot_tree = tree_unflatten(out_td, list(cot_flat))
    return pullback(cot_tree)

# Call vmap
result = vmap(pullback_flat, in_axes=in_axes_spec)(*basis)

print(f"\nvmap result type: {type(result)}")
if isinstance(result, tuple):
    for i, r in enumerate(result):
        if isinstance(r, Tensor):
            print(f"  r[{i}]: shape={r.shape} bd={r.batch_dims} phys={r._impl.physical_global_shape}")
elif isinstance(result, Tensor):
    print(f"  result: shape={result.shape} bd={result.batch_dims} phys={result._impl.physical_global_shape}")

# Now show what _reshape_jacrev would do with these
print(f"\n--- _reshape_jacrev analysis ---")
single_arg = True
total_out = sum(sizes)
out_shape = tuple(int(d) for d in flat_out[0].shape)
print(f"out_shape={out_shape} total_out={total_out}")

# Get the grads
if isinstance(result, tuple):
    grads = result[0] if isinstance(result[0], tuple) else [result[0] if isinstance(result[0], Tensor) else result]
    grad = grads if isinstance(grads, Tensor) else result[0]
else:
    grad = result

print(f"grad shape={grad.shape} bd={grad.batch_dims} phys={grad._impl.physical_global_shape}")
print(f"grad.shape[1:] = {tuple(int(d) for d in grad.shape)[1:]}")
print(f"Would reshape {tuple(int(d) for d in grad.shape)} -> {tuple(int(d) for d in grad.shape)[1:]} ... this is the bug!")
