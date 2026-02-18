import nabla as nb
from tests.unit.common import make_jax_array, tensor_from_jax
from nabla.transforms.jvp import jvp
from nabla.transforms.utils import create_jacobian_helpers, std_basis, lift_basis_to_batch_prefix
from nabla.core.common.pytree import tree_flatten, tree_unflatten
from nabla.core.tensor.api import Tensor

x = tensor_from_jax(make_jax_array(5, seed=5))
f = lambda x: nb.reduce_sum(nb.exp(x))

def df_like(xarg):
    diff_args2, partial2 = create_jacobian_helpers(f, None, (xarg,))
    flat_inputs2, td2 = tree_flatten(diff_args2, is_leaf=lambda t: isinstance(t, Tensor))
    sizes2, basis2 = std_basis(flat_inputs2)
    basis2 = lift_basis_to_batch_prefix(basis2, flat_inputs2)
    prim2 = tree_unflatten(td2, list(flat_inputs2))
    if not isinstance(prim2, tuple):
        prim2 = (prim2,)

    rows = []
    for i in range(sum(sizes2)):
        tangents_flat = [b[i] for b in basis2]
        tang_tree = tree_unflatten(td2, tangents_flat)
        if not isinstance(tang_tree, tuple):
            tang_tree = (tang_tree,)

        _, t_out = jvp(partial2, prim2, tang_tree)
        tflat, _ = tree_flatten(t_out, is_leaf=lambda t: isinstance(t, Tensor))
        row = tflat[0]
        rows.append(row)
        print(
            "inner row",
            i,
            "shape",
            tuple(int(d) for d in row.shape),
            "batch_dims",
            row.batch_dims,
            "has_tangent",
            row.tangent is not None,
        )

    out = rows[0] if len(rows) == 1 else nb.stack(rows, axis=0)
    print("df_like out shape", tuple(int(d) for d in out.shape), "has_tangent", out.tangent is not None)
    return out

# Outer jvp direction uses first basis vector
flat_inputs, _ = tree_flatten((x,), is_leaf=lambda t: isinstance(t, Tensor))
_, tangent_basis = std_basis(flat_inputs)
outer_tangent = tangent_basis[0][0]

print("outer_tangent shape", tuple(int(d) for d in outer_tangent.shape), "batch_dims", outer_tangent.batch_dims)
out, tout = jvp(lambda z: df_like(z), (x,), (outer_tangent,))
print("outer out shape", tuple(int(d) for d in out.shape), "out has tangent", out.tangent is not None)
print("outer tangent_out shape", tuple(int(d) for d in tout.shape))
print("outer tangent_out", tout.to_numpy())
