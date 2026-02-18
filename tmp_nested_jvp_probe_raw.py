import nabla as nb
import importlib
from tests.unit.common import make_jax_array, tensor_from_jax
from nabla.transforms.utils import create_jacobian_helpers, std_basis, lift_basis_to_batch_prefix
from nabla.core.common.pytree import tree_flatten, tree_unflatten
from nabla.core.tensor.api import Tensor

jvp_mod = importlib.import_module("nabla.transforms.jvp")

x = tensor_from_jax(make_jax_array(5, seed=5))
f = lambda x: nb.reduce_sum(nb.exp(x))

def inner_jvp_raw(xarg, basis_idx: int):
    diff_args2, partial2 = create_jacobian_helpers(f, None, (xarg,))
    flat_inputs2, td2 = tree_flatten(diff_args2, is_leaf=lambda t: isinstance(t, Tensor))
    sizes2, basis2 = std_basis(flat_inputs2)
    basis2 = lift_basis_to_batch_prefix(basis2, flat_inputs2)
    prim2 = tree_unflatten(td2, list(flat_inputs2))
    if not isinstance(prim2, tuple):
        prim2 = (prim2,)

    tangents_flat = [b[basis_idx] for b in basis2]
    tang_tree = tree_unflatten(td2, tangents_flat)
    if not isinstance(tang_tree, tuple):
        tang_tree = (tang_tree,)

    saved = jvp_mod._save_and_attach_tangents(prim2, tang_tree)
    try:
        raw_output = partial2(*prim2)
    finally:
        jvp_mod._restore_tangents(saved)

    out, _aux = jvp_mod.split_aux(raw_output, False, name='jvp-probe')
    out_flat, _ = tree_flatten(out, is_leaf=lambda t: isinstance(t, Tensor))
    y = out_flat[0]
    y_tan = Tensor(impl=y._impl.tangent) if y._impl.tangent is not None else None

    print('primal output y shape', tuple(int(d) for d in y.shape), 'y has tangent', y._impl.tangent is not None)
    if y_tan is not None:
        print('y_tan shape', tuple(int(d) for d in y_tan.shape), 'y_tan has tangent', y_tan.tangent is not None)

    extracted = jvp_mod._extract_tangents(out)
    eflat, _ = tree_flatten(extracted, is_leaf=lambda t: isinstance(t, Tensor))
    e = eflat[0]
    print('extracted shape', tuple(int(d) for d in e.shape), 'extracted has tangent', e.tangent is not None)

# outer jvp direction uses first basis vector
flat_inputs, _ = tree_flatten((x,), is_leaf=lambda t: isinstance(t, Tensor))
_, tangent_basis = std_basis(flat_inputs)
outer_tangent = tangent_basis[0][0]

saved_outer = jvp_mod._save_and_attach_tangents((x,), (outer_tangent,))
try:
    print('x has tangent (outer)?', x.tangent is not None)
    inner_jvp_raw(x, 0)
finally:
    jvp_mod._restore_tangents(saved_outer)
