# ===----------------------------------------------------------------------=== #
# Nabla 2026 — Unified Op × Transform × Sharding Test Suite
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""
Systematically tests every op across transform levels and sharding configs.

Transform levels:
  L0  baseline       f(x)
  L1a vjp            vjp(f, *x) → pullback(ones) ~ grad
  L1b jvp            jvp(f, primals, tangents)
  L1c vmap           vmap(f)(batch_x)
  L2a vmap_vjp       vmap(grad_f)(batch_x)
  L2b vmap_jvp       vmap(jvp_f)(batch_x, batch_t)
  L3a jacrev         jacrev(f)(x)
  L3b jacfwd         jacfwd(f)(x)

Sharding configs:
  unsharded          no mesh
  sharded_axis0      first dim on mesh axis "x" with (2,4) mesh

Run examples:
  pytest tests/unit/test_unified.py -x -k "baseline"
  pytest tests/unit/test_unified.py -x -k "exp"
  pytest tests/unit/test_unified.py -x -k "vjp and unary"
  pytest tests/unit/test_unified.py -x -k "sharded"
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import pytest

import nabla as nb
from nabla import jacfwd, jacrev, jvp, vjp, vmap
from nabla.core.sharding.spec import DeviceMesh, DimSpec

from .common import (
    cleanup_caches,
    compare_nested_structures,
    get_shape_for_rank,
    tensor_from_jax,
)
from .unified_registry import (
    ALL_OPS,
    DIFF_OPS,
    UNARY_OPS,
)

SEED = 42
MESH = DeviceMesh("test", (2, 4), ("x", "y"))
TOLERANCE = 5e-4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_args(op, config):
    """Return (nabla_args, nabla_kwargs, jax_args, jax_kwargs)."""
    cleanup_caches()
    (args_nb, kw_nb), (args_jax, kw_jax) = op.get_args(config)
    kw_jax_fixed = {
        k.replace("axes", "axis") if op.name not in ("transpose", "unsqueeze") else k: v
        for k, v in kw_jax.items()
    }
    return args_nb, kw_nb, args_jax, kw_jax_fixed


def _call_nb(op, args, kw):
    return op.nabla_fn(*args, **kw)


def _call_jax(op, args, kw):
    return op.jax_fn(*args, **kw)


def _ones_like_jax(x):
    return jnp.ones_like(x)


def _is_scalar_output(result):
    if hasattr(result, "shape"):
        return result.shape == () or (hasattr(result, "ndim") and result.ndim == 0)
    return False


def _is_list_input(config):
    return getattr(config, "is_list_input", False)


def _batch_list_input(args_nb, args_jax):
    nb_list = [nb.stack([a, a]) for a in args_nb[0]]
    jax_list = [jnp.stack([a, a]) for a in args_jax[0]]
    in_axes = ([0] * len(nb_list),)
    return (nb_list,), (jax_list,), in_axes


# ---------------------------------------------------------------------------
# Build parametrize IDs
# ---------------------------------------------------------------------------


def _op_config_pairs():
    """Yield (id_str, op, config) for all ops × configs."""
    pairs = []
    for op in ALL_OPS:
        for config in op.configs:
            test_id = f"{op.name}_{config.description}"
            pairs.append(pytest.param(op, config, id=test_id))
    return pairs


def _unary_config_pairs():
    pairs = []
    for op in UNARY_OPS:
        for config in op.configs:
            pairs.append(pytest.param(op, config, id=f"{op.name}_{config.description}"))
    return pairs


def _differentiable_pairs():
    """All ops that are differentiable (skip comparison/creation/nondiff + where/gather/scatter)."""
    pairs = []
    nondiff_inputs = {"where", "gather", "scatter"}
    for op in DIFF_OPS:
        if op.name in nondiff_inputs:
            continue
        for config in op.configs:
            pairs.append(pytest.param(op, config, id=f"{op.name}_{config.description}"))
    return pairs


# ===========================================================================
# L0: Baseline forward pass
# ===========================================================================


class TestBaseline:
    @pytest.mark.parametrize("op,config", _op_config_pairs())
    def test_forward(self, op, config):
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        nb_res = _call_nb(op, args_nb, kw_nb)
        jax_res = _call_jax(op, args_jax, kw_jax)
        compare_nested_structures(nb_res, jax_res, tolerance=TOLERANCE)


# ===========================================================================
# L1a: VJP
# ===========================================================================


class TestVJP:
    @pytest.mark.parametrize("op,config", _differentiable_pairs())
    def test_vjp(self, op, config):
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)

        def nb_fn(*a):
            return op.nabla_fn(*a, **kw_nb)

        def jax_fn(*a):
            return op.jax_fn(*a, **kw_jax)

        primals_out_nb, pullback_nb = vjp(nb_fn, *args_nb)
        primals_out_jax, pullback_jax = jax.vjp(jax_fn, *args_jax)

        cot_jax = jax.tree.map(_ones_like_jax, primals_out_jax)
        cot_nb = jax.tree.map(
            lambda x: tensor_from_jax(jnp.ones_like(x)), primals_out_jax
        )

        grads_nb = pullback_nb(cot_nb)
        grads_jax = pullback_jax(cot_jax)

        for g_nb, g_jax in zip(grads_nb, grads_jax, strict=False):
            if g_nb is not None and g_jax is not None:
                compare_nested_structures(g_nb, g_jax, tolerance=TOLERANCE)


# ===========================================================================
# L1b: JVP
# ===========================================================================


class TestJVP:
    @pytest.mark.parametrize("op,config", _differentiable_pairs())
    def test_jvp(self, op, config):
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        list_input = _is_list_input(config)

        def nb_fn(*a):
            return op.nabla_fn(*a, **kw_nb)

        def jax_fn(*a):
            return op.jax_fn(*a, **kw_jax)

        if list_input:
            tangents_jax_list = [jnp.ones_like(a) for a in args_jax[0]]
            tangents_jax = (tangents_jax_list,)
            tangents_nb = ([tensor_from_jax(t) for t in tangents_jax_list],)
        else:
            tangents_jax = tuple(jnp.ones_like(a) for a in args_jax)
            tangents_nb = tuple(tensor_from_jax(t) for t in tangents_jax)

        primals_out_nb, tangents_out_nb = jvp(nb_fn, args_nb, tangents_nb)
        primals_out_jax, tangents_out_jax = jax.jvp(jax_fn, args_jax, tangents_jax)

        compare_nested_structures(primals_out_nb, primals_out_jax, tolerance=TOLERANCE)
        compare_nested_structures(
            tangents_out_nb, tangents_out_jax, tolerance=TOLERANCE
        )


# ===========================================================================
# L1c: vmap
# ===========================================================================


class TestVmap:
    @pytest.mark.parametrize("op,config", _op_config_pairs())
    def test_vmap(self, op, config):
        if not getattr(config, "supports_vmap", True):
            pytest.skip("vmap not supported for this op/config")
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        list_input = _is_list_input(config)

        if list_input:
            batched_nb, batched_jax, in_axes = _batch_list_input(args_nb, args_jax)
        else:
            in_axes = []
            batched_nb = []
            batched_jax = []
            for a_nb, a_jax in zip(args_nb, args_jax, strict=False):
                if hasattr(a_nb, "shape"):
                    batched_nb.append(nb.stack([a_nb, a_nb]))
                    batched_jax.append(jnp.stack([a_jax, a_jax]))
                    in_axes.append(0)
                else:
                    batched_nb.append(a_nb)
                    batched_jax.append(a_jax)
                    in_axes.append(None)

            in_axes = tuple(in_axes)
        nb_vmapped = vmap(partial(op.nabla_fn, **kw_nb), in_axes=in_axes)
        jax_vmapped = jax.vmap(partial(op.jax_fn, **kw_jax), in_axes=in_axes)

        nb_res = nb_vmapped(*batched_nb)
        jax_res = jax_vmapped(*batched_jax)
        compare_nested_structures(nb_res, jax_res, tolerance=TOLERANCE)


# ===========================================================================
# L2a: vmap(vjp)
# ===========================================================================


class TestVmapVJP:
    @pytest.mark.parametrize("op,config", _differentiable_pairs())
    def test_vmap_vjp(self, op, config):
        if not getattr(config, "supports_vmap", True):
            pytest.skip("vmap not supported for this op/config")
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        list_input = _is_list_input(config)

        def nb_grad_fn(*a):
            fn = lambda *x: op.nabla_fn(*x, **kw_nb)
            out, pb = vjp(fn, *a)
            cot = jax.tree.map(lambda x: nb.ones_like(x), out)
            return pb(cot)

        def jax_grad_fn(*a):
            fn = lambda *x: op.jax_fn(*x, **kw_jax)
            out, pb = jax.vjp(fn, *a)
            cot = jax.tree.map(_ones_like_jax, out)
            return pb(cot)

        if list_input:
            batched_nb, batched_jax, in_axes = _batch_list_input(args_nb, args_jax)
        else:
            in_axes = tuple(0 if hasattr(a, "shape") else None for a in args_nb)
            batched_nb = tuple(
                nb.stack([a, a]) if hasattr(a, "shape") else a for a in args_nb
            )
            batched_jax = tuple(
                jnp.stack([a, a]) if hasattr(a, "shape") else a for a in args_jax
            )

        nb_res = vmap(nb_grad_fn, in_axes=in_axes)(*batched_nb)
        jax_res = jax.vmap(jax_grad_fn, in_axes=in_axes)(*batched_jax)

        for g_nb, g_jax in zip(nb_res, jax_res, strict=False):
            if g_nb is not None and g_jax is not None:
                compare_nested_structures(g_nb, g_jax, tolerance=TOLERANCE)


# ===========================================================================
# L2b: vmap(jvp)
# ===========================================================================


class TestVmapJVP:
    @pytest.mark.parametrize("op,config", _differentiable_pairs())
    def test_vmap_jvp(self, op, config):
        if not getattr(config, "supports_vmap", True):
            pytest.skip("vmap not supported for this op/config")
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        list_input = _is_list_input(config)

        def nb_jvp_fn(*a):
            fn = lambda *x: op.nabla_fn(*x, **kw_nb)
            if list_input:
                tangents = ([nb.ones_like(x) for x in a[0]],)
            else:
                tangents = tuple(nb.ones_like(x) for x in a)
            _, t_out = jvp(fn, a, tangents)
            return t_out

        def jax_jvp_fn(*a):
            fn = lambda *x: op.jax_fn(*x, **kw_jax)
            if list_input:
                tangents = ([jnp.ones_like(x) for x in a[0]],)
            else:
                tangents = tuple(jnp.ones_like(x) for x in a)
            _, t_out = jax.jvp(fn, a, tangents)
            return t_out

        if list_input:
            batched_nb, batched_jax, in_axes = _batch_list_input(args_nb, args_jax)
        else:
            in_axes = tuple(0 if hasattr(a, "shape") else None for a in args_nb)
            batched_nb = tuple(
                nb.stack([a, a]) if hasattr(a, "shape") else a for a in args_nb
            )
            batched_jax = tuple(
                jnp.stack([a, a]) if hasattr(a, "shape") else a for a in args_jax
            )

        nb_res = vmap(nb_jvp_fn, in_axes=in_axes)(*batched_nb)
        jax_res = jax.vmap(jax_jvp_fn, in_axes=in_axes)(*batched_jax)
        compare_nested_structures(nb_res, jax_res, tolerance=TOLERANCE)


# ===========================================================================
# L3a: jacrev
# ===========================================================================


class TestJacrev:
    @pytest.mark.parametrize("op,config", _unary_config_pairs())
    def test_jacrev(self, op, config):
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        if len(args_nb) != 1:
            pytest.skip("jacrev single-input only")

        fn_nb = lambda x: op.nabla_fn(x, **kw_nb)
        fn_jax = lambda x: op.jax_fn(x, **kw_jax)

        jac_nb = jacrev(fn_nb)(args_nb[0])
        jac_jax = jax.jacrev(fn_jax)(args_jax[0])
        compare_nested_structures(jac_nb, jac_jax, tolerance=TOLERANCE)


# ===========================================================================
# L3b: jacfwd
# ===========================================================================


class TestJacfwd:
    @pytest.mark.parametrize("op,config", _unary_config_pairs())
    def test_jacfwd(self, op, config):
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        if len(args_nb) != 1:
            pytest.skip("jacfwd single-input only")

        fn_nb = lambda x: op.nabla_fn(x, **kw_nb)
        fn_jax = lambda x: op.jax_fn(x, **kw_jax)

        jac_nb = jacfwd(fn_nb)(args_nb[0])
        jac_jax = jax.jacfwd(fn_jax)(args_jax[0])
        compare_nested_structures(jac_nb, jac_jax, tolerance=TOLERANCE)


# ===========================================================================
# Sharded variants: L0 + L1a + L1b with (2,4) mesh, axis-0 sharding
# ===========================================================================


def _shard_axis0(t: nb.Tensor, mesh: DeviceMesh) -> nb.Tensor:
    """Shard a tensor's first dim on mesh axis 'x'."""
    if len(t.shape) == 0 or int(t.shape[0]) % mesh.shape[0] != 0:
        return t
    rank = len(t.shape)
    specs = [DimSpec(["x"], is_open=False)] + [
        DimSpec([], is_open=True) for _ in range(rank - 1)
    ]
    return t.shard(mesh, specs)


def _maybe_shard_arg(op, idx: int, arg: nb.Tensor, mesh: DeviceMesh) -> nb.Tensor:
    if op.name == "matmul" and idx == 1:
        return arg
    return _shard_axis0(arg, mesh)


def _maybe_shard_arg_tree(op, idx: int, arg, mesh: DeviceMesh):
    if isinstance(arg, list):
        return [
            _maybe_shard_arg(op, idx, a, mesh) if hasattr(a, "shape") else a
            for a in arg
        ]
    return _maybe_shard_arg(op, idx, arg, mesh) if hasattr(arg, "shape") else arg


def _shardable_pairs(ops=None):
    """Ops × configs where first dim is divisible by 2 (mesh axis 'x' size)."""
    if ops is None:
        ops = ALL_OPS
    pairs = []
    for op in ops:
        for config in op.configs:
            shapes = config.primal_shapes or tuple(
                get_shape_for_rank(r) for r in config.ranks
            )
            all_shardable = all(len(s) > 0 and s[0] % 2 == 0 for s in shapes)
            if all_shardable:
                pairs.append(
                    pytest.param(op, config, id=f"{op.name}_{config.description}")
                )
    return pairs


def _shardable_differentiable_pairs():
    """Shardable ops that are also differentiable (skip where/gather/scatter + non-diff)."""
    nondiff_inputs = {"where", "gather", "scatter"}
    ops = [op for op in DIFF_OPS if op.name not in nondiff_inputs]
    return _shardable_pairs(ops)


class TestShardedBaseline:
    @pytest.mark.parametrize("op,config", _shardable_pairs())
    def test_sharded_forward(self, op, config):
        if not getattr(config, "supports_sharding", True):
            pytest.skip("sharding not supported for this op/config")
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        sharded_nb = tuple(
            _maybe_shard_arg_tree(op, i, a, MESH) for i, a in enumerate(args_nb)
        )
        nb_res = op.nabla_fn(*sharded_nb, **kw_nb)
        jax_res = op.jax_fn(*args_jax, **kw_jax)
        compare_nested_structures(nb_res, jax_res, tolerance=TOLERANCE)


class TestShardedVJP:
    @pytest.mark.parametrize("op,config", _shardable_differentiable_pairs())
    def test_sharded_vjp(self, op, config):
        if not getattr(config, "supports_sharding", True):
            pytest.skip("sharding not supported for this op/config")
        args_nb, kw_nb, args_jax, kw_jax = _get_args(op, config)
        sharded_nb = tuple(
            _maybe_shard_arg_tree(op, i, a, MESH) for i, a in enumerate(args_nb)
        )

        def nb_fn(*a):
            return op.nabla_fn(*a, **kw_nb)

        def jax_fn(*a):
            return op.jax_fn(*a, **kw_jax)

        out_nb, pb_nb = vjp(nb_fn, *sharded_nb)
        out_jax, pb_jax = jax.vjp(jax_fn, *args_jax)

        cot_jax = jax.tree.map(_ones_like_jax, out_jax)
        cot_nb = jax.tree.map(lambda x: tensor_from_jax(jnp.ones_like(x)), out_jax)

        grads_nb = pb_nb(cot_nb)
        grads_jax = pb_jax(cot_jax)

        for g_nb, g_jax in zip(grads_nb, grads_jax, strict=False):
            if g_nb is not None and g_jax is not None:
                compare_nested_structures(g_nb, g_jax, tolerance=TOLERANCE)
