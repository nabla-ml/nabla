# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from functools import partial

import jax.numpy as jnp
import pytest

import nabla as nb

from .common import OpConfig, Operation, run_unified_test, standard_get_args

OPS = {}

OPS["sum"] = Operation(
    "sum",
    "REDUCTION",
    nb.reduce_sum,
    jnp.sum,
    [
        OpConfig("Sum_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig(
            "Sum_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}
        ),
    ],
    standard_get_args,
)

OPS["mean"] = Operation(
    "mean",
    "REDUCTION",
    nb.mean,
    jnp.mean,
    [
        OpConfig("Mean_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig(
            "Mean_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}
        ),
    ],
    standard_get_args,
)

OPS["max"] = Operation(
    "max",
    "REDUCTION",
    nb.reduce_max,
    jnp.max,
    [
        OpConfig("Max_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig(
            "Max_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}
        ),
    ],
    standard_get_args,
)

OPS["min"] = Operation(
    "min",
    "REDUCTION",
    nb.reduce_min,
    jnp.min,
    [
        OpConfig("Min_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig(
            "Min_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}
        ),
    ],
    standard_get_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
@pytest.mark.parametrize("config_idx", [0, 1, 2])
def test_reduction_ops(op_name, config_idx):
    op = OPS[op_name]
    if config_idx >= len(op.configs):
        pass
        return

    config = op.configs[config_idx]
    run_unified_test(op, config)


from .common import (
    MESH_CONFIGS,
    DeviceMesh,
    get_sharding_configs,
    run_test_with_consistency_check,
)


@pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("mesh_cfg", MESH_CONFIGS)
def test_reduction_sharding_variance(op_name, mesh_cfg):
    """Test reduction ops on various mesh shapes and sharding configurations."""
    mesh_name, mesh_shape, mesh_axes = mesh_cfg
    mesh = DeviceMesh(mesh_name, mesh_shape, mesh_axes)

    op = OPS[op_name]

    config = OpConfig("Rank2", ranks=(2,), params={"axis": 0})
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)

    input_tensor = args_nb[0]
    rank = len(input_tensor.shape)

    specs = get_sharding_configs(mesh, rank)

    test_axes = [0, 1]

    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)

    for spec_idx, spec in enumerate(specs):
        if spec is None:
            continue

        for reduce_axis in test_axes:

            if reduce_axis >= rank:
                continue

            is_cross_shard = False
            if spec.dim_specs[reduce_axis].axes:
                is_cross_shard = True

            test_id = f"{op_name}_{mesh_name}_Spec{spec_idx}_Axis{reduce_axis}"

            def sharded_exec():
                t = args_nb[0]

                t_sharded = t.with_sharding(spec.mesh, spec.dim_specs)

                res = op.nabla_fn(t_sharded, axis=reduce_axis, keepdims=True)

                return op.nabla_fn(t_sharded, axis=reduce_axis, keepdims=False)

            def expected_exec():

                t_jax = args_jax[0]
                return op.jax_fn(t_jax, axis=reduce_axis, keepdims=False)

            run_test_with_consistency_check(test_id, sharded_exec, expected_exec)


# =============================================================================
# COMPREHENSIVE CROSS-SHARD REDUCTION TESTS
# These tests specifically exercise the auto-AllReduce mechanism that triggers
# when reducing over a sharded axis.
# =============================================================================

import numpy as np

from nabla.core.sharding.spec import DimSpec


def make_array(*shape: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def tensor_from_numpy(arr: np.ndarray) -> nb.Tensor:
    return nb.Tensor.from_dlpack(arr)


def assert_allclose(result: nb.Tensor, expected: np.ndarray, rtol: float = 1e-4):
    np.testing.assert_allclose(result.numpy(), expected, rtol=rtol, atol=1e-5)


REDUCTION_OPS = {
    "sum": (nb.reduce_sum, np.sum),
    "mean": (nb.mean, np.mean),
    "max": (nb.reduce_max, np.max),
    "min": (nb.reduce_min, np.min),
}

# Minimal configs for cross-shard tests (fast execution)
CROSS_SHARD_MESH_CONFIGS = [
    ("2x2", (2, 2), ("x", "y")),
    ("1x4", (1, 4), ("x", "y")),
]


class TestCrossShardReductions:
    """Test reductions where the reduce axis is the sharded axis (triggers AllReduce)."""

    @pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", CROSS_SHARD_MESH_CONFIGS)
    def test_reduce_on_sharded_axis0(self, op_name, mesh_name, mesh_shape, mesh_axes):
        """Reduce axis=0 when axis 0 is sharded (cross-shard reduction)."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        rows, cols = 8, 4
        mesh = DeviceMesh(f"mesh_{op_name}_{mesh_name}", mesh_shape, mesh_axes)

        np_x = make_array(rows, cols, seed=42)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec([])])

        result = nb_fn(x_sharded, axis=0, keepdims=False)
        expected = np_fn(np_x, axis=0)

        assert tuple(int(d) for d in result.shape) == (cols,)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", CROSS_SHARD_MESH_CONFIGS)
    def test_reduce_on_sharded_axis1(self, op_name, mesh_name, mesh_shape, mesh_axes):
        """Reduce axis=1 when axis 1 is sharded (cross-shard reduction)."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        rows, cols = 4, 8
        mesh = DeviceMesh(f"mesh_{op_name}_{mesh_name}", mesh_shape, mesh_axes)

        np_x = make_array(rows, cols, seed=123)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh, [DimSpec([]), DimSpec(["y"])])

        result = nb_fn(x_sharded, axis=1, keepdims=False)
        expected = np_fn(np_x, axis=1)

        assert tuple(int(d) for d in result.shape) == (rows,)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
    def test_reduce_on_sharded_axis_keepdims(self, op_name):
        """Reduce with keepdims=True on sharded axis."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        mesh = DeviceMesh("mesh_keepdims", (2, 2), ("x", "y"))

        np_x = make_array(8, 4, seed=99)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec([])])

        result = nb_fn(x_sharded, axis=0, keepdims=True)
        expected = np_fn(np_x, axis=0, keepdims=True)

        assert tuple(int(d) for d in result.shape) == (1, 4)
        assert_allclose(result, expected)


class TestMultiAxisShardedReductions:
    """Test reductions on tensors sharded on multiple axes (e.g., P("x", "y"))."""

    @pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
    def test_reduce_axis0_tensor_sharded_on_both_dims(self, op_name):
        """Reduce axis=0 when tensor is sharded on both axes."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        mesh = DeviceMesh("mesh_2axis", (2, 2), ("x", "y"))

        np_x = make_array(8, 8, seed=55)
        x = tensor_from_numpy(np_x)

        # Shard on both dimensions
        x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])

        result = nb_fn(x_sharded, axis=0, keepdims=False)
        expected = np_fn(np_x, axis=0)

        assert tuple(int(d) for d in result.shape) == (8,)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
    def test_reduce_axis1_tensor_sharded_on_both_dims(self, op_name):
        """Reduce axis=1 when tensor is sharded on both axes."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        mesh = DeviceMesh("mesh_2axis_ax1", (2, 2), ("x", "y"))

        np_x = make_array(8, 8, seed=77)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])

        result = nb_fn(x_sharded, axis=1, keepdims=False)
        expected = np_fn(np_x, axis=1)

        assert tuple(int(d) for d in result.shape) == (8,)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("mesh_name,mesh_shape,mesh_axes", [("2x4", (2, 4), ("x", "y")), ("4x2", (4, 2), ("x", "y"))])
    def test_reduce_asymmetric_multi_axis_mesh(self, mesh_name, mesh_shape, mesh_axes):
        """Test reduction on asymmetric 2D mesh with both axes sharded."""
        mesh = DeviceMesh(f"mesh_{mesh_name}", mesh_shape, mesh_axes)

        np_x = make_array(8, 16, seed=88)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])

        result = nb.reduce_sum(x_sharded, axis=0, keepdims=False)
        expected = np.sum(np_x, axis=0)

        assert tuple(int(d) for d in result.shape) == (16,)
        assert_allclose(result, expected)


class TestNegativeAxisReductions:
    """Test reductions with negative axis indices on sharded tensors."""

    @pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
    def test_reduce_negative_axis_minus1_sharded(self, op_name):
        """Reduce axis=-1 when last axis is sharded."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        mesh = DeviceMesh("mesh_neg1", (2, 2), ("x", "y"))

        np_x = make_array(4, 8, seed=101)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh, [DimSpec([]), DimSpec(["y"])])

        result = nb_fn(x_sharded, axis=-1, keepdims=False)
        expected = np_fn(np_x, axis=-1)

        assert tuple(int(d) for d in result.shape) == (4,)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
    def test_reduce_negative_axis_minus2_sharded(self, op_name):
        """Reduce axis=-2 on a 3D tensor with sharding."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        mesh = DeviceMesh("mesh_neg2", (2,), ("x",))

        np_x = make_array(4, 8, 4, seed=102)
        x = tensor_from_numpy(np_x)

        x_sharded = x.shard(mesh, [DimSpec([]), DimSpec(["x"]), DimSpec([])])

        result = nb_fn(x_sharded, axis=-2, keepdims=False)
        expected = np_fn(np_x, axis=-2)

        assert tuple(int(d) for d in result.shape) == (4, 4)
        assert_allclose(result, expected)


class TestVmapShardedReductions:
    """Test vmap(reduction) with sharded inputs inside the vmapped function."""

    @pytest.mark.parametrize("op_name", ["sum", "max"])
    def test_vmap_reduction_sharded_inner(self, op_name):
        """vmap(reduce_op) with sharding applied inside the vmapped function."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        batch, features = 4, 16
        mesh = DeviceMesh("mesh_vmap_inner", (2, 2), ("x", "y"))

        def f(x):
            x_sharded = x.shard(mesh, [DimSpec(["y"])])
            return nb_fn(x_sharded, axis=0)

        np_x = make_array(batch, features, seed=200)
        x = tensor_from_numpy(np_x)

        result = nb.vmap(f)(x)
        expected = np_fn(np_x, axis=1)

        assert tuple(int(d) for d in result.shape) == (batch,)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("op_name", ["sum", "max"])
    def test_vmap_reduction_keepdims_sharded(self, op_name):
        """vmap(reduce_op, keepdims=True) with sharding."""
        nb_fn, np_fn = REDUCTION_OPS[op_name]
        batch, features = 4, 8
        mesh = DeviceMesh("mesh_vmap_kd", (2,), ("tp",))

        def f(x):
            x_sharded = x.shard(mesh, [DimSpec(["tp"])])
            return nb_fn(x_sharded, axis=0, keepdims=True)

        np_x = make_array(batch, features, seed=201)
        x = tensor_from_numpy(np_x)

        result = nb.vmap(f)(x)
        expected = np_fn(np_x, axis=1, keepdims=True)

        assert tuple(int(d) for d in result.shape) == (batch, 1)
        assert_allclose(result, expected)

    def test_vmap_chained_reductions_different_ops(self):
        """vmap with chained reduce_sum then reduce_max."""
        batch = 4
        mesh = DeviceMesh("mesh_chain", (2,), ("tp",))

        def f(x):
            x_sharded = x.shard(mesh, [DimSpec([]), DimSpec(["tp"])])
            s = nb.reduce_sum(x_sharded, axis=0, keepdims=False)
            return nb.reduce_max(s, axis=0, keepdims=False)

        np_x = make_array(batch, 4, 8, seed=300)
        x = tensor_from_numpy(np_x)

        result = nb.vmap(f)(x)
        expected = np.max(np.sum(np_x, axis=1), axis=1)

        assert tuple(int(d) for d in result.shape) == (batch,)
        assert_allclose(result, expected)
