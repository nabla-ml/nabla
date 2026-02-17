# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import jax.dlpack
import numpy as np  # Kept for np.prod compatibility if needed, or replace with math
import pytest
from max.dtype import DType

import nabla as nb
from nabla import vmap, Tensor
from nabla.core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec

nb.DType = DType
SEED = 42

MESH_CONFIGS = [
    ("1x2", (1, 2), ("x", "y")),
    ("2x1", (2, 1), ("x", "y")),
    ("2x2", (2, 2), ("x", "y")),
    ("1x4", (1, 4), ("x", "y")),
    ("4x1", (4, 1), ("x", "y")),
]


def make_jax_array(*shape: int, seed: int = 42, dtype=jnp.float32) -> jax.Array:
    """Create a deterministic random JAX array."""
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape, dtype=dtype)


def make_positive_jax_array(
    *shape: int, seed: int = 42, dtype=jnp.float32
) -> jax.Array:
    """Create a deterministic positive random JAX array."""
    key = jax.random.PRNGKey(seed)
    return jnp.abs(jax.random.normal(key, shape, dtype=dtype)) + 0.1


def tensor_from_jax(arr: jax.Array) -> Tensor:
    """Create a nabla Tensor from a JAX array using Zero-Copy DLPack."""
    arr.block_until_ready()
    return Tensor.from_dlpack(arr)


def to_jax(t: Tensor) -> jax.Array:
    """Convert Nabla Tensor to JAX array using Zero-Copy DLPack."""
    return jnp.from_dlpack(t)


def assert_allclose(
    result: Tensor, expected: jax.Array, rtol: float = 1e-5, atol: float = 1e-6
):
    """Assert tensor values match expected JAX array using DLPack conversion."""
    actual_jax = to_jax(result)
    import numpy as np

    np.testing.assert_allclose(actual_jax, expected, rtol=rtol, atol=atol)


def assert_dtype(result: Tensor, expected_dtype):
    """Assert tensor dtype matches expected."""
    assert result.dtype == expected_dtype, (
        f"Dtype mismatch: got {result.dtype}, expected {expected_dtype}"
    )


def assert_batch_dims(result: Tensor, expected: int):
    """Assert tensor batch_dims matches expected."""
    actual = result.batch_dims
    assert actual == expected, f"batch_dims mismatch: got {actual}, expected {expected}"


def assert_shape(result: Tensor, expected_shape: tuple):
    """Assert tensor.shape matches expected (logical shape)."""
    actual = tuple(int(d) for d in result.shape)
    assert actual == expected_shape, (
        f"Shape mismatch: got {actual}, expected {expected_shape}"
    )


def assert_physical_shape(result: Tensor, expected_shape: tuple):
    """Assert tensor's physical shape (global_shape) matches expected."""
    actual = result.global_shape or result.local_shape
    actual = tuple(int(d) for d in actual)
    assert actual == expected_shape, (
        f"Physical shape mismatch: got {actual}, expected {expected_shape}"
    )


def assert_is_sharded(result: Tensor, expected: bool = True):
    """Assert tensor is/isn't sharded."""
    actual = result.is_sharded
    assert actual == expected, f"is_sharded mismatch: got {actual}, expected {expected}"


def shard_on_axis(
    tensor: Tensor, mesh: DeviceMesh, axis: int, mesh_axis: int = 0
) -> Tensor:
    """Shard tensor on a specific axis using specified mesh dimension."""
    rank = len(tensor.shape)

    specs = [DimSpec([], is_open=True) for _ in range(rank)]

    specs[axis] = DimSpec([mesh.axis_names[mesh_axis]], is_open=False)
    return tensor.shard(mesh, specs)


def replicated(tensor: Tensor, mesh: DeviceMesh) -> Tensor:
    """Create a fully replicated sharded tensor."""
    rank = len(tensor.shape)
    specs = [DimSpec([], is_open=True) for _ in range(rank)]
    return tensor.shard(mesh, specs)


@dataclass
class OpConfig:
    """A unified configuration for a single test scenario."""

    description: str
    params: dict = field(default_factory=dict)
    ranks: tuple[int, ...] | None = None
    primal_shapes: tuple[tuple[int, ...], ...] | None = None
    is_list_input: bool = False
    domain_positive: bool = False
    input_dtype: str = "float32"
    use_stable_floats: bool = False
    supports_vmap: bool = True
    supports_sharding: bool = True


@dataclass
class Operation:
    """Encapsulates a logical operation and its testing metadata."""

    name: str
    category: str
    nabla_fn: Callable
    jax_fn: Callable
    configs: list[OpConfig]
    get_args: Callable[[OpConfig], tuple[tuple, tuple]]


def jax_transpose_wrapper(x, axis1, axis2):
    rank = len(x.shape)
    if rank < 2:
        return x
    axes = list(range(rank))
    a1, a2 = (
        (axis1 if axis1 >= 0 else rank + axis1),
        (axis2 if axis2 >= 0 else rank + axis2),
    )
    axes[a1], axes[a2] = axes[a2], axes[a1]
    return jnp.transpose(x, axes=axes)


def jax_expand_dims_wrapper(x, axes):
    res = x
    for axis in sorted(axes):
        res = jnp.expand_dims(res, axis=axis)
    return res


def jax_squeeze_wrapper(x, axis=None, axes=None):
    ax = axis if axis is not None else axes
    if ax is None:
        return jnp.squeeze(x)
    return jnp.squeeze(x, axis=tuple(ax) if isinstance(ax, (list, tuple)) else ax)


def jax_split_wrapper(x, num_splits=None, axis=0, **kwargs):
    if num_splits is None:
        num_splits = kwargs.get("split_size_or_sections")
    return jnp.split(x, num_splits, axis=axis)


def jax_matmul_wrapper(x, y):
    x_rank, y_rank = len(x.shape), len(y.shape)
    if x_rank == 1 and y_rank == 1:
        return jnp.dot(x, y)
    if x_rank == 1:
        return jnp.expand_dims(x, 0) @ y
    if y_rank == 1:
        return jnp.squeeze(x @ jnp.expand_dims(y, 1), -1)
    return jnp.matmul(x, y)


def jax_slice_wrapper(x, slices):
    return x[tuple(slices)]


def jax_pad_inverse_slice(x, slices, target_shape):
    res = jnp.zeros(target_shape, dtype=x.dtype)
    res = res.at[tuple(slices)].set(x)
    return res


def jax_unsqueeze_wrapper(x, axis=None, axes=None):
    res = x
    ax = axis if axis is not None else axes
    if ax is None:
        return x

    if isinstance(ax, int):
        return jnp.expand_dims(res, axis=ax)

    for a in sorted(ax):
        res = jnp.expand_dims(res, axis=a)
    return res


def get_shape_for_rank(rank: int) -> tuple[int, ...]:
    shapes = {
        0: (),
        1: (8,),
        2: (4, 4),
        3: (4, 2, 4),
        4: (2, 2, 2, 2),
    }
    return shapes.get(rank, (2,) * rank)


def get_test_data_for_shapes(shapes, config: OpConfig):
    # np.random.seed(SEED) # No longer needed, using JAX keys
    nabla_primals = []
    jax_primals = []
    key = jax.random.PRNGKey(SEED)

    for i, shape in enumerate(shapes):
        key, subkey = jax.random.split(key)
        num_elements = int(np.prod(shape)) if shape else 1

        if config.input_dtype == "bool":
            jax_base = jnp.arange(num_elements)
            # Create boolean mask
            jax_val = (jax_base.reshape(shape) % 2 == 0) if shape else jnp.array(True)

            # Create Nabla Tensor from JAX array via DLPack
            nb_val = tensor_from_jax(jax_val)
        else:
            if not shape:
                base_val = 2.5 if config.domain_positive else 1.5
                jax_val = jnp.array(base_val, dtype="float32")
                nb_val = tensor_from_jax(jax_val)
            else:
                # Use JAX arange
                jax_base = jnp.arange(num_elements, dtype="float32")

                offset = 1.0 if config.domain_positive else float(i + 1)
                jax_val = (jax_base + offset).reshape(shape)

                if not config.use_stable_floats:
                    jax_val *= 0.1

                nb_val = tensor_from_jax(jax_val)

        nabla_primals.append(nb_val)
        jax_primals.append(jax_val)
    return tuple(nabla_primals), tuple(jax_primals)


def standard_get_args(config: OpConfig):
    shapes = config.primal_shapes or tuple(get_shape_for_rank(r) for r in config.ranks)
    primals_nb, primals_jax = get_test_data_for_shapes(shapes, config)
    return (primals_nb, config.params), (primals_jax, config.params)


def cleanup_caches():
    jax.clear_caches()
    if hasattr(nb, "_clear_caches"):
        nb._clear_caches()


def compare_nested_structures(nb_res, jax_res, path="", tolerance=5e-4):
    """Recursively compare arbitrary nested structures (tuples, lists, dicts)."""

    if hasattr(nb_res, "numpy"):
        # Convert Nabla Tensor to JAX array via DLPack
        nb_val_jax = to_jax(nb_res)
        jax_val = jnp.array(jax_res)  # Ensure expected is JAX array

        if nb_val_jax.shape != jax_val.shape:
            # Special handling for interleaved complex comparison
            # Case 1: (..., 2) vs (...) complex
            if (
                nb_val_jax.ndim == jax_val.ndim + 1
                and nb_val_jax.shape[-1] == 2
                and nb_val_jax.shape[:-1] == jax_val.shape
            ):
                # Convert interleaved to complex
                nb_val_jax = nb_val_jax[..., 0] + 1j * nb_val_jax[..., 1]
            # Case 2: (..., 1, 2) vs (...) complex (observed behavior)
            elif (
                nb_val_jax.ndim == jax_val.ndim + 2
                and nb_val_jax.shape[-2:] == (1, 2)
                and nb_val_jax.shape[:-2] == jax_val.shape
            ):
                nb_val_jax = nb_val_jax[..., 0, 0] + 1j * nb_val_jax[..., 0, 1]

        try:
            # Use np.testing.assert_allclose which handles JAX arrays nicely
            np.testing.assert_allclose(
                nb_val_jax,
                jax_val,
                rtol=tolerance,
                atol=tolerance,
                err_msg=f"Mismatch at {path}",
            )
        except AssertionError as e:
            if hasattr(nb_res, "is_complex") and nb_res.is_complex():
                # Handle complex comparison if needed
                pass
            raise e
        return

    if isinstance(nb_res, (tuple, list)) and isinstance(jax_res, (tuple, list)):
        assert len(nb_res) == len(jax_res), (
            f"Length mismatch at {path}: {len(nb_res)} vs {len(jax_res)}"
        )
        for i, (n, j) in enumerate(zip(nb_res, jax_res, strict=False)):
            compare_nested_structures(n, j, path=f"{path}[{i}]", tolerance=tolerance)
        return

    if isinstance(nb_res, dict) and isinstance(jax_res, dict):
        assert nb_res.keys() == jax_res.keys(), f"Key mismatch at {path}"
        for k in nb_res:
            compare_nested_structures(
                nb_res[k], jax_res[k], path=f"{path}.{k}", tolerance=tolerance
            )
        return

    if nb_res is None or jax_res is None:
        assert nb_res is jax_res, f"None mismatch at {path}"
        return

    try:
        nb_val = np.array(nb_res)
        jax_val = np.array(jax_res)
        np.testing.assert_allclose(nb_val, jax_val, rtol=tolerance, atol=tolerance)
    except Exception:
        assert nb_res == jax_res, f"Value mismatch at {path}: {nb_res} != {jax_res}"


def run_test_with_consistency_check(test_name: str, nabla_fn_lazy, jax_fn_eager):
    """Executes Nabla (lazy) and JAX (eager) functions and compares results."""

    try:
        jax_res = jax_fn_eager()
    except Exception as e:
        pytest.fail(f"[{test_name}] JAX execution failed: {e}")

    try:
        nabla_res = nabla_fn_lazy()
    except Exception as e:
        pytest.fail(f"[{test_name}] Nabla execution failed: {e}")

    try:
        compare_nested_structures(nabla_res, jax_res)
    except AssertionError as e:
        pytest.fail(f"[{test_name}] Value Mismatch: {e}")


def get_sharding_configs(mesh: DeviceMesh, rank: int) -> list[ShardingSpec | None]:
    """Generate interesting sharding specs for a tensor of given rank."""
    configs = [None]

    from nabla.core.sharding.spmd import create_replicated_spec

    configs.append(create_replicated_spec(mesh, rank))

    if rank > 0:
        configs.append(ShardingSpec(mesh, (("x",),) + ((),) * (rank - 1)))

    if rank > 0:
        configs.append(ShardingSpec(mesh, ((),) * (rank - 1) + (("y",),)))

    return configs


def run_vmap_check(test_name, op, config, args_nb, kw_nb, args_jax, kw_jax):
    """Checks automatic vectorization (vmap)."""

    def batchify(x):
        if hasattr(x, "shape"):
            return nb.stack([x, x]) if isinstance(x, nb.Tensor) else jnp.stack([x, x])
        return x

    try:
        if config.is_list_input:
            return
        else:
            in_axes_list = []
            batched_args_nb_list = []
            batched_args_jax_list = []

            for arg_nb, arg_jax in zip(args_nb, args_jax, strict=False):
                is_tensor = hasattr(arg_nb, "shape") and not isinstance(
                    arg_nb, (tuple, list)
                )

                if is_tensor:
                    batched_args_nb_list.append(batchify(arg_nb))
                    batched_args_jax_list.append(batchify(arg_jax))
                    in_axes_list.append(0)
                else:
                    batched_args_nb_list.append(arg_nb)
                    batched_args_jax_list.append(arg_jax)
                    in_axes_list.append(None)

            batched_args_nb = tuple(batched_args_nb_list)
            batched_args_jax = tuple(batched_args_jax_list)
            in_axes = tuple(in_axes_list)

        nb_vmapped = vmap(partial(op.nabla_fn, **kw_nb), in_axes=in_axes)
        nb_res = nb_vmapped(*batched_args_nb)

        kw_jax_fixed = {
            (
                k.replace("axes", "axis")
                if op.name not in ["transpose", "unsqueeze"]
                else k
            ): v
            for k, v in kw_jax.items()
        }
        jax_vmapped = jax.vmap(partial(op.jax_fn, **kw_jax_fixed), in_axes=in_axes)
        jax_res = jax_vmapped(*batched_args_jax)

        compare_nested_structures(nb_res, jax_res, path=f"{test_name}.vmap")

    except Exception as e:
        pytest.fail(f"[{test_name}] VMap check failed: {e}")


def run_sharding_check(test_name, op, config, args_nb, kw_nb):
    """Checks sharding propagation."""

    mesh_shape = (2, 2)
    mesh = DeviceMesh("test_mesh", mesh_shape, ("x", "y"))

    def shard_arg(x):
        if isinstance(x, nb.Tensor) and len(x.shape) > 0:
            if int(x.shape[0]) % mesh_shape[0] == 0:
                spec = ShardingSpec(mesh, (("x",),) + ((),) * (len(x.shape) - 1))
                return x.shard(mesh, spec.dim_specs)
        return x

    sharded_args_nb = []
    if config.is_list_input:
        sharded_args_nb.append([shard_arg(x) for x in args_nb])
    else:
        sharded_args_nb = [shard_arg(x) for x in args_nb]

    try:
        res = op.nabla_fn(*sharded_args_nb, **kw_nb)

        if isinstance(res, nb.Tensor) and res.sharding is not None:
            assert res.sharding.mesh == mesh, "Result mesh mismatch"

    except Exception as e:
        raise e


def assert_spec(tensor: nb.Tensor, expected_dims: tuple[tuple[str, ...], ...]):
    """Assert tensor's sharding spec matches expected dimensions."""
    if tensor.sharding is None:
        assert expected_dims is None, "Expected sharding spec but got None"
        return

    actual_dims = tuple(tuple(ds.axes) for ds in tensor.sharding.dim_specs)

    actual_dims = tuple(tuple(a for a in d) for d in actual_dims)
    expected_dims = tuple(tuple(a for a in d) for d in expected_dims)

    assert actual_dims == expected_dims, (
        f"Sharding spec mismatch: got {actual_dims}, expected {expected_dims}"
    )


def run_unified_test(op: Operation, config: OpConfig, suffix: str = ""):
    """Master runner."""
    test_name = f"{op.name}_{config.description}{suffix}"

    # Move cleanup_caches() here so it's called BEFORE args_nb are created.
    # This ensures arg creation uses the same fresh context as the execution.
    cleanup_caches()

    (args_nb, kw_nb), (args_jax, kw_jax) = op.get_args(config)

    if config.is_list_input:
        nb_inp = [list(args_nb)]
        jax_inp = [list(args_jax)]
    else:
        nb_inp = args_nb
        jax_inp = args_jax

    kw_jax_fixed = {
        k.replace("axes", "axis") if op.name not in ["transpose", "unsqueeze"] else k: v
        for k, v in kw_jax.items()
    }

    run_test_with_consistency_check(
        test_name,
        lambda: op.nabla_fn(*nb_inp, **kw_nb),
        lambda: op.jax_fn(*jax_inp, **kw_jax_fixed),
    )

    if config.supports_vmap:
        run_vmap_check(test_name, op, config, args_nb, kw_nb, args_jax, kw_jax)

    if config.supports_sharding:
        run_sharding_check(test_name, op, config, args_nb, kw_nb)
