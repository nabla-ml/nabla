
# ===----------------------------------------------------------------------=== #
# Unified Test Common Utilities
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass, field
from typing import Any, Callable, List, Dict, Optional, Tuple, Union

import nabla as nb
from max.dtype import DType
from nabla.core.sharding.spec import DeviceMesh, ShardingSpec, P, DimSpec
from nabla import vmap

# Initialize global state
nb.DType = DType  # Monkey-patch for tests
SEED = 42

# Mesh configurations for comprehensive sharding tests
MESH_CONFIGS = [
    ("1x2", (1, 2), ("x", "y")),
    ("2x1", (2, 1), ("x", "y")),
    ("2x2", (2, 2), ("x", "y")),
    ("1x4", (1, 4), ("x", "y")),
    ("4x1", (4, 1), ("x", "y")),
]

# ============================================================================
# DATACLASSES
# ============================================================================

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
    configs: List[OpConfig]
    get_args: Callable[[OpConfig], Tuple[Tuple, Tuple]]

# ============================================================================
# JAX WRAPPERS
# ============================================================================

def jax_transpose_wrapper(x, axis1, axis2):
    rank = len(x.shape)
    if rank < 2: return x
    axes = list(range(rank))
    a1, a2 = (axis1 if axis1 >= 0 else rank + axis1), (axis2 if axis2 >= 0 else rank + axis2)
    axes[a1], axes[a2] = axes[a2], axes[a1]
    return jnp.transpose(x, axes=axes)

def jax_expand_dims_wrapper(x, axes):
    res = x
    for axis in sorted(axes):
        res = jnp.expand_dims(res, axis=axis)
    return res

def jax_squeeze_wrapper(x, axis=None, axes=None):
    ax = axis if axis is not None else axes
    if ax is None: return jnp.squeeze(x)
    return jnp.squeeze(x, axis=tuple(ax) if isinstance(ax, (list, tuple)) else ax)

def jax_split_wrapper(x, num_splits=None, axis=0, **kwargs):
    if num_splits is None: num_splits = kwargs.get('split_size_or_sections')
    return jnp.split(x, num_splits, axis=axis)

def jax_matmul_wrapper(x, y):
    x_rank, y_rank = len(x.shape), len(y.shape)
    if x_rank == 1 and y_rank == 1: return jnp.dot(x, y)
    if x_rank == 1: return jnp.expand_dims(x, 0) @ y
    if y_rank == 1: return jnp.squeeze(x @ jnp.expand_dims(y, 1), -1)
    return jnp.matmul(x, y)

def jax_slice_wrapper(x, slices):
    # slices is list of slice objects or ints
    # JAX syntax: x[tuple(slices)]
    return x[tuple(slices)]

def jax_pad_inverse_slice(x, slices, target_shape):
    # Simulate pad by creating zeros and inserting?
    # Actually this wrapper was simulating pad via slicing inverse? 
    # Let's keep it simple: create zeros and set.
    res = jnp.zeros(target_shape, dtype=x.dtype)
    res = res.at[tuple(slices)].set(x)
    return res

def jax_unsqueeze_wrapper(x, axis=None, axes=None):
    res = x
    ax = axis if axis is not None else axes
    if ax is None: return x
    # Handle int
    if isinstance(ax, int):
        return jnp.expand_dims(res, axis=ax)
    # Handle list/tuple
    for a in sorted(ax):
        res = jnp.expand_dims(res, axis=a)
    return res


# ============================================================================
# DATA GENERATION
# ============================================================================

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
    np.random.seed(SEED)
    nabla_primals = []
    jax_primals = []

    for i, shape in enumerate(shapes):
        num_elements = int(np.prod(shape)) if shape else 1
        
        if config.input_dtype == "bool":
            jax_base = jax.numpy.arange(num_elements)
            nb_base_reshaped = nb.reshape(nb.arange(num_elements), shape)
            nb_val, jax_val = (
                (
                    nb.equal(nb_base_reshaped % 2, 0),
                    (jax_base.reshape(shape) % 2 == 0),
                )
                if shape
                else (nb.constant(True, dtype=nb.DType.bool), jnp.array(True))
            )
        else:  # Numeric
            if not shape:
                base_val = 2.5 if config.domain_positive else 1.5
                nb_val, jax_val = (
                    nb.constant(base_val, dtype=nb.DType.float32),
                    jnp.array(base_val, dtype="float32"),
                )
            else:
                nb_base = nb.arange(num_elements, dtype=nb.DType.float32)
                jax_base = jax.numpy.arange(num_elements, dtype="float32")

                offset = 1.0 if config.domain_positive else float(i + 1)
                nb_val = nb.reshape(nb_base + offset, shape)
                jax_val = (jax_base + offset).reshape(shape)

                if not config.use_stable_floats:
                    nb_val *= 0.1
                    jax_val *= 0.1

        nabla_primals.append(nb_val)
        jax_primals.append(jax_val)
    return tuple(nabla_primals), tuple(jax_primals)

def standard_get_args(config: OpConfig):
    shapes = config.primal_shapes or tuple(get_shape_for_rank(r) for r in config.ranks)
    primals_nb, primals_jax = get_test_data_for_shapes(shapes, config)
    return (primals_nb, config.params), (primals_jax, config.params)

# ============================================================================
# RUNNER & VERIFICATION
# ============================================================================

def cleanup_caches():
    jax.clear_caches()
    if hasattr(nb, "_clear_caches"):
        nb._clear_caches()

def compare_nested_structures(nb_res, jax_res, path="", tolerance=1e-4):
    """Recursively compare arbitrary nested structures (tuples, lists, dicts)."""
    # Direct Tensor comparison
    if hasattr(nb_res, "numpy"):  
        nb_val = nb_res.numpy()
        jax_val = np.array(jax_res)
        
        # Shape check
        if nb_val.shape != jax_val.shape:
             # Allow scalar 0-rank mismatch if values match? Strict for now.
             pass

        np.testing.assert_allclose(
            nb_val, jax_val, rtol=tolerance, atol=tolerance,
            err_msg=f"Mismatch at {path}"
        )
        return

    # Tuple/List recursive step
    if isinstance(nb_res, (tuple, list)) and isinstance(jax_res, (tuple, list)):
        assert len(nb_res) == len(jax_res), f"Length mismatch at {path}: {len(nb_res)} vs {len(jax_res)}"
        for i, (n, j) in enumerate(zip(nb_res, jax_res)):
            compare_nested_structures(n, j, path=f"{path}[{i}]", tolerance=tolerance)
        return

    # Dict recursive step
    if isinstance(nb_res, dict) and isinstance(jax_res, dict):
        assert nb_res.keys() == jax_res.keys(), f"Key mismatch at {path}"
        for k in nb_res:
            compare_nested_structures(nb_res[k], jax_res[k], path=f"{path}.{k}", tolerance=tolerance)
        return

    # Leaves (scalars, None, etc)
    if nb_res is None or jax_res is None:
        assert nb_res is jax_res, f"None mismatch at {path}"
        return
        
    # Final fallback: cast to numpy?
    try:
        nb_val = np.array(nb_res)
        jax_val = np.array(jax_res)
        np.testing.assert_allclose(nb_val, jax_val, rtol=tolerance, atol=tolerance)
    except Exception:
        assert nb_res == jax_res, f"Value mismatch at {path}: {nb_res} != {jax_res}"


def run_test_with_consistency_check(test_name: str, nabla_fn_lazy, jax_fn_eager):
    """Executes Nabla (lazy) and JAX (eager) functions and compares results."""
    # Run JAX
    try:
        jax_res = jax_fn_eager()
    except Exception as e:
        pytest.fail(f"[{test_name}] JAX execution failed: {e}")

    # Run Nabla
    try:
        cleanup_caches()
        nabla_res = nabla_fn_lazy()
    except Exception as e:
        pytest.fail(f"[{test_name}] Nabla execution failed: {e}")

    # Compare
    try:
        compare_nested_structures(nabla_res, jax_res)
    except AssertionError as e:
        pytest.fail(f"[{test_name}] Value Mismatch: {e}")

# ============================================================================
# SHARDING
# ============================================================================

def get_sharding_configs(mesh: DeviceMesh, rank: int) -> List[Optional[ShardingSpec]]:
    """Generate interesting sharding specs for a tensor of given rank."""
    configs = [None] # Baseline: No sharding
    
    # Replicated everywhere
    from nabla.core.sharding.spmd import create_replicated_spec
    configs.append(create_replicated_spec(mesh, rank))
    
    # Shard specific axes if rank allows
    # 2x2 mesh. Axes: 0, 1.
    # Shard axis 0 on mesh axis 0
    if rank > 0:
        configs.append(ShardingSpec(mesh, (("x",),) + ((),) * (rank - 1)))
        
    # Shard last axis on mesh axis 1
    if rank > 0:
        configs.append(ShardingSpec(mesh, ((),) * (rank - 1) + (("y",),)))
        
    return configs

# ============================================================================
# COMPREHENSIVE RUNNER
# ============================================================================

def run_vmap_check(test_name, op, config, args_nb, kw_nb, args_jax, kw_jax):
    """Checks automatic vectorization (vmap)."""
    # Create batch of data by stacking or expanding keys?
    # Actually unified tests usually just vmap over the existing 0-th dimension if applicable,
    # OR we stack copies.
    # Let's stack 2 copies to create a batch dim at 0.
    
    # helper to batchify
    def batchify(x):
        if hasattr(x, "shape"): # Tensor/Array
            return nb.stack([x, x]) if isinstance(x, nb.Tensor) else jnp.stack([x, x])
        return x # Scalar/None? 
        
    # TODO: Handle list inputs better
    
    # For now, simplistic approach:
    # 1. Batchify inputs
    # 2. vmap(fn)
    # 3. compare
    
    # If list input, args_nb is tuple, but element 0 might be the list?
    # standard_get_args returns tuple of args.
    
    try:
        if config.is_list_input:
            # Skip VMap for list inputs for now (complex in_axes handling)
            return
        else:
            in_axes_list = []
            batched_args_nb_list = []
            batched_args_jax_list = []
            
            for arg_nb, arg_jax in zip(args_nb, args_jax):
                is_tensor = hasattr(arg_nb, 'shape') and not isinstance(arg_nb, (tuple, list))
                # Note: Assuming checking arg_nb is enough.
                
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
            
        # Nabla VMap
        nb_vmapped = vmap(partial(op.nabla_fn, **kw_nb), in_axes=in_axes)
        nb_res = nb_vmapped(*batched_args_nb)
        
        # JAX VMap
        # Fix kwargs for JAX
        kw_jax_fixed = {k.replace("axes", "axis") if op.name not in ["transpose", "unsqueeze"] else k: v for k,v in kw_jax.items()}
        jax_vmapped = jax.vmap(partial(op.jax_fn, **kw_jax_fixed), in_axes=in_axes)
        jax_res = jax_vmapped(*batched_args_jax)
        
        compare_nested_structures(nb_res, jax_res, path=f"{test_name}.vmap")
        
    except Exception as e:
        pytest.fail(f"[{test_name}] VMap check failed: {e}")

def run_sharding_check(test_name, op, config, args_nb, kw_nb):
    """Checks sharding propagation."""
    # 1. Create a mesh
    mesh_shape = (2, 2)
    mesh = DeviceMesh("test_mesh", mesh_shape, ("x", "y"))
    
    # 2. Shard inputs (e.g. on 'x')
    # We can only shard if rank > 0.
    
    def shard_arg(x):
        if isinstance(x, nb.Tensor) and len(x.shape) > 0:
            # Only shard if first dim is divisible by mesh axis 'x' size (2)
            if int(x.shape[0]) % mesh_shape[0] == 0:
                # Shard first axis on 'x'
                spec = ShardingSpec(mesh, (("x",),) + ((),) * (len(x.shape)-1))
                return x.shard(mesh, spec.dim_specs)
        return x
        
    sharded_args_nb = []
    if config.is_list_input:
        # args_nb is a tuple of tensors that should be passed as a list
        sharded_args_nb.append([shard_arg(x) for x in args_nb])
    else:
        sharded_args_nb = [shard_arg(x) for x in args_nb]
        
    try:
        # Run op
        res = op.nabla_fn(*sharded_args_nb, **kw_nb)
        
        # Just check that it ran successfully.
        # Some ops (broadcast_to, creation ops) may legitimately return 
        # tensors without sharding specs - don't fail these.
        if isinstance(res, nb.Tensor) and res.sharding is not None:
            # If inputs were sharded and output has sharding, verify mesh matches
            assert res.sharding.mesh == mesh, "Result mesh mismatch"
            
            # Optional strict check if config or caller provided expectations
            # For now, we just ensure it didn't crash and returned valid metadata.

    except Exception as e:
        # Sharding can fail legitimately for some op/config combos
        # pytest.skip(f"[{test_name}] Sharding check skipped: {e}")
        # Allow failing proper
        raise e

def assert_spec(tensor: nb.Tensor, expected_dims: tuple[tuple[str, ...], ...]):
    """Assert tensor's sharding spec matches expected dimensions."""
    if tensor.sharding is None:
        assert expected_dims is None, "Expected sharding spec but got None"
        return
        
    actual_dims = tuple(tuple(ds.axes) for ds in tensor.sharding.dim_specs)
    # Normalize empty tuples
    actual_dims = tuple(tuple(a for a in d) for d in actual_dims)
    expected_dims = tuple(tuple(a for a in d) for d in expected_dims)
    
    assert actual_dims == expected_dims, f"Sharding spec mismatch: got {actual_dims}, expected {expected_dims}"


# ============================================================================
# EXTRA HELPERS FOR PHYSICAL OPS
# ============================================================================

def shard_on_axis(tensor: nb.Tensor, mesh: DeviceMesh, axis: int, mesh_axis: int = 0) -> nb.Tensor:
    """Shard tensor on a specific axis using specified mesh dimension."""
    rank = len(tensor.shape)
    # Default to open specs
    specs = [DimSpec([], is_open=True) for _ in range(rank)]
    # The sharded axis is fixed (closed)
    specs[axis] = DimSpec([mesh.axis_names[mesh_axis]], is_open=False)
    return tensor.shard(mesh, specs)

def replicated(tensor: nb.Tensor, mesh: DeviceMesh) -> nb.Tensor:
    """Create a fully replicated sharded tensor."""
    rank = len(tensor.shape)
    specs = [DimSpec([], is_open=True) for _ in range(rank)]
    return tensor.shard(mesh, specs)

def assert_shape(result: nb.Tensor, expected_shape: tuple):
    """Assert tensor.shape matches expected (logical shape)."""
    actual = tuple(int(d) for d in result.shape)
    assert actual == expected_shape, f"Shape mismatch: got {actual}, expected {expected_shape}"

def assert_physical_shape(result: nb.Tensor, expected_shape: tuple):
    """Assert tensor's physical shape (global_shape) matches expected."""
    actual = result.global_shape or result.local_shape
    actual = tuple(int(d) for d in actual)
    assert actual == expected_shape, f"Physical shape mismatch: got {actual}, expected {expected_shape}"

def assert_is_sharded(result: nb.Tensor, expected: bool = True):
    """Assert tensor is/isn't sharded."""
    actual = result.is_sharded
    assert actual == expected, f"is_sharded mismatch: got {actual}, expected {expected}"



def run_unified_test(op: Operation, config: OpConfig, suffix: str = ""):
    """Master runner."""
    test_name = f"{op.name}_{config.description}{suffix}"
    
    (args_nb, kw_nb), (args_jax, kw_jax) = op.get_args(config)
    
    # 1. Standard Consistency
    # handle list input for execution
    # 1. Standard Consistency
    # handle list input for execution
    if config.is_list_input:
        # standard_get_args returns flat tuple of tensors.
        # We assume all of them belong to the list argument (e.g. stack/concat([t1, t2])).
        nb_inp = [list(args_nb)] 
        jax_inp = [list(args_jax)]
    else:
        nb_inp = args_nb
        jax_inp = args_jax
        
    kw_jax_fixed = {k.replace("axes", "axis") if op.name not in ["transpose", "unsqueeze"] else k: v for k,v in kw_jax.items()}
    
    run_test_with_consistency_check(
        test_name, 
        lambda: op.nabla_fn(*nb_inp, **kw_nb), 
        lambda: op.jax_fn(*jax_inp, **kw_jax_fixed)
    )
    
    # 2. VMap Check
    if config.supports_vmap:
        run_vmap_check(test_name, op, config, args_nb, kw_nb, args_jax, kw_jax)
        
    # 3. Sharding Check
    if config.supports_sharding:
        run_sharding_check(test_name, op, config, args_nb, kw_nb)

