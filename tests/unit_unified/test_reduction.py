
# ===----------------------------------------------------------------------=== #
# Unified Test: Reduction
# ===----------------------------------------------------------------------=== #

import pytest
import nabla as nb
import jax.numpy as jnp
from functools import partial

from .common import (
    Operation, OpConfig, standard_get_args, run_unified_test
)

OPS = {}

# ============================================================================
# Operations
# ============================================================================

OPS["sum"] = Operation(
    "sum", "REDUCTION", nb.reduce_sum, jnp.sum,
    [
        OpConfig("Sum_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig("Sum_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}),
        # OpConfig("Sum_All", ranks=(2,), params={"axis": None}), # Axis None not supported by reduce_sum yet?
    ],
    standard_get_args
)

OPS["mean"] = Operation(
    "mean", "REDUCTION", nb.mean, jnp.mean,
    [
        OpConfig("Mean_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig("Mean_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}),
    ],
    standard_get_args
)

OPS["max"] = Operation(
    "max", "REDUCTION", nb.reduce_max, jnp.max,
    [
        OpConfig("Max_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig("Max_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}),
    ],
    standard_get_args
)

OPS["min"] = Operation(
    "min", "REDUCTION", nb.reduce_min, jnp.min,
    [
        OpConfig("Min_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig("Min_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}),
    ],
    standard_get_args
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

# ============================================================================
# Distributed Variance Tests
# ============================================================================

from .common import MESH_CONFIGS, DeviceMesh, get_sharding_configs, run_test_with_consistency_check

@pytest.mark.parametrize("op_name", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("mesh_cfg", MESH_CONFIGS)
def test_reduction_sharding_variance(op_name, mesh_cfg):
    """Test reduction ops on various mesh shapes and sharding configurations."""
    mesh_name, mesh_shape, mesh_axes = mesh_cfg
    mesh = DeviceMesh(mesh_name, mesh_shape, mesh_axes)
    
    op = OPS[op_name]
    # Use a generic Rank2 config or create a new one for Rank3 test?
    # Let's use Rank2 for simplicity but ensure we test cross-axis reduction.
    config = OpConfig("Rank2", ranks=(2,), params={"axis": 0})
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    
    input_tensor = args_nb[0]
    rank = len(input_tensor.shape)
    
    # Generate varied sharding specs
    specs = get_sharding_configs(mesh, rank)
    
    # Also test reducing on Sharded Axis vs Non-Sharded Axis
    # config params sets axis=0. We can override this in loop.
    test_axes = [0, 1]
    
    nb_fn = partial(op.nabla_fn, **kwargs_nb) # Base kwargs, override axis later
    jax_fn = partial(op.jax_fn, **kwargs_jax)
    
    for spec_idx, spec in enumerate(specs):
        if spec is None: continue
        
        for reduce_axis in test_axes:
            # Skip if axis out of bounds (though Rank2 has 0,1)
            if reduce_axis >= rank: continue
            
            # Determine if this constitutes a cross-shard reduction
            # (i.e. is the reduce_axis sharded in the spec?)
            is_cross_shard = False
            if spec.dim_specs[reduce_axis].axes:
                is_cross_shard = True
                
            test_id = f"{op_name}_{mesh_name}_Spec{spec_idx}_Axis{reduce_axis}"
            
            def sharded_exec():
                t = args_nb[0]
                # Reset sharding if needed? args_nb[0] is reused, so be careful.
                # Actually args_nb is tuple, t is reference.
                # Use with_sharding which creates NEW tensor usually or modify impl?
                # Best to create a fresh tensor or ensure with_sharding returns new wrapper.
                # Nabla Tensor.with_sharding returns self if already sharded? 
                # Better allow it to re-shard.
                
                # Apply sharding
                t_sharded = t.with_sharding(spec.mesh, spec.dim_specs)
                
                # Run reduction
                res = op.nabla_fn(t_sharded, axis=reduce_axis, keepdims=True)
                
                # Prevent optimized-away-squeeze if we wanna verify result values
                # But op.nabla_fn (e.g. reduce_sum wrapper) handles squeeze if keepdims=False
                # Here we force keepdims=True inside sharded_exec?
                # Actually, standard_get_args returns kwargs_nb with axis=0.
                # We need to override axis.
                return op.nabla_fn(t_sharded, axis=reduce_axis, keepdims=False)

            def expected_exec():
                 # JAX execution
                 t_jax = args_jax[0]
                 return op.jax_fn(t_jax, axis=reduce_axis, keepdims=False)

            run_test_with_consistency_check(
                test_id,
                sharded_exec,
                expected_exec
            )
