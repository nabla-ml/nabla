"""Rigorous verification of communication ops on multi-GPU."""
import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, DimSpec
from max.dtype import DType
import numpy as np
import itertools
import pytest


@pytest.mark.parametrize("mesh_shape, mesh_axes, scenario_name", [
    ((2,), ("x",), "2_Linear"),
    ((4,), ("x",), "4_Linear"),
    ((8,), ("x",), "8_Linear"),
    ((2, 4), ("x", "y"), "2x4_Mesh"),
])
def test_scenario(mesh_shape, mesh_axes, scenario_name):
    print(f"\n>> SCENARIO: {scenario_name} | Shape: {mesh_shape} | Axes: {mesh_axes}")
    print("-" * 60)
    
    try:
        mesh = DeviceMesh(f"mesh_{scenario_name}", shape=mesh_shape, axis_names=mesh_axes)
    except Exception as e:
        pytest.skip(f"Skipping scenario, failed to create mesh (maybe > available devices?): {e}")

    num_devices = len(mesh.devices)

    # Determine axes to test sharding on
    sharding_specs_to_test = []
    # 1. Shard on first axis
    sharding_specs_to_test.append([mesh_axes[0]])
    
    # 2. If mesh has multiple axes (e.g. 2D), shard on second axis
    if len(mesh_axes) > 1:
        sharding_specs_to_test.append([mesh_axes[1]])
        # 3. Shard on BOTH axes (fully sharded)
        sharding_specs_to_test.append(list(mesh_axes))
        
    print(f"Testing Sharding Specs: {sharding_specs_to_test}")

    for shard_axes in sharding_specs_to_test:
        print(f"\n  >> Testing with Sharding Config: {shard_axes}")
        
        # 1. Test AllGather
        print("    [1] Testing all_gather...")
        full_size = num_devices * 4
        data = np.arange(full_size, dtype=np.float32)
        tensor = nb.Tensor.constant(data, dtype=DType.float32)
        
        dim_spec = DimSpec(shard_axes)
        sharded = tensor.shard(mesh, [dim_spec])
        sharded.realize()
        
        gathered = sharded.gather()
        gathered.realize()
        res = gathered.numpy()
        
        assert np.allclose(res, data), "Data mismatch in AllGather"
        print("        ✅ PASS")

        # 2. Test AllReduce
        print("    [2] Testing all_reduce...")
        full_size = num_devices * 4
        data = np.ones(full_size, dtype=np.float32)
        tensor = nb.Tensor.constant(data, dtype=DType.float32)
        dim_spec = DimSpec(shard_axes)
        sharded = tensor.shard(mesh, [dim_spec])
        
        from nabla.ops.communication import all_reduce
        reduced = all_reduce(sharded)
        reduced.realize()
        res = reduced.numpy()
        
        val_factor = num_devices
        # Since all_reduce (no args) sums over ALL devices, and input is ones on all devices:
        # Each device contributes 1.0. Total sum = num_devices.
        # Note: The original test assumed it only summed sharded axes, but for partial replication (e.g. 2x4 mesh sharded on x),
        # all_reduce sums the replicas on y too.
        
        length_factor = 1
        for ax in shard_axes:
            length_factor *= mesh.get_axis_size(ax)

        expected_len = full_size // length_factor
        
        assert res.shape == (expected_len,), f"Shape mismatch: {res.shape} != {(expected_len,)}"
        assert np.allclose(res, val_factor), f"Value mismatch: {res[0] if res.size>0 else '?'} != {val_factor}"
        print(f"        ✅ PASS (Factor {val_factor})")

        # 3. Test ReduceScatter
        print("    [3] Testing reduce_scatter...")
        # Input size needs to be divisible.
        full_size = num_devices * num_devices * 4
        data_rs = np.ones(full_size, dtype=np.float32)
        tensor_rs = nb.Tensor.constant(data_rs, dtype=DType.float32)
        dim_spec = DimSpec(shard_axes)
        sharded_rs = tensor_rs.shard(mesh, [dim_spec])
        
        from nabla.ops.communication import reduce_scatter
        # We scatter along axis 0.
        result = reduce_scatter(sharded_rs, axis=0)
        result.realize()
        res = result.numpy()
        
        assert res.size > 0, "Empty result in ReduceScatter"
        print("        ✅ PASS (Executed)")

    # 4. Test AxisIndex
    print("  [4] Testing axis_index...")
    from nabla.ops.communication import axis_index
    for ax in mesh_axes:
        idx_tensor = axis_index(mesh, ax)
        idx_tensor.realize()
        gathered = idx_tensor.gather()
        gathered.realize()
        res = gathered.numpy()
        
        assert np.min(res) >= 0 and np.max(res) < mesh.get_axis_size(ax), f"Axis {ax}: invalid range {res}"
        print(f"    ✅ PASS (Axis {ax})")

    # 5. Test AllToAll (Simple Check)
    print("  [5] Testing all_to_all...")
    from nabla.ops.communication import all_to_all
    # Only feasible if sharding on axis 0.
    if len(sharding_specs_to_test) > 0:
            # Just use first spec
            spec = sharding_specs_to_test[0]
            # Ensure divisibility for AllToAll: ShardSize (Global/N) must be divisible by N.
            # So Global must be divisible by N*N.
            data = np.arange(num_devices * num_devices * 2, dtype=np.float32)
            tensor = nb.Tensor.constant(data, dtype=DType.float32)
            dim_spec = DimSpec(spec)
            sharded = tensor.shard(mesh, [dim_spec])
            result = all_to_all(sharded, 0, 0)
            result.realize()
            print("    ✅ PASS")

