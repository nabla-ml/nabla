"""Systematically test ALL communication operations on real multi-GPU setup."""

import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, DimSpec
from max.dtype import DType
import numpy as np

print("Testing Communication Operations on Multi-GPU")
print("=" * 70)

# Setup
mesh_2 = DeviceMesh("mesh_2gpu", shape=(2,), axis_names=("x",))
print(f"Mesh: {mesh_2}, is_distributed: {mesh_2.is_distributed}\n")


def create_sharded_tensor(data, mesh, dim_specs):
    """Helper to create and shard a tensor."""
    tensor = nb.Tensor.constant(data, dtype=DType.float32)
    sharded = tensor.shard(mesh, dim_specs)
    sharded.realize()
    return sharded


# Test 1: all_gather (already working from our previous test)
print("1. Testing all_gather...")
try:
    data = np.arange(16, dtype=np.float32)
    sharded = create_sharded_tensor(data, mesh_2, [DimSpec(["x"], is_open=False)])

    print(f"   Sharded tensor: {len(sharded._impl._bufferss)} shards")
    for i, s in enumerate(sharded._impl._bufferss):
        print(f"     Shard {i}: device={s.device}, shape={s.shape}")

    # Use the .gather() method
    gathered = sharded.gather()
    gathered.realize()
    result = gathered.numpy()

    print(f"   Gathered result: {result}")
    print(f"   ✓ all_gather works: {np.allclose(result, data)}")
except Exception as e:
    print(f"   ✗ all_gather FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 2: all_reduce (sum reduction across devices)
print("\n2. Testing all_reduce...")
try:
    from nabla.ops.communication import all_reduce

    data = np.ones(8, dtype=np.float32)  # All ones
    sharded = create_sharded_tensor(data, mesh_2, [DimSpec(["x"], is_open=False)])

    # All reduce should sum across devices
    reduced = all_reduce(sharded)
    reduced.realize()
    result = reduced.numpy()

    print(f"   Input: ones(8) sharded across 2 devices")
    print(f"   Result shape: {result.shape}")
    print(f"   All-reduce result: {result}")
    # Implementation sums the shards element-wise: [1,1,1,1] + [1,1,1,1] = [2,2,2,2]
    # Result is size 4 (half of original 8)
    expected = np.ones(4, dtype=np.float32) * 2.0
    print(f"   ✓ all_reduce works: {np.allclose(result, expected)}")
except Exception as e:
    print(f"   ✗ all_reduce FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 3: reduce_scatter
print("\n3. Testing reduce_scatter...")
try:
    from nabla.ops.communication import reduce_scatter

    # Create replicated tensor (not sharded)
    data = np.arange(16, dtype=np.float32)
    tensor = nb.Tensor.constant(data, dtype=DType.float32)

    # reduce_scatter should partition and reduce
    result = reduce_scatter(tensor, axis=0)
    result.realize()

    print(f"   Input: arange(16) replicated")
    print(f"   Result is sharded: {result.is_sharded}")
    print(f"   ✓ reduce_scatter executed (shape check needed)")
except Exception as e:
    print(f"   ✗ reduce_scatter FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 4: all_to_all
print("\n4. Testing all_to_all...")
try:
    from nabla.ops.communication import all_to_all

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    sharded = create_sharded_tensor(
        data, mesh_2, [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)]
    )

    # All-to-all transposes sharding
    result = all_to_all(sharded, split_axis=0, concat_axis=1)
    result.realize()

    print(f"   Input shape: {data.shape}, sharded on axis 0")
    print(f"   Result is sharded: {result.is_sharded}")
    print(f"   ✓ all_to_all executed")
except Exception as e:
    print(f"   ✗ all_to_all FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 5: ppermute
print("\n5. Testing ppermute...")
try:
    from nabla.ops.communication import ppermute

    data = np.array([100.0, 200.0], dtype=np.float32)
    sharded = create_sharded_tensor(data, mesh_2, [DimSpec(["x"], is_open=False)])

    # Swap devices: 0->1, 1->0
    permutation = [(0, 1), (1, 0)]
    result = ppermute(sharded, permutation)
    result.realize()

    print(f"   Input: [100, 200] sharded on 2 devices")
    print(f"   Permutation: {permutation}")
    print(f"   ✓ ppermute executed")
except Exception as e:
    print(f"   ✗ ppermute FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 6: axis_index
print("\n6. Testing axis_index...")
try:
    from nabla.ops.communication import axis_index

    # axis_index returns the device coordinate for an axis
    idx = axis_index(mesh_2, "x")
    idx.realize()

    print(f"   Axis index for 'x': computed")
    print(f"   ✓ axis_index executed")
except Exception as e:
    print(f"   ✗ axis_index FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
print("Communication operations testing complete!")
print("Check above for any FAILED operations that need fixing.")
