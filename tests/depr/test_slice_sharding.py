import os
import numpy as np
import pytest
import nabla as nb
from nabla import ops
from nabla.core.sharding.spec import DeviceMesh, PartitionSpec


def test_sharded_slice_ops():
    print("\n--- Test Sharded Slice Ops ---")

    # 1. Setup Mesh (simulated 2 devices)
    try:
        mesh = DeviceMesh("test_mesh", (2,), ("x",))
        print("Mesh created:", mesh)
    except Exception as e:
        print("Skipping sharding test (no devices or mesh init failed):", e)
        return

    shape = (4, 4)
    # Shard along axis 0 ('x')
    sharding = PartitionSpec("x", None)

    # 2. Input Tensor (Sharded)
    x_np = np.random.randn(*shape).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)
    x = nb.shard(x, mesh, sharding)
    print("X sharding:", x.sharding)

    # 3. Test SliceTensor on Sharded Input
    # Slice first half (should be on Dev 0)
    # slice: x[0:2, :] -> size (2, 4)
    print("Testing SliceTensor...")
    y_slice = ops.slice_tensor(x, start=(0, 0), size=(2, 4))
    # Output should ideally retain sharding-compatible factors or be replicated?
    # Our rule preserves factors.
    # Output (2,4). Factor 'x' (size 2).
    # Since 2 divides 2, it works?
    print("Y Slice shape:", y_slice.shape)

    # Force realization/compile
    y_val = y_slice.numpy()

    expected_slice = x_np[0:2, :]
    np.testing.assert_allclose(y_val, expected_slice, atol=1e-5)
    print("✅ SliceTensor Sharded -> Correct")

    # 4. Test SliceUpdate on Sharded Input
    # Update bottom half (Dev 1 mainly?)
    # update: u[2:4, :]
    update_shape = (2, 4)
    u_np = np.ones(update_shape).astype(np.float32) * 5.0
    u = nb.Tensor.from_dlpack(u_np)
    # Update is currently unsharded (replicated by default if not shard called)

    print("Testing SliceUpdate...")
    y_update = ops.slice_update(x, u, start=(2, 0), size=(2, 4))
    y_update_val = y_update.numpy()

    expected_update = x_np.copy()
    expected_update[2:4, :] = u_np

    np.testing.assert_allclose(y_update_val, expected_update, atol=1e-5)
    print("✅ SliceUpdate Sharded -> Correct")


if __name__ == "__main__":
    # Force mock XLA devices for testing if needed
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    test_sharded_slice_ops()
