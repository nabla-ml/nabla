
import asyncio
import numpy as np
import pytest
from nabla import Tensor, DeviceMesh, DimSpec

def test_matmul_values_sharded():
    """Verify that sharded execution produces correct numerical results."""
    print("\n=== Test: Matmul Values Sharded ===")
    
    mesh = DeviceMesh("m", (2,), ("x",))
    
    # A: (4, 8) ones
    # Sharded on x (dim 0) -> 2 shards of (2, 8)
    A = Tensor.ones((4, 8)).trace()
    A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
    
    # B: (8, 4) ones
    # Replicated
    B = Tensor.ones((8, 4)).trace()
    
    # C = A @ B -> (4, 4) of 8.0s
    C = A @ B
    
    # Execute
    asyncio.run(C.realize)
    
    assert len(C._impl._storages) == 2, "Should have 2 shards"
    
    for i, storage in enumerate(C._impl._storages):
        arr = storage.to_numpy()
        print(f"Shard {i} shape: {arr.shape}")
        print(f"Shard {i} values:\n{arr}")
        
        expected = np.full((2, 4), 8.0, dtype=np.float32)
        if not np.allclose(arr, expected):
            raise AssertionError(f"Shard {i} incorrect! Expected 8.0, got:\n{arr}")

    print("✓ Sharded matmul values correct!")

if __name__ == "__main__":
    test_matmul_values_sharded()
