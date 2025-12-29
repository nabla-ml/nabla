
"""Rigorous tests for sharding transformation verifying NUMERICAL CORRECTNESS.

Tests complex scenarios:
1. Matrix Multiplication (Tensor Parallelism)
2. Broadcasting (Replicated -> Sharded)
3. Chain of Ops
4. Explicit Resharding Logic
"""

import asyncio
import numpy as np
import pytest
from nabla import Tensor, DeviceMesh, DimSpec
from nabla.sharding.spec import ShardingSpec

def assert_shard_values(tensor: Tensor, expected_full: np.ndarray, mesh_axis: str = "x"):
    """Helper to verify that shards reconstruct to the expected full array."""
    assert tensor._impl.is_realized
    assert len(tensor._impl._storages) > 0, "No fragments found"
    
    # Simple reconstruction for 1D sharding on axis 0 or 1
    # We assume the sharding spec tells us how to reconstruct, 
    # OR we just concatenate based on what we know the test did.
    # For these tests, we know we shard on specific dimensions.
    
    shards = [s.to_numpy() for s in tensor._impl._storages]
    
    # 1. Determine how to concat based on sharding spec
    spec = tensor._impl.sharding
    assert isinstance(spec, ShardingSpec)
    
    # Find sharded axis
    sharded_dim = -1
    for i, dim_spec in enumerate(spec.dim_specs):
        if mesh_axis in dim_spec.axes:
            sharded_dim = i
            break
            
    if sharded_dim == -1:
        # Replicated! All shards should be identical and equal to full
        for i, s in enumerate(shards):
            if not np.allclose(s, expected_full):
                raise AssertionError(f"Shard {i} (Replicated) mismatch.\nExpected:\n{expected_full}\nGot:\n{s}")
        print(f"    ✓ Replicated shards match expected value.")
        return

    # 2. Concatenate shards
    reconstructed = np.concatenate(shards, axis=sharded_dim)
    
    if not np.allclose(reconstructed, expected_full):
        print("Reconstructed:")
        print(reconstructed)
        print("Expected:")
        print(expected_full)
        raise AssertionError(f"Reconstructed tensor mismatch on axis {sharded_dim}")
        
    print(f"    ✓ Sharded reconstruction (axis {sharded_dim}) matches expected value.")


def test_matmul_tensor_parallel():
    """Test A @ B where A is sharded (Rows) and B is Replicated.
    
    Logic:
    A: (4, 4) sharded on dim 0 (Rows) -> 2 shards of (2, 4)
    B: (4, 4) replicated
    C = A @ B -> (4, 4) sharded on dim 0
    """
    print("\n=== Test: Matmul (Row Parallel) Values ===")
    
    mesh = DeviceMesh("m", (2,), ("x",))
    
    # Prepare Data
    # A = [[1, 1, 1, 1], ... ]
    np_A = np.ones((4, 4), dtype=np.float32)
    # B = Identity matrix * 2
    np_B = np.eye(4, dtype=np.float32) * 2
    
    expected_C = np_A @ np_B # Should be all 2s
    
    # Create Tensors
    A = Tensor.from_dlpack(np_A).trace()
    A.shard(mesh, [DimSpec(["x"]), DimSpec([])]) # Shard rows
    
    B = Tensor.from_dlpack(np_B).trace()
    # B initially unsharded
    
    C = A @ B
    
    # Execute
    asyncio.run(C.realize)
    
    # Verify C is sharded on dim 0
    assert len(C._impl._storages) == 2
    assert_shard_values(C, expected_C, "x")
    
    # Verify B became replicated (it was already, but partitioner handles it)
    # Wait, B is input. It stays as provided (replicated).
    # The op is performed on each shard: (2,4) @ (4,4) -> (2,4)
    # This requires B to be present on all shards.
    pass


def test_broadcasting_replicated_to_sharded():
    """Test A (Sharded) + B (Replicated) -> C (Sharded).
    
    B must be sliced/scattered or kept replicated?
    If A is (4, 4) sharded on rows.
    B is (4, 4).
    A + B.
    Shards of A are (2, 4).
    Shards of B must be (2, 4) to add!
    So B (Replicated) must be SLICED to match A's distribution.
    """
    print("\n=== Test: Broadcasting (Replicated -> Sharded) ===")
    
    mesh = DeviceMesh("m", (2,), ("x",))
    
    np_A = np.full((4, 4), 10.0, dtype=np.float32) # All 10s
    # B has distinct rows so we can check slicing correctness
    # Row 0: [0,0,0,0], Row 1: [1,1,1,1], ...
    np_B = np.arange(4, dtype=np.float32).reshape(4, 1) * np.ones((1, 4), dtype=np.float32)
    
    expected_C = np_A + np_B
    
    A = Tensor.from_dlpack(np_A).trace()
    A.shard(mesh, [DimSpec(["x"]), DimSpec([])]) # Shard Rows
    
    B = Tensor.from_dlpack(np_B).trace()
    # B is Replicated
    
    C = A + B
    
    asyncio.run(C.realize)
    
    # Verify C
    assert_shard_values(C, expected_C, "x")
    
    # Check shards of C individually to ensure B was sliced correctly
    # Shard 0 should have rows 0,1 -> 10+0=10, 10+1=11
    s0 = C._impl._storages[0].to_numpy()
    assert np.allclose(s0[0], 10.0)
    assert np.allclose(s0[1], 11.0)
    
    # Shard 1 should have rows 2,3 -> 10+2=12, 10+3=13
    s1 = C._impl._storages[1].to_numpy()
    assert np.allclose(s1[0], 12.0)
    assert np.allclose(s1[1], 13.0)
    
    print("    ✓ B was correctly sliced to match A's row sharding.")


def test_reduction_allreduce():
    """Test explicit contraction -> AllReduce.
    
    A: (4, 4) sharded on col (dim 1).
    B: (4, 4) sharded on row (dim 0).
    Wait, A @ B.
    A [M, K_sharded] @ B [K_sharded, N]. (Contraction dim is sharded)
    Result [M, N] is Partial Sums!
    Needs AllReduce to correct.
    """
    print("\n=== Test: Reduction (AllReduce) ===")
    
    mesh = DeviceMesh("m", (2,), ("x",))
    
    # A = All 1s
    np_A = np.ones((4, 4), dtype=np.float32)
    # B = All 1s
    np_B = np.ones((4, 4), dtype=np.float32)
    
    expected_C = np_A @ np_B # (4,4) of 4.0s
    
    A = Tensor.from_dlpack(np_A).trace()
    A.shard(mesh, [DimSpec([]), DimSpec(["x"])]) # Shard Cols (dim 1)
    
    B = Tensor.from_dlpack(np_B).trace()
    B.shard(mesh, [DimSpec(["x"]), DimSpec([])]) # Shard Rows (dim 0)
    
    C = A @ B
    
    # Logic trace:
    # A shards: (4, 2) [cols 0-1], (4, 2) [cols 2-3]
    # B shards: (2, 4) [rows 0-1], (2, 4) [rows 2-3]
    # Shard 0 mul: A_sub @ B_sub = (4,2)@(2,4) = (4,4) partial
    # Shard 1 mul: A_sub @ B_sub = (4,2)@(2,4) = (4,4) partial
    # Result = Sum(Shard0, Shard1)
    
    # The partitioner should auto-insert AllReduce.
    # Result C should be Replicated (4,4).
    
    asyncio.run(C.realize)
    
    assert len(C._impl._storages) == 2
    # Check both shards contain the FULL SUM (AllReduce behavior)
    # If it was ReduceScatter, they would be split. Matmul usually produces Replicated after AllReduce?
    # Our simple logic probably produces Replicated output for generic ops, 
    # but here Matmul rule might specify output is replicated?
    # Actually, A @ B -> C. k is contracted. result is (i, j). 
    # i and j are NOT sharded in inputs. So output is (Replicated, Replicated).
    
    # So both shards should match full expected_C
    assert_shard_values(C, expected_C, "x") 

    # Verify AllReduce happened:
    # If no AllReduce, Shard 0 would be parts of the sum (e.g. 2.0s instead of 4.0s)
    s0 = C._impl._storages[0].to_numpy()
    if np.allclose(s0, 2.0):
        raise AssertionError("Result appears to be partial sum (2.0). AllReduce failed!")
    
    assert np.allclose(s0, 4.0)
    print("    ✓ AllReduce correctly summed partial results.")


def test_reshard_mismatch_axes():
    """Test A (Sharded X) + B (Sharded Y, but using 1D mesh?? No, same mesh).
    
    If mesh is 1D "m" with axis "x". 
    We can't shard on "y".
    
    But we can test:
    A (Sharded on dim 0) + B (Sharded on dim 1).
    One of them must be resharded to match the other, or both to replicated.
    
    Propagation typically unifies. If we force specs?
    The current propagation logic (factors) will likely pick one strategy.
    Let's see if the partitioner handles the transition correctly.
    """
    print("\n=== Test: Resharding (Sharded Dim 0 + Sharded Dim 1) ===")
    mesh = DeviceMesh("m", (2,), ("x",))
    
    np_A = np.zeros((4, 4), dtype=np.float32)
    np_A[:, 0] = 10.0 # Col 0 is 10s
    
    np_B = np.zeros((4, 4), dtype=np.float32)
    np_B[0, :] = 5.0 # Row 0 is 5s
    
    # Expected: (0,0)=15, others 10 or 5 or 0.
    expected_C = np_A + np_B
    
    A = Tensor.from_dlpack(np_A).trace()
    A.shard(mesh, [DimSpec(["x"]), DimSpec([])]) # Shard Row (dim 0)
    
    B = Tensor.from_dlpack(np_B).trace()
    B.shard(mesh, [DimSpec([]), DimSpec(["x"])]) # Shard Col (dim 1)
    
    # A + B. 
    # Propagation:
    # A uses factors (i, j). i is sharded.
    # B uses factors (i, j). j is sharded.
    # Result factors (i, j).
    # Conflict? 
    # If strategy is Aggressive, it might try to shard both?
    # But mesh is 1D. Can't shard i on x AND j on x simultaneously unless tensor is (replicated/split?). No.
    # It must pick one or replicate.
    
    C = A + B
    
    asyncio.run(C.realize)
    
    print(f"    Result Sharding: {C._impl.sharding}")
    assert_shard_values(C, expected_C, "x")
    print("    ✓ Mismatched sharding handled correctly.")


def test_chain_complex():
    """Test (A @ B).relu() + C.sum(0)
    
    A: (4,4) sharded rows
    B: (4,4) replicated
    C: (4,4) sharded cols
    """
    print("\n=== Test: Complex Chain ===")
    mesh = DeviceMesh("m", (2,), ("x",))
    
    np_A = np.random.randn(4, 4).astype(np.float32)
    np_B = np.random.randn(4, 4).astype(np.float32)
    
    # C is all 2s
    np_C = np.ones((4, 4), dtype=np.float32) * 2
    
    # D = (A@B).relu + C.sum(0)
    # C.sum(0) -> shape (4,). Summing over Rows (dim 0).
    # C is sharded on Cols (dim 1). So sum(0) preserves sharding on dim 1?
    # Result of sum is (4,), sharded on dim 0 (which was dim 1 of C).
    
    np_res = (np.maximum(np_A @ np_B, 0)) + np_C.sum(axis=0)
    
    A = Tensor.from_dlpack(np_A).trace()
    A.shard(mesh, [DimSpec(["x"]), DimSpec([])]) # Rows
    
    B = Tensor.from_dlpack(np_B).trace()
    # Replicated
    
    C = Tensor.from_dlpack(np_C).trace()
    C.shard(mesh, [DimSpec([]), DimSpec(["x"])]) # Cols
    
    # Ops
    from nabla import ops
    matmul = A @ B
    relu = ops.unary.relu(matmul)
    
    c_sum = ops.reduce_sum(C, axis=0) 
    print(f"DEBUG: c_sum shape logical: {c_sum.shape}") 
    
    # relu is (4,4) sharded rows (from A)
    # c_sum is (4,) sharded... ??
    # C is (4, 4). Dim 1 is sharded.
    # Sum(0) removes Dim 0. Dim 1 becomes Dim 0.
    # So c_sum should be (4,) sharded on Dim 0.
    
    # relu + c_sum.
    # relu: (4, 4), sharded Dim 0.
    # c_sum: (4,), sharded Dim 0.
    # Broadcasting c_sum to (4,4).
    # Matches sharding!
    
    D = relu + c_sum
    
    asyncio.run(D.realize)
    
    assert_shard_values(D, np_res, "x")
    print("    ✓ Complex chain with mixed sharding ops correct.")


def main():
    test_matmul_tensor_parallel()
    test_broadcasting_replicated_to_sharded()
    test_reduction_allreduce()
    test_reshard_mismatch_axes()
    test_chain_complex()
    print("\nALL RIGOROUS TESTS PASSED")

if __name__ == "__main__":
    main()
