"""
Factor Propagation Stress Test - GPU Ready

Demonstrates Nabla's factor-based sharding propagation through:
1. Conflicting input shardings (triggering implicit reshard)
2. Matmul with partial computation
3. AllReduce for aggregating partial results
4. Reshape that merges/splits sharded dimensions
5. AllGather to replicate results

Auto-detects GPU availability and switches between CPU simulation and real distributed.
"""

import asyncio
import pytest
import numpy as np
from nabla import Tensor, DimSpec
import nabla.ops as ops
from nabla.sharding import ShardingSpec
from nabla.utils import debug

# Import shared utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "sharding"))
from test_utils import create_mesh, get_mode_string, is_distributed_mode


def test_full_sharding_showcase():
    """
    Comprehensive test of sharding propagation and communication.
    
    Uses 2x2 mesh (4 devices) - will use GPUs if 4+ available, else simulates.
    
    Input shardings:
    - A: [4, 8] <dp, tp> (fully sharded on 2x2 mesh)
    - B: [4, 8] <tp, dp> (CONFLICT - swapped axes)
    
    The elementwise add(A, B) must trigger all_gather to align the shardings.
    Then matmul with replicated W, followed by reshape factor tracking.
    """
    print(f"\n  Mode: {get_mode_string()}")
    
    m = create_mesh("mesh", shape=(2, 2), axis_names=("dp", "tp"))
    
    # === Inputs with CONFLICTING sharding ===
    A = Tensor.zeros((4, 8)).with_sharding(m, [DimSpec(["dp"]), DimSpec(["tp"])])
    B = Tensor.zeros((4, 8)).with_sharding(m, [DimSpec(["tp"]), DimSpec(["dp"])])
    W = Tensor.zeros((8, 4)).with_sharding(m, [DimSpec([]), DimSpec([])])
    
    def forward(a, b, w):
        h = ops.add(a, b)
        h = ops.matmul(h, w)
        h = h.with_sharding(m, [DimSpec(["dp"]), DimSpec(["tp"])])
        out = ops.reshape(h, (16,))
        out = ops.all_gather(out, axis=0)
        return out

    trace = debug.capture_trace(forward, A, B, W)
    print("\n" + "=" * 70)
    print("TRACE: Full Sharding Showcase")
    print("=" * 70)
    print(debug.GraphPrinter(trace).to_string())


def test_2gpu_dp_stress():
    """
    2-GPU specific test for data parallelism stress.
    
    Uses 1x2 mesh (2 devices) - ready for 2x H100/A100 testing.
    """
    print(f"\n  Mode: {get_mode_string()}")
    
    m = create_mesh("dp_mesh", shape=(2,), axis_names=("dp",))
    
    # Large batch, sharded on batch dimension
    a_np = np.random.randn(16, 32).astype(np.float32) * 0.1
    b_np = np.random.randn(32, 16).astype(np.float32) * 0.1
    
    A = Tensor.constant(a_np).with_sharding(m, [DimSpec(["dp"]), DimSpec([])])
    B = Tensor.constant(b_np).with_sharding(m, [DimSpec([]), DimSpec([])])
    
    # Forward
    C = ops.matmul(A, B)
    C = ops.relu(C)
    D = ops.matmul(C, B.T if hasattr(B, 'T') else ops.swap_axes(Tensor.constant(b_np), 0, 1))
    
    # Gather results
    result = ops.all_gather(D, axis=0)
    
    asyncio.run(result.realize)
    result_np = result.to_numpy()
    
    # Numpy reference
    expected = np.maximum(a_np @ b_np, 0) @ b_np.T
    
    np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)
    print(f"  ✓ 2-GPU DP stress: shape {result_np.shape}, max diff = {np.abs(result_np - expected).max():.2e}")


def test_2gpu_conflicting_shardings():
    """
    2-GPU test for conflicting shardings requiring reshard.
    """
    print(f"\n  Mode: {get_mode_string()}")
    
    m = create_mesh("conflict_mesh", shape=(2,), axis_names=("x",))
    
    a_np = np.random.randn(8, 4).astype(np.float32)
    b_np = np.random.randn(8, 4).astype(np.float32)
    
    # A sharded on dim 0, B sharded on dim 1 - CONFLICT
    A = Tensor.constant(a_np).with_sharding(m, [DimSpec(["x"]), DimSpec([])])
    B = Tensor.constant(b_np).with_sharding(m, [DimSpec([]), DimSpec(["x"])])
    
    # Add requires resharding
    result = ops.add(A, B)
    
    asyncio.run(result.realize)
    result_np = result.to_numpy()
    
    expected = a_np + b_np
    np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
    print(f"  ✓ 2-GPU conflict resolution: max diff = {np.abs(result_np - expected).max():.2e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SHARDING STRESS TESTS - GPU Ready")
    print(f"Mode: {get_mode_string()}")
    print("=" * 70)
    
    tests = [
        test_full_sharding_showcase,
        test_2gpu_dp_stress,
        test_2gpu_conflicting_shardings,
    ]
    
    for test in tests:
        print(f"\n  Running: {test.__name__}")
        try:
            test()
            print(f"  ✓ {test.__name__} passed")
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
