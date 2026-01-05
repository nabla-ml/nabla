"""
Distributed Sharding Tests - Portable CPU/GPU with Numerical Verification

Auto-detects available accelerators:
- 0 accelerators: Uses CPU mesh (simulated sharding)
- 2+ accelerators: Uses GPU mesh (real distributed)

Each test includes numpy reference computation for numerical correctness.

Run with: python tests/unit/sharding/test_distributed_gpu.py
"""

import asyncio
import pytest
import numpy as np
from max.graph import DeviceRef
from max import driver

from nabla import Tensor, DeviceMesh, DimSpec
from nabla.sharding import ShardingSpec
from nabla.utils import debug
import nabla.ops as ops


# ============================================================================
# Auto-Detection Helpers
# ============================================================================

def get_accelerator_count() -> int:
    """Get number of available accelerators."""
    try:
        return driver.accelerator_count()
    except Exception:
        return 0


def create_mesh_auto(name: str, num_devices: int, axis_names: tuple) -> DeviceMesh:
    """Create mesh with auto-detected device refs."""
    accel_count = get_accelerator_count()
    
    if accel_count >= num_devices:
        device_refs = [DeviceRef.GPU(i) for i in range(num_devices)]
        print(f"  [GPU MODE] Using {num_devices} accelerators")
    else:
        device_refs = [DeviceRef.CPU() for _ in range(num_devices)]
        print(f"  [CPU MODE] Simulating {num_devices} devices")
    
    shape = (num_devices,)
    return DeviceMesh(name, shape, axis_names, device_refs=device_refs)


def is_distributed_mode() -> bool:
    return get_accelerator_count() >= 2


# ============================================================================
# Numerical Verification Tests
# ============================================================================

class TestNumericalCorrectness:
    """Tests with numpy reference computations for numerical verification."""
    
    def test_elementwise_add_numerical(self):
        """Elementwise add: verify against numpy."""
        mesh = create_mesh_auto("add_mesh", 2, ("x",))
        
        # Create input data
        a_np = np.random.randn(8, 4).astype(np.float32)
        b_np = np.random.randn(8, 4).astype(np.float32)
        
        # Numpy reference
        expected = a_np + b_np
        
        # Nabla computation
        A = Tensor.constant(a_np).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        B = Tensor.constant(b_np).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        
        result = ops.add(A, B)
        
        # Evaluate and compare
        asyncio.run(result.realize)
        result_np = result.to_numpy()
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
        print(f"  ✓ Elementwise add: max diff = {np.abs(result_np - expected).max():.2e}")
    
    def test_matmul_numerical(self):
        """Matmul with sharded M dimension: verify against numpy."""
        mesh = create_mesh_auto("matmul_mesh", 2, ("x",))
        
        # Input data
        a_np = np.random.randn(8, 16).astype(np.float32)
        b_np = np.random.randn(16, 4).astype(np.float32)
        
        # Numpy reference
        expected = a_np @ b_np
        
        # Nabla: A sharded on M (batch), B replicated
        A = Tensor.constant(a_np).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        B = Tensor.constant(b_np).with_sharding(mesh, [DimSpec([]), DimSpec([])])
        
        result = ops.matmul(A, B)
        
        asyncio.run(result.realize)
        result_np = result.to_numpy()
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)
        print(f"  ✓ Matmul: max diff = {np.abs(result_np - expected).max():.2e}")
    
    def test_relu_numerical(self):
        """ReLU with sharding: verify against numpy."""
        mesh = create_mesh_auto("relu_mesh", 2, ("x",))
        
        # Input with positive and negative values
        x_np = np.random.randn(8, 4).astype(np.float32)
        
        # Numpy reference
        expected = np.maximum(x_np, 0)
        
        # Nabla
        X = Tensor.constant(x_np).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        result = ops.relu(X)
        
        asyncio.run(result.realize)
        result_np = result.to_numpy()
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
        print(f"  ✓ ReLU: max diff = {np.abs(result_np - expected).max():.2e}")
    
    def test_mlp_forward_numerical(self):
        """MLP forward pass: verify against numpy."""
        mesh = create_mesh_auto("mlp_mesh", 2, ("dp",))
        
        # Input data (small for stability)
        x_np = np.random.randn(8, 16).astype(np.float32) * 0.1
        w1_np = np.random.randn(16, 32).astype(np.float32) * 0.1
        w2_np = np.random.randn(32, 8).astype(np.float32) * 0.1
        
        # Numpy reference: x @ W1 -> relu -> @ W2
        h = x_np @ w1_np
        h = np.maximum(h, 0)  # relu
        expected = h @ w2_np
        
        # Nabla: x sharded on batch, weights replicated
        X = Tensor.constant(x_np).with_sharding(mesh, [DimSpec(["dp"]), DimSpec([])])
        W1 = Tensor.constant(w1_np).with_sharding(mesh, [DimSpec([]), DimSpec([])])
        W2 = Tensor.constant(w2_np).with_sharding(mesh, [DimSpec([]), DimSpec([])])
        
        h = ops.matmul(X, W1)
        h = ops.relu(h)
        result = ops.matmul(h, W2)
        
        asyncio.run(result.realize)
        result_np = result.to_numpy()
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)
        print(f"  ✓ MLP forward: max diff = {np.abs(result_np - expected).max():.2e}")
    
    def test_allgather_numerical(self):
        """AllGather: verify reconstructs full tensor."""
        mesh = create_mesh_auto("gather_mesh", 2, ("x",))
        
        # Create full tensor
        x_np = np.arange(16).reshape(8, 2).astype(np.float32)
        
        # Nabla: shard then gather
        X = Tensor.constant(x_np).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        gathered = ops.all_gather(X, axis=0)
        
        asyncio.run(gathered.realize)
        result_np = gathered.to_numpy()
        
        # AllGather should return the original full tensor
        np.testing.assert_allclose(result_np, x_np, rtol=1e-5, atol=1e-6)
        print(f"  ✓ AllGather: reconstructed correctly, shape {result_np.shape}")
    
    def test_reshape_numerical(self):
        """Reshape with sharding: verify data preservation."""
        mesh = create_mesh_auto("reshape_mesh", 2, ("x",))
        
        # Create data
        x_np = np.arange(32).reshape(8, 4).astype(np.float32)
        expected = x_np.reshape(32)
        
        # Nabla
        X = Tensor.constant(x_np).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        result = ops.reshape(X, (32,))
        
        asyncio.run(result.realize)
        result_np = result.to_numpy()
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-6)
        print(f"  ✓ Reshape: data preserved, shape {result_np.shape}")


# ============================================================================
# Trace-Only Tests (no numerical verification)
# ============================================================================

class TestTraceGeneration:
    """Tests that verify trace generation without numerical checks."""
    
    def test_conflicting_shardings_trace(self):
        """Conflicting shardings should trigger resharding."""
        mesh = create_mesh_auto("conflict_mesh", 2, ("x",))
        
        A = Tensor.zeros((8, 4)).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        B = Tensor.zeros((8, 4)).with_sharding(mesh, [DimSpec([]), DimSpec(["x"])])
        
        def forward(a, b):
            return ops.add(a, b)
        
        trace = debug.capture_trace(forward, A, B)
        trace_str = debug.GraphPrinter(trace).to_string()
        
        print("\n  Trace (conflicting shardings):")
        print("  " + trace_str.replace("\n", "\n  "))
        print("  ✓ Conflict resolution trace generated")
    
    def test_sharding_spec_creation(self):
        """Verify sharding spec creation and accessors."""
        mesh = create_mesh_auto("spec_mesh", 2, ("x",))
        
        A = Tensor.zeros((8, 4)).with_sharding(mesh, [DimSpec(["x"]), DimSpec([])])
        
        # Check spec properties
        assert A.sharding is not None
        assert A.sharding.mesh.name == "spec_mesh"
        assert A.sharding.dim_specs[0].axes == ["x"]
        assert A.sharding.dim_specs[1].is_replicated()
        assert tuple(A.shape) == (8, 4)
        
        print(f"  ✓ Sharding spec: {A.sharding}")


# ============================================================================
# Main Runner
# ============================================================================

def run_all_tests():
    """Run all tests with nice output."""
    accel_count = get_accelerator_count()
    
    print("\n" + "=" * 70)
    print("DISTRIBUTED SHARDING TESTS WITH NUMERICAL VERIFICATION")
    print(f"Detected accelerators: {accel_count}")
    print(f"Mode: {'GPU (distributed)' if accel_count >= 2 else 'CPU (simulated)'}")
    print("=" * 70)
    
    test_classes = [
        TestNumericalCorrectness,
        TestTraceGeneration,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)
        
        instance = test_class()
        for name in sorted(dir(instance)):
            if name.startswith("test_"):
                method = getattr(instance, name)
                try:
                    print(f"\n  Running: {name}")
                    method()
                    passed += 1
                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    failed += 1
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
