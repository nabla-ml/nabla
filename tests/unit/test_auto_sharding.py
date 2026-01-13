
import asyncio
import unittest
import json
import numpy as np

from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.transforms.shard_map import shard_map
from nabla.transforms.graph_extractor import ShardingGraphExtractor
from nabla.optimizer.simple_solver import SimpleSolver
from nabla.core.trace import trace
from nabla import ops

class TestAutoSharding(unittest.TestCase):
    """Tests for Automated Sharding System."""

    def setUp(self):
        # 4 devices
        self.mesh = DeviceMesh("test_mesh", (4,), ("d",), devices=[0, 1, 2, 3])

    def test_graph_extraction(self):
        """Verify Graph Extractor produces valid JSON."""
        print("\nTEST: Graph Extraction")
        
        def func(a, b):
            return a @ b

        a = Tensor.from_dlpack(np.random.rand(128, 128).astype(np.float32))
        b = Tensor.from_dlpack(np.random.rand(128, 128).astype(np.float32))

        # Trace
        t = trace(func, a, b)
        
        # Extract
        extractor = ShardingGraphExtractor(t, in_specs={}, out_specs={})
        json_str = extractor.extract()
        data = json.loads(json_str)

        print(json_str)

        self.assertIn("tensors", data)
        self.assertIn("nodes", data)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["op_name"], "matmul")
        
        # Verify cost model
        flops = data["nodes"][0]["compute_stats"]["flops"]
        # 2 * 128^3 = 4194304
        self.assertEqual(flops, 2 * 128 * 128 * 128)

    def test_simple_solver_logic(self):
        """Verify Solver logic for choosing DP vs MP with propagation-based solver."""
        print("\nTEST: Solver Logic (Propagation-Based)")
        
        solver = SimpleSolver(self.mesh.shape, self.mesh.axis_names)
        
        # Case 1: Prefer DP (M divisible by 4, Low Comm)
        # Mock Graph JSON with equation for propagation
        graph_dp = {
            "tensors": [
                {"id": 0, "shape": [1024, 1024]}, # A
                {"id": 1, "shape": [1024, 1024]}, # B (Replicated)
                {"id": 2, "shape": [1024, 1024]}  # C
            ],
            "nodes": [
                {
                    "id": 0, "op_name": "matmul",
                    "inputs": [0, 1], "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 1024, "k": 1024, "n": 1024}
                    },
                    "compute_stats": {"flops": 2e9}
                }
            ]
        }
        
        sol_dp = solver.solve(json.dumps(graph_dp), debug=True)
        print("DP Solution:", json.dumps(sol_dp, indent=2))
        
        # NEW FORMAT: Node-centric solution
        # Check node "0" has correct output sharding (Split M = dim 0)
        self.assertIn("nodes", sol_dp)
        self.assertIn("0", sol_dp["nodes"])
        node_sol = sol_dp["nodes"]["0"]
        
        # Output should be sharded on dim 0 ("d") for DP
        self.assertIn("outputs", node_sol)
        self.assertIn("0", node_sol["outputs"])
        out_dims = node_sol["outputs"]["0"]["dims"]
        self.assertEqual(out_dims[0], ["d"])  # Split M (dim 0)
        
        # Input A should also be sharded on dim 0
        self.assertIn("inputs", node_sol)
        self.assertIn("0", node_sol["inputs"])
        in_a_dims = node_sol["inputs"]["0"]["dims"]
        self.assertEqual(in_a_dims[0], ["d"])  # Split M (dim 0)
        
        # Case 2: Prefer MP (M small/indivisible, K large)
        graph_mp = {
            "tensors": [
                {"id": 0, "shape": [3, 4096]}, 
                {"id": 1, "shape": [4096, 1024]},
                {"id": 2, "shape": [3, 1024]} 
            ],
            "nodes": [
                {
                    "id": 0, "op_name": "matmul",
                    "inputs": [0, 1], "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 3, "k": 4096, "n": 1024}
                    },
                    "compute_stats": {"flops": 2e9}
                }
            ]
        }
        
        sol_mp = solver.solve(json.dumps(graph_mp), debug=True)
        print("MP Solution:", json.dumps(sol_mp, indent=2))
        
        # For MP: Input A (dim 1 = K), Input B (dim 0 = K) sharded
        node_sol_mp = sol_mp["nodes"]["0"]
        
        # Input 0 (A): [m, k] -> split dim 1 ("d")
        in_a_dims_mp = node_sol_mp["inputs"]["0"]["dims"]
        self.assertEqual(in_a_dims_mp[1], ["d"])  # Split K (dim 1)
        
        # Input 1 (B): [k, n] -> split dim 0 ("d")
        in_b_dims_mp = node_sol_mp["inputs"]["1"]["dims"]
        self.assertEqual(in_b_dims_mp[0], ["d"])  # Split K (dim 0)
        
        # Output 2 (C): Replicated (None sharding on both dims)
        out_dims_mp = node_sol_mp["outputs"]["0"]["dims"]
        self.assertEqual(out_dims_mp, [None, None])

    def test_solver_bidirectional_propagation(self):
        """Verify that constraints propagate bidirectionally through the graph.
        
        This tests the key feature: output constraints should propagate back to inputs.
        """
        print("\nTEST: Bidirectional Propagation")
        
        solver = SimpleSolver(self.mesh.shape, self.mesh.axis_names)
        
        # Graph: A -> matmul -> B -> add -> C
        # If we fix output C's sharding, it should propagate back through add to B,
        # then through matmul to A.
        graph = {
            "tensors": [
                {"id": 0, "shape": [1024, 1024]},  # A
                {"id": 1, "shape": [1024, 1024]},  # weight
                {"id": 2, "shape": [1024, 1024]},  # B = A @ weight
                {"id": 3, "shape": [1024, 1024]},  # bias
                {"id": 4, "shape": [1024, 1024], "fixed_sharding": {"dims": [["d"], None], "replicated": []}}  # C (fixed!)
            ],
            "nodes": [
                {
                    "id": 0, "op_name": "matmul",
                    "inputs": [0, 1], "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 1024, "k": 1024, "n": 1024}
                    },
                    "compute_stats": {"flops": 2e9}
                },
                {
                    "id": 1, "op_name": "add",
                    "inputs": [2, 3], "outputs": [4],
                    "sharding_rule": {
                        "equation": "m n, m n -> m n",
                        "factor_sizes": {"m": 1024, "n": 1024}
                    },
                    "compute_stats": {"flops": 1e6}
                }
            ]
        }
        
        sol = solver.solve(json.dumps(graph), debug=True)
        print("Bidirectional Solution:", json.dumps(sol, indent=2))
        
        # The output C is fixed to be sharded on dim 0.
        # This should propagate back:
        # - add's output (4) is fixed -> add's inputs (2, 3) should get sharded on dim 0
        # - matmul's output (2) sharded on dim 0 -> matmul should prefer DP (split M)
        
        # Check output of matmul (node 0) has dim 0 sharded
        matmul_out = sol["nodes"]["0"]["outputs"]["0"]["dims"]
        # Expectation: dim 0 sharded due to downstream constraint propagation
        # Note: This verifies bidirectional flow is working
        self.assertIsNotNone(matmul_out[0])  # Should have some sharding

    def test_integration_e2e(self):
        """Verify shard_map(auto_sharding=True) executes and produces correct results."""
        print("\nTEST: E2E Auto Sharding")
        
        def func(a, b):
            return a @ b
            
        # Large enough to trigger DP (1024x1024)
        M, K, N = 128, 128, 128
        
        sharded_fn = shard_map(
            func,
            self.mesh,
            in_specs={0: None, 1: None}, # Inputs start replicated
            out_specs=None,
            auto_sharding=True,
            debug=True
        )
        
        np.random.seed(42)
        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)
        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        
        # Execute with auto-sharding
        result = sharded_fn(a, b)
        
        # Verify numerical correctness against NumPy
        expected = a_np @ b_np
        result_np = result.to_numpy()
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-5,
            err_msg="Auto-sharded matmul produced incorrect results")
        
        print("✅ E2E test passed: Auto-sharded result matches expected!")

    def test_run_e2e(self):
         self.test_integration_e2e()

    def test_multi_op_chain(self):
        """Test auto-sharding with a chain of operations: matmul -> add -> reduce.
        
        This verifies that:
        1. Sharding propagates correctly through multiple ops
        2. Factor-based propagation handles different op types
        3. Communication ops are inserted where needed
        """
        print("\nTEST: Multi-Op Chain Auto-Sharding")
        
        def mlp_layer(x, w, b):
            # Matmul -> Add bias -> ReduceSum
            h = x @ w
            h = h + b
            return h.sum(axis=-1, keepdims=True)
        
        M, K, N = 128, 64, 32
        
        sharded_fn = shard_map(
            mlp_layer,
            self.mesh,
            in_specs={0: None, 1: None, 2: None},
            out_specs=None,
            auto_sharding=True,
            debug=True
        )
        
        np.random.seed(123)
        x_np = np.random.rand(M, K).astype(np.float32)
        w_np = np.random.rand(K, N).astype(np.float32)
        b_np = np.random.rand(N).astype(np.float32)
        
        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)
        b = Tensor.from_dlpack(b_np)
        
        result = sharded_fn(x, w, b)
        
        # NumPy reference
        expected = (x_np @ w_np + b_np).sum(axis=-1, keepdims=True)
        result_np = result.to_numpy()
        
        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-4,
            err_msg="Multi-op chain produced incorrect results")
        
        print("✅ Multi-op chain test passed!")

    def test_propagation_debug_output(self):
        """Verify debug output shows propagation iterations."""
        print("\nTEST: Propagation Debug Output")
        
        solver = SimpleSolver(self.mesh.shape, self.mesh.axis_names)
        
        # Two-node graph to force propagation across nodes
        graph = {
            "tensors": [
                {"id": 0, "shape": [1024, 1024]},  # x
                {"id": 1, "shape": [1024, 1024]},  # w
                {"id": 2, "shape": [1024, 1024]},  # h = x @ w
                {"id": 3, "shape": [1024, 1024]},  # b
                {"id": 4, "shape": [1024, 1024]},  # out = h + b
            ],
            "nodes": [
                {
                    "id": 0, "op_name": "matmul",
                    "inputs": [0, 1], "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 1024, "k": 1024, "n": 1024}
                    },
                    "compute_stats": {"flops": 2e9}
                },
                {
                    "id": 1, "op_name": "add",
                    "inputs": [2, 3], "outputs": [4],
                    "sharding_rule": {
                        "equation": "m n, m n -> m n",
                        "factor_sizes": {"m": 1024, "n": 1024}
                    },
                    "compute_stats": {"flops": 1e6}
                }
            ]
        }
        
        import io
        import sys
        
        # Capture debug output
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        
        solution = solver.solve(json.dumps(graph), debug=True)
        
        sys.stdout = old_stdout
        debug_output = captured.getvalue()
        
        print("Debug output:", debug_output[:500])
        
        # Verify propagation happened
        self.assertIn("Propagation iteration", debug_output)
        self.assertIn("Fixed-point reached", debug_output)
        
        # Verify solution has entries for both nodes
        self.assertIn("0", solution["nodes"])
        self.assertIn("1", solution["nodes"])
        
        print("✅ Propagation debug output verified!")

if __name__ == "__main__":
    unittest.main()
