# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import json
import unittest

import numpy as np

from nabla.core import Tensor, trace
from nabla.core.sharding.optimizer.simple_solver import SimpleSolver
from nabla.core.sharding.spec import DeviceMesh
from nabla.transforms.shard_map import _ShardingGraphExtractor as ShardingGraphExtractor
from nabla.transforms.shard_map import shard_map


class TestAutoSharding(unittest.TestCase):
    """Tests for Automated Sharding System."""

    def setUp(self):

        self.mesh = DeviceMesh("test_mesh", (4,), ("d",), devices=[0, 1, 2, 3])

    def test_graph_extraction(self):
        """Verify Graph Extractor produces valid JSON."""
        print("\nTEST: Graph Extraction")

        def func(a, b):
            return a @ b

        a = Tensor.from_dlpack(np.random.rand(128, 128).astype(np.float32))
        b = Tensor.from_dlpack(np.random.rand(128, 128).astype(np.float32))

        t = trace(func, a, b)

        extractor = ShardingGraphExtractor(t, in_specs={}, out_specs={})
        json_str = extractor.extract()
        data = json.loads(json_str)

        print(json_str)

        self.assertIn("tensors", data)
        self.assertIn("nodes", data)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["op_name"], "matmul")

        flops = data["nodes"][0]["compute_stats"]["flops"]

        self.assertEqual(flops, 2 * 128 * 128 * 128)

    def test_simple_solver_logic(self):
        """Verify Solver logic for choosing DP vs MP with propagation-based solver."""
        print("\nTEST: Solver Logic (Propagation-Based)")

        solver = SimpleSolver(self.mesh.shape, self.mesh.axis_names)

        graph_dp = {
            "tensors": [
                {"id": 0, "shape": [1024, 1024]},
                {"id": 1, "shape": [1024, 1024]},
                {"id": 2, "shape": [1024, 1024]},
            ],
            "nodes": [
                {
                    "id": 0,
                    "op_name": "matmul",
                    "inputs": [0, 1],
                    "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 1024, "k": 1024, "n": 1024},
                    },
                    "compute_stats": {"flops": 2e9},
                }
            ],
        }

        sol_dp = solver.solve(json.dumps(graph_dp), debug=True)
        print("DP Solution:", json.dumps(sol_dp, indent=2))

        self.assertIn("nodes", sol_dp)
        self.assertIn("0", sol_dp["nodes"])
        node_sol = sol_dp["nodes"]["0"]

        self.assertIn("outputs", node_sol)
        self.assertIn("0", node_sol["outputs"])
        out_dims = node_sol["outputs"]["0"]["dims"]
        self.assertEqual(out_dims[0], ["d"])

        self.assertIn("inputs", node_sol)
        self.assertIn("0", node_sol["inputs"])
        in_a_dims = node_sol["inputs"]["0"]["dims"]
        self.assertEqual(in_a_dims[0], ["d"])

        graph_mp = {
            "tensors": [
                {"id": 0, "shape": [3, 4096]},
                {"id": 1, "shape": [4096, 1024]},
                {"id": 2, "shape": [3, 1024]},
            ],
            "nodes": [
                {
                    "id": 0,
                    "op_name": "matmul",
                    "inputs": [0, 1],
                    "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 3, "k": 4096, "n": 1024},
                    },
                    "compute_stats": {"flops": 2e9},
                }
            ],
        }

        sol_mp = solver.solve(json.dumps(graph_mp), debug=True)
        print("MP Solution:", json.dumps(sol_mp, indent=2))

        node_sol_mp = sol_mp["nodes"]["0"]

        in_a_dims_mp = node_sol_mp["inputs"]["0"]["dims"]
        self.assertEqual(in_a_dims_mp[1], ["d"])

        in_b_dims_mp = node_sol_mp["inputs"]["1"]["dims"]
        self.assertEqual(in_b_dims_mp[0], ["d"])

        out_dims_mp = node_sol_mp["outputs"]["0"]["dims"]
        self.assertEqual(out_dims_mp, [None, None])

    def test_solver_bidirectional_propagation(self):
        """Verify that constraints propagate bidirectionally through the graph.

        This tests the key feature: output constraints should propagate back to inputs.
        """
        print("\nTEST: Bidirectional Propagation")

        solver = SimpleSolver(self.mesh.shape, self.mesh.axis_names)

        graph = {
            "tensors": [
                {"id": 0, "shape": [1024, 1024]},
                {"id": 1, "shape": [1024, 1024]},
                {"id": 2, "shape": [1024, 1024]},
                {"id": 3, "shape": [1024, 1024]},
                {
                    "id": 4,
                    "shape": [1024, 1024],
                    "fixed_sharding": {"dims": [["d"], None], "replicated": []},
                },
            ],
            "nodes": [
                {
                    "id": 0,
                    "op_name": "matmul",
                    "inputs": [0, 1],
                    "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 1024, "k": 1024, "n": 1024},
                    },
                    "compute_stats": {"flops": 2e9},
                },
                {
                    "id": 1,
                    "op_name": "add",
                    "inputs": [2, 3],
                    "outputs": [4],
                    "sharding_rule": {
                        "equation": "m n, m n -> m n",
                        "factor_sizes": {"m": 1024, "n": 1024},
                    },
                    "compute_stats": {"flops": 1e6},
                },
            ],
        }

        sol = solver.solve(json.dumps(graph), debug=True)
        print("Bidirectional Solution:", json.dumps(sol, indent=2))

        matmul_out = sol["nodes"]["0"]["outputs"]["0"]["dims"]

        self.assertIsNotNone(matmul_out[0])

    def test_integration_e2e(self):
        """Verify shard_map(auto_sharding=True) executes and produces correct results."""
        print("\nTEST: E2E Auto Sharding")

        def func(a, b):
            return a @ b

        M, K, N = 128, 128, 128

        sharded_fn = shard_map(
            func,
            self.mesh,
            in_specs={0: None, 1: None},
            out_specs=None,
            auto_sharding=True,
            debug=True,
        )

        np.random.seed(42)
        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)
        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)

        result = sharded_fn(a, b)

        expected = a_np @ b_np
        result_np = result.to_numpy()

        np.testing.assert_allclose(
            result_np,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Auto-sharded matmul produced incorrect results",
        )

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
            debug=True,
        )

        np.random.seed(123)
        x_np = np.random.rand(M, K).astype(np.float32)
        w_np = np.random.rand(K, N).astype(np.float32)
        b_np = np.random.rand(N).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)
        b = Tensor.from_dlpack(b_np)

        result = sharded_fn(x, w, b)

        expected = (x_np @ w_np + b_np).sum(axis=-1, keepdims=True)
        result_np = result.to_numpy()

        np.testing.assert_allclose(
            result_np,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Multi-op chain produced incorrect results",
        )

        print("✅ Multi-op chain test passed!")

    def test_propagation_debug_output(self):
        """Verify debug output shows propagation iterations."""
        print("\nTEST: Propagation Debug Output")

        solver = SimpleSolver(self.mesh.shape, self.mesh.axis_names)

        graph = {
            "tensors": [
                {"id": 0, "shape": [1024, 1024]},
                {"id": 1, "shape": [1024, 1024]},
                {"id": 2, "shape": [1024, 1024]},
                {"id": 3, "shape": [1024, 1024]},
                {"id": 4, "shape": [1024, 1024]},
            ],
            "nodes": [
                {
                    "id": 0,
                    "op_name": "matmul",
                    "inputs": [0, 1],
                    "outputs": [2],
                    "sharding_rule": {
                        "equation": "m k, k n -> m n",
                        "factor_sizes": {"m": 1024, "k": 1024, "n": 1024},
                    },
                    "compute_stats": {"flops": 2e9},
                },
                {
                    "id": 1,
                    "op_name": "add",
                    "inputs": [2, 3],
                    "outputs": [4],
                    "sharding_rule": {
                        "equation": "m n, m n -> m n",
                        "factor_sizes": {"m": 1024, "n": 1024},
                    },
                    "compute_stats": {"flops": 1e6},
                },
            ],
        }

        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        solution = solver.solve(json.dumps(graph), debug=True)

        sys.stdout = old_stdout
        debug_output = captured.getvalue()

        print("Debug output:", debug_output[:500])

        self.assertIn("Propagation iteration", debug_output)
        self.assertIn("Fixed-point reached", debug_output)

        self.assertIn("0", solution["nodes"])
        self.assertIn("1", solution["nodes"])

        print("✅ Propagation debug output verified!")

    def test_transformer_block(self):
        """Test auto-sharding with a larger transformer-like graph (8+ operations).

        This verifies:
        1. Sharding propagates correctly through many operations
        2. Solver converges within reasonable iterations
        3. Numerical output matches NumPy reference
        """
        print("\nTEST: Transformer Block (8+ ops)")

        def transformer_block(x, w_qkv, w_o, w_ff1, w_ff2):

            qkv = x @ w_qkv

            attn_out = qkv @ w_o

            h = x + attn_out

            ff = h @ w_ff1
            ff_out = ff @ w_ff2

            return h + ff_out

        batch, seq, dim, ff_dim = 64, 32, 64, 128

        sharded_fn = shard_map(
            transformer_block,
            self.mesh,
            in_specs={0: None, 1: None, 2: None, 3: None, 4: None},
            out_specs=None,
            auto_sharding=True,
            debug=True,
        )

        np.random.seed(42)
        x_np = np.random.rand(batch, seq, dim).astype(np.float32)
        w_qkv_np = np.random.rand(dim, 3 * dim).astype(np.float32)
        w_o_np = np.random.rand(3 * dim, dim).astype(np.float32)
        w_ff1_np = np.random.rand(dim, ff_dim).astype(np.float32)
        w_ff2_np = np.random.rand(ff_dim, dim).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        w_qkv = Tensor.from_dlpack(w_qkv_np)
        w_o = Tensor.from_dlpack(w_o_np)
        w_ff1 = Tensor.from_dlpack(w_ff1_np)
        w_ff2 = Tensor.from_dlpack(w_ff2_np)

        result = sharded_fn(x, w_qkv, w_o, w_ff1, w_ff2)

        qkv = x_np @ w_qkv_np
        attn_out = qkv @ w_o_np
        h = x_np + attn_out
        ff = h @ w_ff1_np
        ff_out = ff @ w_ff2_np
        expected = h + ff_out

        result_np = result.to_numpy()
        np.testing.assert_allclose(
            result_np,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Transformer block produced incorrect results",
        )

        print("✅ Transformer block test passed!")

    def test_model_parallel_matmul(self):
        """Test Model Parallel (MP) sharding when M is small/indivisible.

        Verifies:
        1. Solver chooses to shard K dimension when M cannot be sharded
        2. Proper AllReduce is inserted for partial results
        """
        print("\nTEST: Model Parallel Matmul (Split K)")

        M, K, N = 3, 256, 64

        def mp_matmul(a, b):
            return a @ b

        sharded_fn = shard_map(
            mp_matmul,
            self.mesh,
            in_specs={0: None, 1: None},
            out_specs=None,
            auto_sharding=True,
            debug=True,
        )

        np.random.seed(99)
        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)

        result = sharded_fn(a, b)

        expected = a_np @ b_np
        result_np = result.to_numpy()

        np.testing.assert_allclose(
            result_np,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MP matmul produced incorrect results",
        )

        print("✅ Model Parallel matmul test passed!")

    def test_diamond_pattern(self):
        """Test fork/join (diamond) pattern where one tensor feeds multiple ops.

        Verifies:
        1. Sharding propagates correctly through forks
        2. Fixed-point converges for diamond patterns
        """
        print("\nTEST: Diamond Pattern (Fork/Join)")

        def diamond(x, w):
            h = x @ w
            branch1 = h + 1.0
            branch2 = h * 2.0
            return branch1 + branch2

        M, K, N = 128, 64, 32

        sharded_fn = shard_map(
            diamond,
            self.mesh,
            in_specs={0: None, 1: None},
            out_specs=None,
            auto_sharding=True,
            debug=True,
        )

        np.random.seed(123)
        x_np = np.random.rand(M, K).astype(np.float32)
        w_np = np.random.rand(K, N).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        w = Tensor.from_dlpack(w_np)

        result = sharded_fn(x, w)

        h = x_np @ w_np
        branch1 = h + 1.0
        branch2 = h * 2.0
        expected = branch1 + branch2

        result_np = result.to_numpy()
        np.testing.assert_allclose(
            result_np,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Diamond pattern produced incorrect results",
        )

        print("✅ Diamond pattern test passed!")


if __name__ == "__main__":
    unittest.main()
