# ===----------------------------------------------------------------------=== #
# Nabla 2026 - Robustness Tests for Graph Construction
# ===----------------------------------------------------------------------=== #

import unittest
import weakref
import gc
from typing import Any

import nabla
from nabla import Tensor
from nabla.core.compute_graph import GRAPH

class TestGraphConstructionRobustness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # We use explicit GC here only to isolate tests from each other.
        # This is standard test harness hygiene.
        gc.collect()
        self.initial_epoch = GRAPH.epoch

    async def test_full_evaluation_hygiene(self):
        """Test that evaluate() clears the graph completely."""
        x = Tensor.ones((2, 2))
        y = x + 1
        z = y * 2
        
        # Evaluate z. This triggers the internal graph compilation and flush.
        res = await z
        
        # After evaluation, the unrealized set should be empty because
        # everything needed was realized.
        self.assertEqual(len(GRAPH.unrealized), 0, "Graph should be empty after full chain eval")
        self.assertTrue(x.real)
        self.assertTrue(y.real)
        self.assertTrue(z.real)

    async def test_intermediate_reuse(self):
        """Test that intermediates needed for FUTURE computations are realized/cached."""
        x = Tensor.ones((2, 2))
        
        # 'temp' is used by both z (eval now) and w (eval later)
        temp = x + 1
        z = temp * 2
        w = temp * 3
        
        # 1. Evaluate ONLY z
        await z
        
        # Because 'temp' was kept alive (by w and local var), 
        # it should have been realized during z's evaluation.
        self.assertTrue(temp.real, "temp should be realized implicitly during z evaluation")
        
        # 2. Evaluate w
        # This should happen purely on realized inputs.
        await w
        self.assertTrue(w.real)

    async def test_cyclic_cleanup_on_eval(self):
        """Test that a loop creating many Tensors results in a clean graph after eval."""
        x = Tensor.ones((1,))
        
        # Create 100 intermediate tensors
        for _ in range(100):
            x = x + 1
            
        # At this point, there might be 100 dead tensors in memory (if no GC ran).
        # BUT, when we call evaluate(x), the system should:
        # 1. Run internal cleanup (gc.collect inside evaluate)
        # 2. Only build the graph for the LIVE 'x'.
        # 3. Flush/Empty dependencies.
        
        await x
        
        # Verify graph is clean
        self.assertEqual(len(GRAPH.unrealized), 0)
        self.assertTrue(x.real)

    async def test_independent_graphs_hygiene(self):
        """Evaluating one chain should not break another, and clean up handled ones."""
        # Chain A
        a = Tensor.ones((1,))
        a_out = a + 1
        
        # Chain B
        b = Tensor.ones((1,))
        b_out = b + 1
        
        # Evaluate A
        await a_out
        
        # Because of "Flush The World" strategy:
        # If 'b_out' is reachable, it MIGHT be realized too. 
        # But if 'b_out' was dead (not reachable), it would be dropped.
        # Here 'b_out' is reachable.
        
        self.assertTrue(a_out.real)
        self.assertTrue(b_out.real, "Everything accessible should be realized (Flush Strategy)")
        self.assertEqual(len(GRAPH.unrealized), 0)

    async def test_dead_branch_pruning(self):
        """Verify that UNUSED intermediates in a pure function are destroyed and NOT compiled."""
        
        def complex_function(val):
            # Path A (Used)
            res = val + 1
            
            # Path B (Dead/Unused)
            # These tensors are created but fall out of scope when function returns
            dead_1 = val * 999.0
            dead_2 = dead_1 - 5.0
            
            return res
            
        x = Tensor.ones((2, 2))
        y = complex_function(x)
        
        # At this point, dead_1 and dead_2 are technically "leaked" into unrealized 
        # because of Python scoping lag, BUT evaluate() has gc.collect().
        
        # We capture the number of outputs from the graph just before execution.
        # We can unfortunately only verify this by running it and knowing it succeeds without error,
        # OR by inspecting the graph state.
        # Let's inspect GRAPH.unrealized count *after* a manual GC here to assert "Pre-Eval Hygiene".
        
        gc.collect()
        
        # If pruning works:
        # unrealized should contain: x (ones), y (add). Count = 2.
        # It should NOT contain: mul (dead_1), sub (dead_2).
        
        # Filter constants
        unrealized_ops = [t._impl.op_name if hasattr(t._impl, "output_refs") and t._impl.output_refs else "leaf" 
                          for t in GRAPH.unrealized.values()]
        real_ops = [op for op in unrealized_ops if op not in ("constant", "seed")]
        
        # Expectation: 
        # - x (ones)
        # - y (add)
        # FAIL condition: if we see 'mul' or 'sub', then they leaked.
        
        self.assertNotIn("mul", real_ops, "Unused 'mul' op leaked into graph compilation!")
        self.assertNotIn("sub", real_ops, "Unused 'sub' op leaked into graph compilation!")
        self.assertEqual(len(real_ops), 2, f"Expected 2 ops (ones, add), found {real_ops}")
        
        # Run it to be sure
        await y
        self.assertTrue(y.real)

if __name__ == '__main__':
    unittest.main()
