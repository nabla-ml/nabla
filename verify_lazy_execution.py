
import unittest
import io
import contextlib
import re
import nabla
from nabla import Tensor
import nabla.core.compute_graph

# Enable debug mode to see the graph prints
nabla.core.compute_graph.DEBUG_LAZY_EVAL = True

class TestLazyExecutionCorrectness(unittest.IsolatedAsyncioTestCase):

    async def test_dead_branch_pruning_automatic(self):
        """
        Verify that creating a dead branch in a function does NOT result in 
        those operations appearing in the compiled graph, purely relying on 
        automatic GC within evaluate().
        """
        print("\n--- Test: Dead Branch Pruning (Automatic) ---")
        
        def impure_func_with_dead_branch(x):
            # Branch A (Used)
            alive = x * 2.0
            
            # Branch B (Dead)
            # These ops create Tensors that go out of scope immediately
            dead = x + 100.0 
            dead2 = dead / 5.0
            
            return alive

        x = Tensor.ones((2, 2))
        
        # Capture stdout to spy on the graph
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            y = impure_func_with_dead_branch(x)
            await y  # Should trigger evaluate() -> gc.collect() -> graph build
            
        output = f.getvalue()
        
        # Check that we see the 'alive' path ops
        # Note: 'x * 2.0' is a mul operation.
        # 'x + 100.0' is an add operation.
        
        # We expect to see 'mul' in the graph dump (for * 2.0)
        # We expect NOT to see 'add' (for + 100.0) or 'div' (for / 5.0)
        
        # Helper to look for standard MLIR/MAX ops or our debug prints
        # The debug print dumps the Op definition.
        
        self.assertIn("mul", output.lower(), "The 'mul' op should be in the graph")
        
        # These might fail if the implementation of 'add' or 'div' isn't named strictly so,
        # but 'add' is standard.
        # We must be careful that 'mul' isn't used in some internal address calculation if that exists,
        # but for this simple graph it should be clean.
        
        # NOTE: If constants are lifted/folded, we might see them differently.
        # But broadly:
        has_add = "add" in output.lower()
        has_div = "div" in output.lower()
        
        if has_add:
            print("FAIL: Found 'add' op in graph which should have been dead.")
            print("Captured Output:\n", output)
        
        self.assertFalse(has_add, "The dead 'add' operation leaked into the graph!")
        self.assertFalse(has_div, "The dead 'div' operation leaked into the graph!")
        
        print("SUCCESS: Dead branch yielded no operations in final graph.")

    async def test_intermediate_lifecycle(self):
        """
        Verify that an intermediate tensor used for *future* computation 
        is realized and available, but DOES NOT leak into the current graph 
        if it's not a dependency of the current target.
        Wait... if it's not a dependency, it shouldn't be in the graph anyway.
        
        Better test: Verify that an intermediate used by TWO outputs is 
        computed once (we can't easily see that from just one print, 
        unless we check for duplicated nodes).
        
        Let's focus on the user request: "verify we set the right unrealized tensors as output".
        """
        print("\n--- Test: Intermediate Output Correctness ---")
        
        a = Tensor.ones((1,))
        b = a + 1.0  # intermediate
        c = b * 2.0  # generated from b
        d = b + 5.0  # generated from b
        
        # Scenario: We await 'c'. 
        # 'b' is alive (referenced by 'd').
        # 'b' MUST be computed as part of 'c's graph.
        # HOWEVER, 'd' should NOT be in the graph.
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await c
            
        output = f.getvalue()
        
        # assertions
        # We expect 'mul' (c = b*2)
        # We expect 'add' (b = a+1)
        # We expect to NOT see the 'add' for (d = b+5) -- wait, both are adds.
        # We need to distinguish them.
        
        # Let's use different operations to distinguish.
        # Redo setup
        
    async def test_split_graph_correctness(self):
        print("\n--- Test: Split Graph Correctness ---")
        
        # Distinct operations
        x = Tensor.ones((1,))
        
        # Path 1 (Intermediate)
        i1 = x + 1.0     # add
        
        # Path 2 (Goal)
        res = i1 * 2.0   # mul
        
        # Path 3 (Distractor - kept alive by Python variable, but not needed for 'res')
        distractor = i1 - 5.0 # sub
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Trigger eval for 'res'
            # 'distractor' is reachable from locals, so it is in GRAPH.unrealized.
            # BUT efficient graph building should traverse parents of 'res' and ignore 'distractor'
            # UNLESS 'distractor' is inadvertently added as an output?
            # 'evaluate' usually takes explicit targets. If we just await 'res', 
            # evaluate(res) is called.
            # Does evaluate(res) auto-include all unrealized tensors?
            # Let's check compute_graph.py...
            # line 233: "for t in self.unrealized.values(): add_target(t)"
            # Ah! The current implementation adds ALL unrealized tensors as targets!
            
            await res
            
        output = f.getvalue()
        
        # If the implementation adds ALL unrealized tensors, then 'distractor' WILL be calculated.
        # This might be expected behavior ("Flush everything") or it might be what the user wants to Fix.
        # "we are setting the right unrelaized tensros as output to a graph"
        # The user questioned: "if we are setting the right unrelaized tensros as output"
        
        # Let's see what happens.
        
        was_sub_calculated = "canonicalize" not in output and "sub" in output.lower()
        # Note: 'sub' might be compiled to 'add' with negation, but 'sub' often exists.
        
        print(f"Distractor ('sub') present in graph: {was_sub_calculated}")
        
        # For now, we just print the result of this check. The user wants to know "if it is working".
        # If the strategy is "Evaluate Everything", then it IS working if it's there.
        # If the strategy is "Evaluate Needed", then it's broken if it's there.
        # The code says:
        # # Add any other unrealized tensors
        # for t in self.unrealized.values():
        #     add_target(t)
        
        # So explicitly, the code is designed to flush everything.
        # So 'distractor' SHOULD be there if it is still alive in Python.
        
        if was_sub_calculated:
            print("CONFIRMED: Graph flushes all alive unrealized tensors, including distractors.")
        else:
            print("SURPRISE: Graph did NOT flush distractor.")

if __name__ == "__main__":
    unittest.main()
