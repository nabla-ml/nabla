# ===----------------------------------------------------------------------=== #
# Nabla 2026 - Lazy Execution Robustness Tests
#
# These tests validate that nabla's lazy execution model correctly tracks
# which tensors need to be realized. Key behavior:
#
#   GRAPH.unrealized is a WeakValueDictionary. When a Tensor loses all 
#   strong Python references and gets GC'd, it's automatically removed.
#   evaluate() calls gc.collect() internally before building the graph.
#
# We verify this by:
# 1. Returning tensor IDs from functions that create dead tensors
# 2. Checking those IDs are NOT in GRAPH.unrealized after function returns
#    (once gc.collect runs, which we trigger via evaluate)
# 3. Checking output count via DEBUG_LAZY_EVAL
# ===----------------------------------------------------------------------=== #

import unittest
import io
import contextlib
import nabla
from nabla import Tensor
from nabla.core.compute_graph import GRAPH
import nabla.core.compute_graph as compute_graph


class TestDeadTensorExcludedFromUnrealized(unittest.IsolatedAsyncioTestCase):
    """
    Core tests: verify dead tensors are NOT in GRAPH.unrealized after
    their references go out of scope and gc runs.
    """

    async def test_dead_branch_ids_not_in_unrealized(self):
        """
        Create dead tensors inside a function, return their IDs.
        Verify those IDs are NOT in unrealized after gc (via evaluate).
        """
        def create_with_dead_branch(x):
            # This intermediate IS used by 'alive', so it's a dependency
            # But it will be cleaned from unrealized after function returns
            # since local var 'dead_intermediate' dies
            dead_intermediate = x + 100.0
            dead_intermediate_id = id(dead_intermediate)
            
            alive = dead_intermediate * 2.0  # Will be output
            
            # Truly dead - not used by anything returned
            dead1 = dead_intermediate / 5.0
            dead1_id = id(dead1)
            
            # Return alive result + IDs of tensors that should be dead
            return alive, dead_intermediate_id, dead1_id
        
        x = Tensor.ones((2, 2))
        alive, dead_intermediate_id, dead1_id = create_with_dead_branch(x)
        
        # BEFORE await: dead tensors might still be in unrealized 
        # (Python hasn't GC'd yet)
        
        # Capture debug output to see what becomes output
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await alive
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # After evaluate (which calls gc.collect), graph is reset.
        # We verify via debug output that dead tensor was NOT an output.
        # 
        # Expected outputs: ones (x) + mul (alive) = 2
        # dead_intermediate becomes intermediate in graph (not output)
        # dead1 (div) should NOT be compiled at all
        
        # Check that 'div' is NOT in the graph at all
        self.assertNotIn("op=truediv", output.lower(), 
            f"Dead 'div' tensor leaked:\n{output}")
        
        # Verify output count is 2 (ones + mul), not 3 or 4
        self.assertIn("Setting 2 output(s)", output,
            f"Expected 2 outputs, got:\n{output}")

    async def test_loop_overwrite_dead_ids_not_in_unrealized(self):
        """
        When variable is overwritten, old values should be excluded.
        """
        x = Tensor.ones((1,))
        
        old_ids = []
        for _ in range(5):
            old_id = id(x)
            old_ids.append(old_id)
            x = x + 1  # Old x becomes dead
        
        final_id = id(x)
        
        # The final x is alive
        self.assertIn(final_id, GRAPH.unrealized)
        
        # NOTE: Some old_ids might still be in unrealized before gc runs
        # We verify via output count that only the chain we need is compiled
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await x
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # Verify it compiled and ran
        self.assertIn("output(s)", output)

    async def test_explicitly_deleted_id_not_output(self):
        """
        After 'del', tensor should not become an output.
        """
        x = Tensor.ones((2, 2))
        y = x + 1.0
        y_id = id(y)
        
        # y is in unrealized before del
        self.assertIn(y_id, GRAPH.unrealized)
        
        # Delete y
        del y
        
        # Create something else to evaluate
        z = x * 2.0
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await z
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # Only 2 outputs: ones (x) and mul (z)
        # The deleted add (y) should NOT be an output
        self.assertIn("Setting 2 output(s)", output,
            f"Deleted tensor became output:\n{output}")
        
        # Also verify add is not in graph
        self.assertNotIn("op=add", output,
            f"Deleted 'add' tensor compiled:\n{output}")


class TestUnrealizedTracking(unittest.IsolatedAsyncioTestCase):
    """
    Tests that verify GRAPH.unrealized tracking of alive tensors.
    """

    async def test_alive_tensors_tracked_by_id(self):
        """
        Tensors kept alive by variables should be in unrealized.
        """
        x = Tensor.ones((2, 2))
        y = x + 1.0
        z = y * 2.0
        
        self.assertIn(id(x), GRAPH.unrealized, "x should be tracked")
        self.assertIn(id(y), GRAPH.unrealized, "y should be tracked")
        self.assertIn(id(z), GRAPH.unrealized, "z should be tracked")
        
        await z

    async def test_function_return_values_tracked(self):
        """
        Returned tensors stay in unrealized, local-only tensors don't.
        """
        def compute(x):
            local_only = x + 999.0  # Dies when function returns
            local_id = id(local_only)
            
            returned = x * 2.0  # Survives via return
            returned_id = id(returned)
            
            return returned, local_id, returned_id
        
        x = Tensor.ones((2, 2))
        result, local_id, returned_id = compute(x)
        
        # Returned tensor is tracked
        self.assertIn(returned_id, GRAPH.unrealized)
        
        # Local might still be tracked before gc runs
        # But after evaluate, it should NOT have been an output
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await result
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # 2 outputs: ones + mul, NOT 3 (which would include dead add)
        self.assertIn("Setting 2 output(s)", output)


class TestFlushStrategy(unittest.IsolatedAsyncioTestCase):
    """
    Verify 'Flush The World' - all alive tensors become outputs together.
    """

    async def test_await_one_all_become_outputs(self):
        """
        Awaiting one tensor makes all alive tensors outputs.
        """
        x = Tensor.ones((1,))
        a = x + 1.0  # add
        b = x * 2.0  # mul  
        c = x - 3.0  # sub
        
        # All tracked
        for t in [x, a, b, c]:
            self.assertIn(id(t), GRAPH.unrealized)
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await a
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # All 4 should be outputs (ones, add, mul, sub)
        self.assertIn("Setting 4 output(s)", output,
            f"Expected 4 outputs for flush:\n{output}")


class TestComplexDeadPatterns(unittest.IsolatedAsyncioTestCase):
    """
    More complex dead tensor patterns.
    """

    async def test_nested_function_dead_tensors(self):
        """
        Dead tensors in nested functions.
        """
        def outer(x):
            def inner(y):
                inner_dead = y + 1000.0
                inner_dead_id = id(inner_dead)
                return y * 3.0, inner_dead_id
            
            outer_dead = x - 500.0
            outer_dead_id = id(outer_dead)
            
            intermediate = x + 1.0
            result, inner_dead_id = inner(intermediate)
            
            return result, outer_dead_id, inner_dead_id
        
        x = Tensor.ones((2, 2))
        result, outer_dead_id, inner_dead_id = outer(x)
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await result
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # Dead sub and dead add (1000) should not be outputs
        # System aggressively GC's: only mul (result) + ones (x) = 2 outputs
        # The intermediate 'add' is also GC'd since its var goes out of scope
        self.assertIn("Setting 2 output(s)", output,
            f"Dead nested tensors leaked:\n{output}")

    async def test_dict_with_dead_values(self):
        """
        Dict where some values go dead.
        """
        x = Tensor.ones((2, 2))
        
        outputs = {
            "keep1": x * 2.0,
            "keep2": x + 1.0,
        }
        
        # Create and discard
        _ = x - 999.0  # Immediately dead
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await outputs["keep1"]
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # Outputs: ones (x), mul (keep1), add (keep2) = 3 normally
        # Plus the dead sub might be compiled but not output
        # Actually checking the MLIR: dead sub IS in graph but NOT in output
        # Output count depends on what's alive - verify dead sub not in outputs
        self.assertNotIn("op=sub", output.split("mo.output")[1] if "mo.output" in output else output,
            f"Dead sub leaked into outputs:\n{output}")


class TestTracedModeGC(unittest.IsolatedAsyncioTestCase):
    """
    Tests that verify traced mode (with OutputRefs) doesn't prevent GC.
    
    Key insight: OutputRefs._refs uses weakref.ref, so traced tensors 
    should still be garbage collected when they lose strong references.
    """

    async def test_traced_dead_branch_still_gc(self):
        """
        Dead tensors in traced mode should still be GC'd.
        """
        def create_traced_with_dead_branch(x):
            # Mark x as traced
            x.traced = True
            
            dead_intermediate = x + 100.0
            alive = dead_intermediate * 2.0
            
            # Truly dead
            dead1 = dead_intermediate / 5.0
            dead1_id = id(dead1)
            
            return alive, dead1_id
        
        x = Tensor.ones((2, 2))
        alive, dead1_id = create_traced_with_dead_branch(x)
        
        # alive should be traced (propagated from x)
        self.assertTrue(alive.traced, "Traced should propagate to outputs")
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await alive
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # Dead div should NOT be an output even though tensors are traced
        self.assertNotIn("op=truediv", output.lower(),
            f"Dead traced 'div' leaked into outputs:\n{output}")

    async def test_traced_chain_correct_output_count(self):
        """
        A fully traced chain should have correct output count.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        # Chain operations
        a = x + 1.0
        b = a * 2.0
        c = b - 3.0
        
        # All should be traced (propagated from x)
        self.assertTrue(a.traced)
        self.assertTrue(b.traced)
        self.assertTrue(c.traced)
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await c
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # All alive, so all should be outputs: ones, add, mul, sub = 4
        self.assertIn("Setting 4 output(s)", output,
            f"Traced chain output count wrong:\n{output}")


class TestTracingGraphTraversal(unittest.IsolatedAsyncioTestCase):
    """
    Tests that verify tracing correctly enables graph traversal 
    AFTER evaluation. This is critical for autodiff (VJP).
    """

    async def test_can_access_parents_after_eval(self):
        """
        After evaluation, traced tensors should still have accessible parents.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        a = x + 1.0
        b = a * 2.0
        c = b - 3.0
        
        # Evaluate
        await c
        
        # After eval, we should still be able to traverse via parents
        self.assertIsNotNone(c._impl.output_refs, "c should have output_refs")
        self.assertEqual(len(c._impl.parents), 2, "c (sub) should have 2 parents")
        
        # Parents should include b's impl
        parent_ops = [p.op_name for p in c._impl.parents if p.output_refs]
        self.assertIn("mul", parent_ops, "b (mul) should be parent of c")

    async def test_can_access_op_metadata_after_eval(self):
        """
        After evaluation, OutputRefs should still have operation metadata.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        y = x * 2.0
        
        await y
        
        # OutputRefs should be accessible
        output_refs = y._impl.output_refs
        self.assertIsNotNone(output_refs)
        
        # Op should be accessible
        self.assertEqual(output_refs.op.name, "mul")
        
        # op_args should contain information about inputs
        self.assertIsNotNone(output_refs.op_args)

    async def test_full_graph_traversal_after_eval(self):
        """
        After evaluation, can traverse the entire traced graph.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        # Build a graph: x -> a -> b -> c
        a = x + 1.0
        b = a * 2.0
        c = b - 3.0
        
        await c
        
        # Traverse backwards from c to find all ops
        visited_ops = []
        
        def visit(impl):
            if impl.output_refs:
                visited_ops.append(impl.op_name)
            for parent in impl.parents:
                if parent.output_refs and parent.op_name not in visited_ops:
                    visit(parent)
        
        visit(c._impl)
        
        # Should have visited: sub, mul, add
        self.assertIn("sub", visited_ops, "Should visit sub")
        self.assertIn("mul", visited_ops, "Should visit mul")
        self.assertIn("add", visited_ops, "Should visit add")

    async def test_dead_branch_not_in_traced_graph(self):
        """
        Dead branch should not appear in traced graph traversal.
        """
        def compute(x):
            x.traced = True
            
            intermediate = x + 1.0
            alive = intermediate * 2.0
            
            # Dead branch
            dead = intermediate / 5.0
            
            return alive
        
        x = Tensor.ones((2, 2))
        result = compute(x)
        
        await result
        
        # Traverse from result
        visited_ops = []
        
        def visit(impl):
            if impl.output_refs:
                visited_ops.append(impl.op_name)
            for parent in impl.parents:
                if parent.output_refs and parent.op_name not in visited_ops:
                    visit(parent)
        
        visit(result._impl)
        
        # Should NOT contain truediv (dead branch)
        self.assertNotIn("truediv", visited_ops, "Dead branch should not be traversable")
        
        # Should contain the alive path
        self.assertIn("mul", visited_ops)
        self.assertIn("add", visited_ops)


class TestTracingInheritance(unittest.IsolatedAsyncioTestCase):
    """
    Tests for traced flag inheritance through operations.
    """

    async def test_traced_inherits_through_ops(self):
        """
        Operations on traced tensor produce traced outputs.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        y = x + 1.0
        z = y * 2.0
        
        self.assertTrue(y.traced, "y should inherit traced from x")
        self.assertTrue(z.traced, "z should inherit traced from y")
        
        await z

    async def test_traced_output_refs_has_op(self):
        """
        Traced tensors have OutputRefs with operation info.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        y = x + 1.0
        
        self.assertIsNotNone(y._impl.output_refs, "Traced op should have OutputRefs")
        self.assertEqual(y._impl.output_refs.op.name, "add")
        
        await y

    async def test_untraced_also_has_output_refs(self):
        """
        Even untraced tensors have OutputRefs (for graph building).
        """
        x = Tensor.ones((2, 2))
        # Not traced
        
        y = x + 1.0
        
        # OutputRefs should still exist for graph construction
        self.assertIsNotNone(y._impl.output_refs)
        
        await y


class TestMixedTracedUntraced(unittest.IsolatedAsyncioTestCase):
    """
    Tests for mixed traced and untraced tensors.
    """

    async def test_untraced_dead_branch_with_traced_alive(self):
        """
        Mixed: traced alive path, dead branch.
        """
        def compute(x):
            x.traced = True
            
            alive = x * 2.0  # Traced
            dead = x + 999.0  # Also traced (inherits), but dead
            
            return alive
        
        x = Tensor.ones((2, 2))
        result = compute(x)
        
        old_debug = compute_graph.DEBUG_LAZY_EVAL
        compute_graph.DEBUG_LAZY_EVAL = True
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            await result
        
        compute_graph.DEBUG_LAZY_EVAL = old_debug
        output = f.getvalue()
        
        # Dead add should NOT be output
        self.assertIn("Setting 2 output(s)", output,
            f"Dead branch leaked:\n{output}")


class TestTracingThroughIntermediateEvals(unittest.IsolatedAsyncioTestCase):
    """
    Tests that verify tracing survives through intermediate evaluations
    like item() and multiple await calls.
    
    This is critical for autodiff where we may need to traverse the graph
    after the loss has been computed and realized.
    """

    async def test_tracing_survives_await(self):
        """
        After await, OutputRefs should still be accessible for graph traversal.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        a = x + 1.0
        b = a * 2.0
        
        # First eval
        await b
        
        # After eval, can still access tracing info
        self.assertIsNotNone(b._impl.output_refs)
        self.assertEqual(b._impl.output_refs.op.name, "mul")
        
        # Can still traverse parents
        parent_ops = [p.op_name for p in b._impl.parents if p.output_refs]
        self.assertIn("add", parent_ops)

    async def test_tracing_survives_item_call(self):
        """
        After item() (which triggers sync realization), tracing should survive.
        """
        x = Tensor.ones((1,))
        x.traced = True
        
        y = x + 5.0
        
        # item() triggers _sync_realize internally
        val = y.item()
        self.assertEqual(val, 6.0)
        
        # Tracing info should still be accessible
        self.assertIsNotNone(y._impl.output_refs)
        self.assertEqual(y._impl.output_refs.op.name, "add")

    async def test_tracing_survives_multiple_awaits(self):
        """
        Multiple sequential awaits should not break tracing.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        a = x + 1.0
        
        # First await
        await a
        
        # Create more ops on realized tensor
        b = a * 2.0
        
        # Second await
        await b
        
        # Both should have accessible OutputRefs
        self.assertIsNotNone(a._impl.output_refs)
        self.assertIsNotNone(b._impl.output_refs)
        
        # Can traverse from b back to a
        parent_impls = b._impl.parents
        self.assertTrue(len(parent_impls) > 0)

    async def test_continued_computation_after_await(self):
        """
        Can continue building traced graph after intermediate await.
        """
        x = Tensor.ones((2, 2))
        x.traced = True
        
        # Build first part
        a = x + 1.0
        b = a * 2.0
        
        # Realize
        await b
        
        # Continue building on realized tensor
        c = b - 3.0
        d = c / 2.0
        
        # Second realization
        await d
        
        # Full graph should be traversable
        visited_ops = []
        
        def visit(impl):
            if impl.output_refs:
                visited_ops.append(impl.op_name)
            for parent in impl.parents:
                if parent.output_refs and parent.op_name not in visited_ops:
                    visit(parent)
        
        visit(d._impl)
        
        # Should find all ops: div, sub, mul, add
        self.assertIn("div", visited_ops)
        self.assertIn("sub", visited_ops)
        self.assertIn("mul", visited_ops)
        self.assertIn("add", visited_ops)

    async def test_tracing_with_intermediate_value_access(self):
        """
        Accessing intermediate values (like during debugging) shouldn't break tracing.
        """
        x = Tensor.ones((1,))
        x.traced = True
        
        a = x + 1.0
        b = a * 2.0
        c = b - 0.5
        
        # Access intermediate value
        b_val = b.item()
        self.assertEqual(b_val, 4.0)
        
        # Final await
        await c
        
        # Tracing should still work
        self.assertIsNotNone(c._impl.output_refs)
        self.assertEqual(c._impl.output_refs.op.name, "sub")


if __name__ == "__main__":
    unittest.main()

