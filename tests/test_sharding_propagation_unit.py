"""Unit tests for sharding propagation logic (factor-based, XLA Shardy style).

These tests directly verify OpShardingRule templates and propagate_sharding()
without creating actual Tensor objects or running operations.
"""

import pytest
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.sharding.propagation import (
    OpShardingRule,
    propagate_sharding,
    matmul_template,
    elementwise_template,
    reduce_template,
    PropagationStrategy,
)


# ============================================================================
# Part 1: Matmul Template Tests
# ============================================================================

class TestMatmulTemplate:
    """Test matmul_template creates correct factor mappings."""
    
    def test_matmul_2d_factor_mapping(self):
        """Matmul 2D: (m, k) @ (k, n) -> (m, n)."""
        template = matmul_template(batch_dims=0)
        rule = template.instantiate(
            input_shapes=[(4, 8), (8, 2)],
            output_shapes=[(4, 2)]
        )
        
        # Verify factor mappings
        assert rule.input_mappings[0] == {0: ["m"], 1: ["k"]}  # A: (m, k)
        assert rule.input_mappings[1] == {0: ["k"], 1: ["n"]}  # B: (k, n)
        assert rule.output_mappings[0] == {0: ["m"], 1: ["n"]}  # C: (m, n)
        
        # Verify factor sizes
        assert rule.factor_sizes == {"m": 4, "k": 8, "n": 2}
        
        # Verify einsum notation
        assert rule.to_einsum_notation() == "(m, k), (k, n) -> (m, n)"
    
    def test_matmul_with_batch_dims(self):
        """Matmul with batch: (b, m, k) @ (b, k, n) -> (b, m, n)."""
        template = matmul_template(batch_dims=1)
        rule = template.instantiate(
            input_shapes=[(2, 4, 8), (2, 8, 3)],
            output_shapes=[(2, 4, 3)]
        )
        
        # Verify factor mappings include batch
        assert rule.input_mappings[0] == {0: ["b0"], 1: ["m"], 2: ["k"]}
        assert rule.input_mappings[1] == {0: ["b0"], 1: ["k"], 2: ["n"]}
        assert rule.output_mappings[0] == {0: ["b0"], 1: ["m"], 2: ["n"]}
        
        # Verify factor sizes
        assert rule.factor_sizes == {"b0": 2, "m": 4, "k": 8, "n": 3}
    
    def test_matmul_contracting_factors(self):
        """Verify k is identified as contracting factor."""
        template = matmul_template(batch_dims=0)
        rule = template.instantiate(
            input_shapes=[(4, 8), (8, 2)],
            output_shapes=[(4, 2)]
        )
        
        contracting = rule.get_contracting_factors()
        assert contracting == {"k"}


# ============================================================================
# Part 2: Matmul Propagation Tests
# ============================================================================

class TestMatmulPropagation:
    """Test sharding propagation through matmul operations."""
    
    def test_forward_row_sharding(self):
        """A[m=x, k] @ B[k, n] -> C[m=x, n] (row-parallel)."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A sharded on first dim (m factor)
        a_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # m -> sharded on x
            DimSpec([], is_open=True)        # k -> replicated
        ])
        
        # B replicated
        b_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # k
            DimSpec([], is_open=True)   # n
        ])
        
        # C initially open
        c_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # m
            DimSpec([], is_open=True)   # n
        ])
        
        # Create rule and propagate
        template = matmul_template(batch_dims=0)
        rule = template.instantiate(
            input_shapes=[(4, 8), (8, 2)],
            output_shapes=[(4, 2)]
        )
        
        changed = propagate_sharding(rule, [a_spec, b_spec], [c_spec])
        
        # Verify C got m sharding from A
        assert changed
        assert c_spec.dim_specs[0].axes == ["x"]  # m factor has x
        assert c_spec.dim_specs[1].axes == []     # n factor has nothing
    
    def test_forward_col_sharding(self):
        """A[m, k] @ B[k, n=x] -> C[m, n=x] (col-parallel)."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A replicated
        a_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True)
        ])
        
        # B sharded on second dim (n factor)
        b_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec(["x"], is_open=False)  # n -> sharded on x
        ])
        
        # C initially open
        c_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True)
        ])
        
        template = matmul_template(batch_dims=0)
        rule = template.instantiate(
            input_shapes=[(4, 8), (8, 2)],
            output_shapes=[(4, 2)]
        )
        
        changed = propagate_sharding(rule, [a_spec, b_spec], [c_spec])
        
        # Verify C got n sharding from B
        assert changed
        assert c_spec.dim_specs[0].axes == []     # m has nothing
        assert c_spec.dim_specs[1].axes == ["x"]  # n factor has x
    
    def test_backward_to_contracting_dim(self):
        """If contracting dim k is sharded, both inputs should get it."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # A has k sharded
        a_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec(["x"], is_open=False)  # k sharded
        ])
        
        # B initially open
        b_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # k
            DimSpec([], is_open=True)
        ])
        
        c_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True)
        ])
        
        template = matmul_template(batch_dims=0)
        rule = template.instantiate(
            input_shapes=[(4, 8), (8, 2)],
            output_shapes=[(4, 2)]
        )
        
        changed = propagate_sharding(rule, [a_spec, b_spec], [c_spec])
        
        # Verify B got k sharding from A (factor k propagates)
        assert changed
        assert b_spec.dim_specs[0].axes == ["x"]  # k factor


# ============================================================================
# Part 3: Conflict Resolution Tests
# ============================================================================

class TestConflictResolution:
    """Test BASIC vs AGGRESSIVE conflict resolution strategies."""
    
    def test_basic_conflict_takes_common_prefix(self):
        """BASIC strategy: conflicting axes -> take common prefix."""
        mesh = DeviceMesh("test", (4,), ("x",))
        
        # Elementwise: both inputs contribute to same factor d0
        # Input 1: d0 sharded on "x"
        # Input 2: d0 sharded on different (but we only have "x")
        # Let's simulate by having different priority levels
        
        spec1 = ShardingSpec(mesh, [DimSpec(["x"], is_open=True, priority=0)])
        spec2 = ShardingSpec(mesh, [DimSpec([], is_open=True, priority=0)])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = elementwise_template(rank=1)
        rule = template.instantiate(
            input_shapes=[(4,), (4,)],
            output_shapes=[(4,)]
        )
        
        # With BASIC, if inputs have same priority, common prefix is taken
        propagate_sharding(rule, [spec1, spec2], [out_spec], strategy=PropagationStrategy.BASIC)
        
        # Output should have x (from spec1 which has actual sharding)
        assert out_spec.dim_specs[0].axes == ["x"]
    
    def test_priority_stronger_wins(self):
        """Lower priority number (p0) beats higher (p1)."""
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # spec1 has priority 0 (stronger)
        spec1 = ShardingSpec(mesh, [DimSpec(["x"], is_open=False, priority=0)])
        # spec2 has priority 1 (weaker)
        spec2 = ShardingSpec(mesh, [DimSpec([], is_open=True, priority=1)])
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = elementwise_template(rank=1)
        rule = template.instantiate([(4,), (4,)], [(4,)])
        
        propagate_sharding(rule, [spec1, spec2], [out_spec])
        
        # Output should get priority 0 sharding (x)
        assert out_spec.dim_specs[0].axes == ["x"]
        assert out_spec.dim_specs[0].priority == 0


# ============================================================================
# Part 4: Reduce Operation Tests (Shape-Changing)
# ============================================================================

class TestReduceTemplate:
    """Test reduce_template for operations that remove dimensions."""
    
    def test_reduce_sum_keepdims_false(self):
        """Reduce sum with keepdims=False: (d0, d1, d2) -> (d0, d2) [axis=1]."""
        template = reduce_template(rank=3, reduce_dims=[1], keepdims=False)
        rule = template.instantiate(
            input_shapes=[(4, 8, 2)],
            output_shapes=[(4, 2)]
        )
        
        # Input: all 3 dims map to factors
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"], 2: ["d2"]}
        
        # Output: d1 is GONE (reduced away)
        assert rule.output_mappings[0] == {0: ["d0"], 1: ["d2"]}
        
        # Factor sizes
        assert rule.factor_sizes == {"d0": 4, "d1": 8, "d2": 2}
        
        # Verify einsum notation
        assert rule.to_einsum_notation() == "(d0, d1, d2) -> (d0, d2)"
    
    def test_reduce_sum_keepdims_true(self):
        """Reduce sum with keepdims=True: (d0, d1, d2) -> (d0, 1, d2) [axis=1]."""
        template = reduce_template(rank=3, reduce_dims=[1], keepdims=True)
        rule = template.instantiate(
            input_shapes=[(4, 8, 2)],
            output_shapes=[(4, 1, 2)]
        )
        
        # Input: all dims present
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"], 2: ["d2"]}
        
        # Output: d1 position has EMPTY list (size-1 dimension)
        assert rule.output_mappings[0] == {0: ["d0"], 1: [], 2: ["d2"]}
    
    def test_reduce_first_dim(self):
        """Reduce first dimension: (d0, d1) -> (d1)."""
        template = reduce_template(rank=2, reduce_dims=[0], keepdims=False)
        rule = template.instantiate(
            input_shapes=[(8, 4)],
            output_shapes=[(4,)]
        )
        
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"]}
        assert rule.output_mappings[0] == {0: ["d1"]}  # Only d1 remains


class TestReducePropagation:
    """Test sharding propagation through reduce operations."""
    
    def test_reduce_sharded_dim_requires_allreduce(self):
        """If we reduce over a SHARDED dimension, need AllReduce.
        
        Input: (d0=x, d1) sharded on first dim
        Reduce axis=0 -> (d1)
        
        Since d0 is sharded, each shard computes partial sum.
        AllReduce needed to get full sum.
        """
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on first dim (which will be reduced)
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # d0 sharded
            DimSpec([], is_open=True)
        ])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])  # Only d1
        
        template = reduce_template(rank=2, reduce_dims=[0], keepdims=False)
        rule = template.instantiate(
            input_shapes=[(8, 4)],
            output_shapes=[(4,)]
        )
        
        propagate_sharding(rule, [in_spec], [out_spec])
        
        # Output should be replicated (d0 factor disappeared)
        assert out_spec.dim_specs[0].axes == []
        
        # Verify d0 is a contracting factor (appears in input, not output)
        contracting = rule.get_contracting_factors()
        assert "d0" in contracting
    
    def test_reduce_unsharded_dim_preserves_sharding(self):
        """Reduce over UNSHARDED dimension preserves other sharding.
        
        Input: (d0, d1=x) sharded on second dim
        Reduce axis=0 -> (d1=x)
        
        Since d1 is sharded and preserved, output should be sharded.
        """
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input sharded on second dim
        in_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),      # d0 not sharded
            DimSpec(["x"], is_open=False)   # d1 sharded
        ])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])
        
        template = reduce_template(rank=2, reduce_dims=[0], keepdims=False)
        rule = template.instantiate(
            input_shapes=[(8, 4)],
            output_shapes=[(4,)]
        )
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # Output should inherit d1 sharding
        assert changed
        assert out_spec.dim_specs[0].axes == ["x"]
    
    def test_reduce_middle_dim(self):
        """Reduce middle dimension: (d0=x, d1, d2=y) reduce axis=1 -> (d0=x, d2=y).
        
        Both d0 and d2 shardings should propagate to output.
        """
        mesh = DeviceMesh("test", (2, 2), ("x", "y"))
        
        # Input: first and last dims sharded
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # d0
            DimSpec([], is_open=True),      # d1 (will be reduced)
            DimSpec(["y"], is_open=False)   # d2
        ])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # d0
            DimSpec([], is_open=True)   # d2
        ])
        
        template = reduce_template(rank=3, reduce_dims=[1], keepdims=False)
        rule = template.instantiate(
            input_shapes=[(4, 8, 2)],
            output_shapes=[(4, 2)]
        )
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # Both d0 and d2 shardings should propagate
        assert changed
        assert out_spec.dim_specs[0].axes == ["x"]
        assert out_spec.dim_specs[1].axes == ["y"]


# ============================================================================
# Part 5: Transpose Operation Tests (Factor Permutation)
# ============================================================================

class TestTransposeTemplate:
    """Test transpose_template for dimension permutation."""
    
    def test_transpose_2d(self):
        """Transpose 2D: (d0, d1) -> (d1, d0)."""
        from nabla.sharding.propagation import transpose_template
        
        template = transpose_template(rank=2, perm=[1, 0])
        rule = template.instantiate(
            input_shapes=[(4, 8)],
            output_shapes=[(8, 4)]
        )
        
        # Input: d0, d1
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"]}
        
        # Output: d1, d0 (permuted)
        assert rule.output_mappings[0] == {0: ["d1"], 1: ["d0"]}
        
        # Einsum notation
        assert rule.to_einsum_notation() == "(d0, d1) -> (d1, d0)"
    
    def test_transpose_3d_cycle(self):
        """Transpose 3D with cyclic permutation: (d0, d1, d2) -> (d1, d2, d0)."""
        from nabla.sharding.propagation import transpose_template
        
        template = transpose_template(rank=3, perm=[1, 2, 0])
        rule = template.instantiate(
            input_shapes=[(2, 4, 8)],
            output_shapes=[(4, 8, 2)]
        )
        
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"], 2: ["d2"]}
        assert rule.output_mappings[0] == {0: ["d1"], 1: ["d2"], 2: ["d0"]}


class TestTransposePropagation:
    """Test sharding propagation through transpose operations."""
    
    def test_transpose_2d_sharding_follows(self):
        """Transpose 2D: if input dim0 is sharded on x, output dim1 should be sharded on x.
        
        Input: (d0=x, d1) -> Transpose -> Output: (d1, d0=x)
        """
        from nabla.sharding.propagation import transpose_template
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input: first dim sharded
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # d0
            DimSpec([], is_open=True)        # d1
        ])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # d1
            DimSpec([], is_open=True)   # d0
        ])
        
        template = transpose_template(rank=2, perm=[1, 0])
        rule = template.instantiate([(4, 8)], [(8, 4)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # Output dim0 corresponds to factor d1 (not sharded)
        # Output dim1 corresponds to factor d0 (sharded on x)
        assert changed
        assert out_spec.dim_specs[0].axes == []     # d1 not sharded
        assert out_spec.dim_specs[1].axes == ["x"]  # d0 sharded
    
    def test_transpose_3d_all_sharded(self):
        """Transpose 3D with all dimensions sharded: verify each axis follows its factor.
        
        Input: (d0=x, d1=y, d2=z)
        Transpose perm=[2, 0, 1] -> (d2, d0, d1)
        Output should be: (z, x, y)
        """
        mesh = DeviceMesh("test", (2, 2, 2), ("x", "y", "z"))
        
        # Input: each dim sharded on different axis
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),
            DimSpec(["y"], is_open=False),
            DimSpec(["z"], is_open=False)
        ])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),
            DimSpec([], is_open=True),
            DimSpec([], is_open=True)
        ])
        
        from nabla.sharding.propagation import transpose_template
        template = transpose_template(rank=3, perm=[2, 0, 1])
        rule = template.instantiate([(4, 8, 2)], [(2, 4, 8)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # Output perm=[2, 0, 1] means: out[0]=in[2], out[1]=in[0], out[2]=in[1]
        # So shardings should follow: z, x, y
        assert changed
        assert out_spec.dim_specs[0].axes == ["z"]
        assert out_spec.dim_specs[1].axes == ["x"]
        assert out_spec.dim_specs[2].axes == ["y"]


# ============================================================================
# Part 6: Broadcast Operation Tests (Dimension Expansion)
# ============================================================================

class TestBroadcastTemplate:
    """Test broadcast_with_shapes_template for dimension expansion."""
    
    def test_broadcast_rank_expansion(self):
        """Broadcast (4,) -> (3, 4): rank 1 to rank 2."""
        from nabla.sharding.propagation import broadcast_with_shapes_template
        
        template = broadcast_with_shapes_template(in_shape=(4,), out_shape=(3, 4))
        rule = template.instantiate(
            input_shapes=[(4,)],
            output_shapes=[(3, 4)]
        )
        
        # Input: d0
        assert rule.input_mappings[0] == {0: ["d0"]}
        
        # Output: new0, d0 (new dimension added at front)
        assert rule.output_mappings[0] == {0: ["new0"], 1: ["d0"]}
        
        # New dimension gets its own factor
        assert "new0" in rule.factor_sizes
        assert "d0" in rule.factor_sizes
    
    def test_broadcast_size_1_expansion(self):
        """Broadcast (1, 4) -> (3, 4): size-1 dimension expanded."""
        from nabla.sharding.propagation import broadcast_with_shapes_template
        
        template = broadcast_with_shapes_template(in_shape=(1, 4), out_shape=(3, 4))
        rule = template.instantiate(
            input_shapes=[(1, 4)],
            output_shapes=[(3, 4)]
        )
        
        # Input: in0 (size-1), d1
        assert rule.input_mappings[0] == {0: ["in0"], 1: ["d1"]}
        
        # Output: expand0 (new factor for expanded dim), d1
        assert rule.output_mappings[0] == {0: ["expand0"], 1: ["d1"]}
        
        # Expanded dimension gets NEW factor (not shared with input)
        assert "expand0" in rule.factor_sizes
        assert rule.factor_sizes["expand0"] == 3
    
    def test_broadcast_multiple_expansions(self):
        """Broadcast (1, 4, 1) -> (3, 4, 5): multiple size-1 expansions."""
        from nabla.sharding.propagation import broadcast_with_shapes_template
        
        template = broadcast_with_shapes_template(in_shape=(1, 4, 1), out_shape=(3, 4, 5))
        rule = template.instantiate(
            input_shapes=[(1, 4, 1)],
            output_shapes=[(3, 4, 5)]
        )
        
        # Input: in0, d1, in2
        assert rule.input_mappings[0] == {0: ["in0"], 1: ["d1"], 2: ["in2"]}
        
        # Output: expand0, d1, expand2
        assert rule.output_mappings[0] == {0: ["expand0"], 1: ["d1"], 2: ["expand2"]}


class TestBroadcastPropagation:
    """Test sharding propagation through broadcast operations."""
    
    def test_broadcast_preserves_matching_dim_sharding(self):
        """Broadcast (4,) -> (3, 4): sharding on dim=4 should propagate.
        
        Input: (d0=x)
        Output: (new0, d0=x) -> second dim should be sharded on x
        """
        from nabla.sharding.propagation import broadcast_with_shapes_template
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input: sharded on only dimension
        in_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False)])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # new0
            DimSpec([], is_open=True)   # d0
        ])
        
        template = broadcast_with_shapes_template((4,), (3, 4))
        rule = template.instantiate([(4,)], [(3, 4)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # First dim (new0) should be replicated
        # Second dim (d0) should inherit sharding
        assert changed
        assert out_spec.dim_specs[0].axes == []     # new dimension, not sharded
        assert out_spec.dim_specs[1].axes == ["x"]  # d0 gets x sharding
    
    def test_broadcast_size_1_creates_new_factor(self):
        """Broadcast (1, 4) -> (3, 4): expanded dim should NOT inherit sharding.
        
        Input: (in0=x, d1) where in0 has size 1
        Output: (expand0, d1)
        
        Since in0 and expand0 are DIFFERENT factors, x sharding should NOT propagate.
        """
        from nabla.sharding.propagation import broadcast_with_shapes_template
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input: first dim (size-1) is sharded
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),  # in0 (size-1)
            DimSpec([], is_open=True)        # d1
        ])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # expand0
            DimSpec([], is_open=True)   # d1
        ])
        
        template = broadcast_with_shapes_template((1, 4), (3, 4))
        rule = template.instantiate([(1, 4)], [(3, 4)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # expand0 is a NEW factor, should NOT get x sharding
        assert out_spec.dim_specs[0].axes == []  # expand0 not sharded
        assert out_spec.dim_specs[1].axes == []  # d1 not sharded


# ============================================================================
# Part 7: Unsqueeze Operation Tests (Insert Size-1 Dimension)
# ============================================================================

class TestUnsqueezeTemplate:
    """Test unsqueeze_template for inserting size-1 dimensions."""
    
    def test_unsqueeze_at_front(self):
        """Unsqueeze (4,) -> (1, 4) at axis=0."""
        from nabla.sharding.propagation import unsqueeze_template
        
        template = unsqueeze_template(in_rank=1, axis=0)
        rule = template.instantiate(
            input_shapes=[(4,)],
            output_shapes=[(1, 4)]
        )
        
        # Input: d0
        assert rule.input_mappings[0] == {0: ["d0"]}
        
        # Output: new, d0
        assert rule.output_mappings[0] == {0: ["new"], 1: ["d0"]}
        
        assert rule.to_einsum_notation() == "(d0) -> (new, d0)"
    
    def test_unsqueeze_at_end(self):
        """Unsqueeze (4,) -> (4, 1) at axis=-1."""
        from nabla.sharding.propagation import unsqueeze_template
        
        template = unsqueeze_template(in_rank=1, axis=-1)
        rule = template.instantiate(
            input_shapes=[(4,)],
            output_shapes=[(4, 1)]
        )
        
        # Input: d0
        assert rule.input_mappings[0] == {0: ["d0"]}
        
        # Output: d0, new
        assert rule.output_mappings[0] == {0: ["d0"], 1: ["new"]}
    
    def test_unsqueeze_middle(self):
        """Unsqueeze (4, 8) -> (4, 1, 8) at axis=1."""
        from nabla.sharding.propagation import unsqueeze_template
        
        template = unsqueeze_template(in_rank=2, axis=1)
        rule = template.instantiate(
            input_shapes=[(4, 8)],
            output_shapes=[(4, 1, 8)]
        )
        
        # Input: d0, d1
        assert rule.input_mappings[0] == {0: ["d0"], 1: ["d1"]}
        
        # Output: d0, new, d1
        assert rule.output_mappings[0] == {0: ["d0"], 1: ["new"], 2: ["d1"]}


class TestUnsqueezePropagation:
    """Test sharding propagation through unsqueeze operations."""
    
    def test_unsqueeze_preserves_existing_sharding(self):
        """Unsqueeze (d0=x) -> (new, d0=x): existing sharding preserved.
        
        New dimension gets no sharding, original dimension keeps its sharding.
        """
        from nabla.sharding.propagation import unsqueeze_template
        
        mesh = DeviceMesh("test", (2,), ("x",))
        
        # Input: sharded
        in_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False)])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # new
            DimSpec([], is_open=True)   # d0
        ])
        
        template = unsqueeze_template(in_rank=1, axis=0)
        rule = template.instantiate([(4,)], [(1, 4)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # New dimension not sharded, d0 gets x
        assert changed
        assert out_spec.dim_specs[0].axes == []     # new dimension
        assert out_spec.dim_specs[1].axes == ["x"]  # d0
    
    def test_unsqueeze_middle_preserves_both_shardings(self):
        """Unsqueeze (d0=x, d1=y) -> (d0=x, new, d1=y): both shardings preserved."""
        from nabla.sharding.propagation import unsqueeze_template
        
        mesh = DeviceMesh("test", (2, 2), ("x", "y"))
        
        # Input: both dims sharded
        in_spec = ShardingSpec(mesh, [
            DimSpec(["x"], is_open=False),
            DimSpec(["y"], is_open=False)
        ])
        
        # Output initially open
        out_spec = ShardingSpec(mesh, [
            DimSpec([], is_open=True),  # d0
            DimSpec([], is_open=True),  # new
            DimSpec([], is_open=True)   # d1
        ])
        
        template = unsqueeze_template(in_rank=2, axis=1)
        rule = template.instantiate([(4, 8)], [(4, 1, 8)])
        
        changed = propagate_sharding(rule, [in_spec], [out_spec])
        
        # Both original shardings preserved
        assert changed
        assert out_spec.dim_specs[0].axes == ["x"]  # d0
        assert out_spec.dim_specs[1].axes == []     # new
        assert out_spec.dim_specs[2].axes == ["y"]  # d1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
