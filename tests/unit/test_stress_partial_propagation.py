# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Numerical correctness stress tests for partial-tensor SPMD propagation.

Each test:
  1. Runs the computation with sharded tensors on a mock multi-device mesh.
  2. Runs the exact same computation unsharded (JAX as reference).
  3. Asserts numerical equality with assert_allclose.

This validates that all_reduce barriers fire at algebraically correct positions
— not just in the trace string, but in the actual computed values.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import nabla as nb
from nabla.core.sharding.spec import DeviceMesh

from .common import (
    assert_allclose,
    cleanup_caches,
    make_jax_array,
    replicated,
    shard_on_axis,
    tensor_from_jax,
    to_jax,
)


class TestNumericalCorrectnessPartialPropagation:
    """Verify sharded computations produce bit-exact results vs single-device JAX."""

    def test_row_parallel_matmul_then_pow(self):
        """Row-parallel: x@w is partial on 'tp'. pow is non-linear so all_reduce
        must fire before (x@w)^e. Result must match jnp.power(x @ w, e).

        Without the all_reduce fix, each device would compute power(partial_sum, e)
        and sum those — giving a completely wrong answer.
        """
        cleanup_caches()
        M, K, N = 4, 8, 4
        mesh = DeviceMesh("tp_pow", (2,), ("tp",))

        x_jax = make_jax_array(M, K, seed=1)
        w_jax = make_jax_array(K, N, seed=2)
        e_jax = make_jax_array(1, seed=3)

        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)
        e_nb = tensor_from_jax(e_jax)

        # Row-parallel: shard the contracting (inner) dimension
        x_sharded = shard_on_axis(x_nb, mesh, axis=1, mesh_axis=0)  # [M, K/2] per device
        w_sharded = shard_on_axis(w_nb, mesh, axis=0, mesh_axis=0)  # [K/2, N] per device
        e_replicated = replicated(e_nb, mesh)

        def f(x, w, e):
            # x@w is partial on 'tp' (row-parallel over contracting dim)
            # pow is non-linear: compiler must all_reduce FIRST, then pow
            return nb.pow(x @ w, e)

        result = f(x_sharded, w_sharded, e_replicated)
        expected = jnp.power(x_jax @ w_jax, e_jax)

        nabla_vals = to_jax(result)
        print(f"\n[pow] nabla result (first row): {nabla_vals[0]}")
        print(f"[pow] jax   result (first row): {expected[0]}")
        print(f"[pow] max abs diff: {float(np.max(np.abs(nabla_vals - expected))):.2e}")
        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_row_parallel_matmul_then_exp(self):
        """Row-parallel: x@w is partial on 'tp'. exp is non-linear so all_reduce
        must fire before exp(x@w). Result must match jnp.exp(x @ w).
        """
        cleanup_caches()
        M, K, N = 4, 8, 4
        mesh = DeviceMesh("tp_exp", (2,), ("tp",))

        x_jax = make_jax_array(M, K, seed=4)
        w_jax = make_jax_array(K, N, seed=5)

        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)

        x_sharded = shard_on_axis(x_nb, mesh, axis=1, mesh_axis=0)
        w_sharded = shard_on_axis(w_nb, mesh, axis=0, mesh_axis=0)

        def f(x, w):
            return nb.exp(x @ w)

        result = f(x_sharded, w_sharded)
        expected = jnp.exp(x_jax @ w_jax)

        nabla_vals = to_jax(result)
        print(f"\n[exp] nabla result (first row): {nabla_vals[0]}")
        print(f"[exp] jax   result (first row): {expected[0]}")
        print(f"[exp] max abs diff: {float(np.max(np.abs(nabla_vals - expected))):.2e}")
        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_row_parallel_matmul_then_scalar_mul_deferred(self):
        """Row-parallel: x@w is partial on 'tp'. Multiply by a replicated scalar s
        is distributive: (A0+A1)*s = A0*s + A1*s, so reduction stays deferred.
        No premature all_reduce. Result must still match (x @ w) * s.
        """
        cleanup_caches()
        M, K, N = 4, 8, 4
        mesh = DeviceMesh("tp_scalarmul", (2,), ("tp",))

        x_jax = make_jax_array(M, K, seed=6)
        w_jax = make_jax_array(K, N, seed=7)
        s_jax = make_jax_array(1, seed=8)

        x_nb = tensor_from_jax(x_jax)
        w_nb = tensor_from_jax(w_jax)
        s_nb = tensor_from_jax(s_jax)

        x_sharded = shard_on_axis(x_nb, mesh, axis=1, mesh_axis=0)
        w_sharded = shard_on_axis(w_nb, mesh, axis=0, mesh_axis=0)
        s_replicated = replicated(s_nb, mesh)

        def f(x, w, s):
            # Partial * replicated scalar defers: still correct at realization
            return (x @ w) * s

        result = f(x_sharded, w_sharded, s_replicated)
        expected = (x_jax @ w_jax) * s_jax

        nabla_vals = to_jax(result)
        print(f"\n[scalar_mul] nabla result (first row): {nabla_vals[0]}")
        print(f"[scalar_mul] jax   result (first row): {expected[0]}")
        print(f"[scalar_mul] max abs diff: {float(np.max(np.abs(nabla_vals - expected))):.2e}")
        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_megatron_column_relu_row(self):
        """Megatron-LM style column-parallel → relu → row-parallel MLP block.

        - W1 sharded column-wise (output dim): x@W1 produces a sharded output (NOT partial)
        - relu over sharded intermediate: still sharded (relu is element-wise, safe)
        - W2 sharded row-wise (input dim): relu(x@W1) @ W2 is partial on 'tp' → all_reduce

        Result must match jnp.maximum(x @ W1, 0) @ W2 computed unsharded.
        """
        cleanup_caches()
        M, H, H2 = 4, 8, 4
        mesh = DeviceMesh("tp_megatron", (2,), ("tp",))

        x_jax = make_jax_array(M, H, seed=10)
        w1_jax = make_jax_array(H, H2, seed=11)
        w2_jax = make_jax_array(H2, H, seed=12)

        x_nb = tensor_from_jax(x_jax)
        w1_nb = tensor_from_jax(w1_jax)
        w2_nb = tensor_from_jax(w2_jax)

        # Column-parallel W1: shard output dim → result is sharded (not partial)
        w1_sharded = shard_on_axis(w1_nb, mesh, axis=1, mesh_axis=0)  # [H, H2/2]
        # Row-parallel W2: shard input dim → result is partial on 'tp'
        w2_sharded = shard_on_axis(w2_nb, mesh, axis=0, mesh_axis=0)  # [H2/2, H]
        x_replicated = replicated(x_nb, mesh)

        def f(x, w1, w2):
            h = nb.relu(x @ w1)  # sharded on tp, relu element-wise over shards
            return h @ w2        # partial on tp → all_reduce at realization

        result = f(x_replicated, w1_sharded, w2_sharded)
        expected = jnp.maximum(x_jax @ w1_jax, 0) @ w2_jax

        nabla_vals = to_jax(result)
        print(f"\n[megatron] nabla result (first row): {nabla_vals[0]}")
        print(f"[megatron] jax   result (first row): {expected[0]}")
        print(f"[megatron] max abs diff: {float(np.max(np.abs(nabla_vals - expected))):.2e}")
        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

# ---- old trace-only stubs removed ----
# def test_trace_nonlinear_barrier_pow  → replaced by test_row_parallel_matmul_then_pow
# def test_trace_unary_nonlinear_barrier_exp → replaced by test_row_parallel_matmul_then_exp
# def test_trace_2d_orthogonal_partial_matmul → covered by test_megatron_column_relu_row


class TestPartialThroughViewOps:
    """Verify that view ops (slice, gather, concat) correctly defer or force
    all_reduce on partial tensors.

    These tests validate the correctness of the four bugs fixed:
      1. CastOp: no longer defers (narrowing casts break distributivity)
      2. SliceTensorOp: now correctly defers (slice is distributive)
      3. GatherOp: now correctly defers (gather is distributive)
      4. ConcatenateOp: now uses all-inputs-must-be-partial check
    """

    def test_slice_defers_through_partial(self):
        """SliceTensorOp is distributive: slice(p0+p1) = slice(p0) + slice(p1).

        After the fix, the all_reduce is deferred past the slice. The result
        must still match JAX's (x @ w)[:, 0:N//2].
        """
        cleanup_caches()
        M, K, N = 4, 8, 4
        mesh = DeviceMesh("tp_slice", (2,), ("tp",))

        x_jax = make_jax_array(M, K, seed=20)
        w_jax = make_jax_array(K, N, seed=21)

        x_sharded = shard_on_axis(tensor_from_jax(x_jax), mesh, axis=1, mesh_axis=0)
        w_sharded = shard_on_axis(tensor_from_jax(w_jax), mesh, axis=0, mesh_axis=0)

        def f(x, w):
            # x@w is partial on 'tp'; slice over the output (n) dim
            # slice(p0+p1) = slice(p0) + slice(p1) ✓ — safe to defer
            return nb.slice_tensor(x @ w, start=(0, 0), size=(M, N // 2))

        result = f(x_sharded, w_sharded)
        expected = (x_jax @ w_jax)[:, : N // 2]

        actual = to_jax(result)
        print(f"\n[slice] nabla (first row): {actual[0]}")
        print(f"[slice] jax   (first row): {expected[0]}")
        print(f"[slice] max abs diff: {float(np.max(np.abs(actual - expected))):.2e}")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_gather_defers_through_partial(self):
        """GatherOp is distributive: gather(p0+p1, idx) = gather(p0,idx)+gather(p1,idx).

        After the fix, gather from a partial tensor defers the all_reduce.
        The result must match JAX's (x @ w)[indices, :].
        """
        cleanup_caches()
        M, K, N = 6, 8, 4
        mesh = DeviceMesh("tp_gather", (2,), ("tp",))

        x_jax = make_jax_array(M, K, seed=22)
        w_jax = make_jax_array(K, N, seed=23)

        # Indices: select rows 0, 2, 4 out of M=6
        idx_jax = jnp.array([0, 2, 4], dtype=jnp.int32)
        idx_nb = tensor_from_jax(idx_jax)

        x_sharded = shard_on_axis(tensor_from_jax(x_jax), mesh, axis=1, mesh_axis=0)
        w_sharded = shard_on_axis(tensor_from_jax(w_jax), mesh, axis=0, mesh_axis=0)
        idx_rep = replicated(idx_nb, mesh)

        def f(x, w, idx):
            # gather rows from partial result — safe to defer
            return nb.gather(x @ w, idx, axis=0)

        result = f(x_sharded, w_sharded, idx_rep)
        expected = (x_jax @ w_jax)[jnp.array([0, 2, 4]), :]

        actual = to_jax(result)
        print(f"\n[gather] nabla result shape: {actual.shape}")
        print(f"[gather] nabla (row 0): {actual[0]}")
        print(f"[gather] jax   (row 0): {expected[0]}")
        print(f"[gather] max abs diff: {float(np.max(np.abs(actual - expected))):.2e}")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_concat_both_partial_defers(self):
        """concat([A_partial, B_partial]) defers when BOTH inputs are partial on
        the same axis. concat([A0+A1, B0+B1]) = concat([A0,B0]) + concat([A1,B1]).

        This represents a realistic scenario: concatenating outputs from two
        row-parallel matmuls (e.g. two expert outputs before residual add).
        """
        cleanup_caches()
        M, K, N = 4, 8, 4
        mesh = DeviceMesh("tp_concat_both", (2,), ("tp",))

        x_jax = make_jax_array(M, K, seed=24)
        w1_jax = make_jax_array(K, N, seed=25)
        w2_jax = make_jax_array(K, N, seed=26)

        x_sharded = shard_on_axis(tensor_from_jax(x_jax), mesh, axis=1, mesh_axis=0)
        w1_sharded = shard_on_axis(tensor_from_jax(w1_jax), mesh, axis=0, mesh_axis=0)
        w2_sharded = shard_on_axis(tensor_from_jax(w2_jax), mesh, axis=0, mesh_axis=0)

        def f(x, w1, w2):
            # Both matmuls produce partial results on 'tp'
            # concat of two partials on the same axis is safe to defer
            r1 = x @ w1  # partial on tp
            r2 = x @ w2  # partial on tp
            return nb.concatenate([r1, r2], axis=1)

        result = f(x_sharded, w1_sharded, w2_sharded)
        expected = jnp.concatenate([x_jax @ w1_jax, x_jax @ w2_jax], axis=1)

        actual = to_jax(result)
        print(f"\n[concat-both-partial] nabla shape: {actual.shape}")
        print(f"[concat-both-partial] nabla (first row): {actual[0]}")
        print(f"[concat-both-partial] jax   (first row): {expected[0]}")
        print(f"[concat-both-partial] max abs diff: {float(np.max(np.abs(actual - expected))):.2e}")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_cast_does_not_defer_through_partial(self):
        """After the CastOp fix, cast forces all_reduce BEFORE the cast.

        Bug scenario: with allows_partial_passthrough=True (old broken code),
        the engine would cast each shard's partial sum to the target dtype, then
        sum those cast values. For narrowing dtypes that is WRONG.

        After the fix, the engine all_reduces first (producing the exact float32
        sum), then casts. The result matches JAX's jnp.array(x@w, dtype=float16).
        """
        cleanup_caches()
        M, K, N = 4, 8, 4
        mesh = DeviceMesh("tp_cast", (2,), ("tp",))

        # Use large seed values so float32→float16 rounding can show differences
        x_jax = make_jax_array(M, K, seed=30) * 10.0
        w_jax = make_jax_array(K, N, seed=31) * 10.0

        x_sharded = shard_on_axis(tensor_from_jax(x_jax), mesh, axis=1, mesh_axis=0)
        w_sharded = shard_on_axis(tensor_from_jax(w_jax), mesh, axis=0, mesh_axis=0)

        from max.dtype import DType

        def f(x, w):
            # all_reduce must fire BEFORE cast (non-distributive for narrowing)
            return nb.cast(x @ w, DType.float16)

        result = f(x_sharded, w_sharded)
        # JAX reference: full matmul in float32, THEN cast to float16
        expected = jnp.array(x_jax @ w_jax, dtype=jnp.float16)

        actual = to_jax(result)
        # Cast result back to float32 for comparison
        actual_f32 = np.array(actual, dtype=np.float32)
        expected_f32 = np.array(expected, dtype=np.float32)
        print(f"\n[cast] nabla (first row, f16→f32): {actual_f32[0]}")
        print(f"[cast] jax   (first row, f16→f32): {expected_f32[0]}")
        print(f"[cast] max abs diff: {float(np.max(np.abs(actual_f32 - expected_f32))):.2e}")
        # Both are float16 arrays — should match exactly (same rounding point)
        np.testing.assert_array_equal(actual_f32, expected_f32)


class TestNegativeOracle:
    """Prove the positive tests are actually sensitive to the bug.

    We manually simulate what the engine would compute WITHOUT a proper
    all_reduce before a non-linear op:  each device computes f(partial_sum)
    independently and the results are then summed.  That is WRONG for
    non-linear f.  We assert the wrong answer differs from JAX so we know
    our passing tests would have caught the bug.
    """

    def test_premature_exp_gives_wrong_answer(self):
        """exp(p0 + p1) != exp(p0) + exp(p1) for generic p0, p1.

        This proves test_row_parallel_matmul_then_exp would FAIL if the
        engine skipped the all_reduce before exp.
        """
        M, K, N = 4, 8, 4
        x_jax = make_jax_array(M, K, seed=4)
        w_jax = make_jax_array(K, N, seed=5)

        # Simulate device 0 and device 1 partial sums (each holds K/2 columns)
        p0 = x_jax[:, : K // 2] @ w_jax[: K // 2, :]  # device-0 partial
        p1 = x_jax[:, K // 2 :] @ w_jax[K // 2 :, :]  # device-1 partial

        correct = jnp.exp(p0 + p1)          # all_reduce first, then exp
        wrong   = jnp.exp(p0) + jnp.exp(p1) # bug: exp on partial, then sum

        print(f"\n[neg-oracle exp] correct (first row): {correct[0]}")
        print(f"[neg-oracle exp] wrong   (first row): {wrong[0]}")
        print(f"[neg-oracle exp] max abs diff correct vs wrong: "
              f"{float(np.max(np.abs(correct - wrong))):.4f}")

        # They must be meaningfully different — confirming sensitivity
        assert not np.allclose(correct, wrong, rtol=1e-3, atol=1e-3), (
            "Bug: exp(p0)+exp(p1) accidentally equals exp(p0+p1) — "
            "oracle data needs different seeds"
        )

    def test_premature_pow_gives_wrong_answer(self):
        """power(p0+p1, e) != power(p0,e) + power(p1,e) for e != 1.

        This proves test_row_parallel_matmul_then_pow would FAIL if the
        engine skipped the all_reduce before pow.
        """
        M, K, N = 4, 8, 4
        x_jax = make_jax_array(M, K, seed=1)
        w_jax = make_jax_array(K, N, seed=2)
        e_jax = make_jax_array(1, seed=3)
        e = float(e_jax[0])

        p0 = x_jax[:, : K // 2] @ w_jax[: K // 2, :]
        p1 = x_jax[:, K // 2 :] @ w_jax[K // 2 :, :]

        # Avoid NaN from negative bases with non-integer exponents by using abs
        p0_abs = jnp.abs(p0)
        p1_abs = jnp.abs(p1)
        correct = jnp.power(p0_abs + p1_abs, e)
        wrong   = jnp.power(p0_abs, e) + jnp.power(p1_abs, e)

        print(f"\n[neg-oracle pow] exponent e={e:.4f}")
        print(f"[neg-oracle pow] correct (first row): {correct[0]}")
        print(f"[neg-oracle pow] wrong   (first row): {wrong[0]}")
        print(f"[neg-oracle pow] max abs diff correct vs wrong: "
              f"{float(np.max(np.abs(correct - wrong))):.4f}")

        assert not np.allclose(correct, wrong, rtol=1e-3, atol=1e-3), (
            "Oracle data accidentally satisfies power(a+b,e)=power(a,e)+power(b,e)"
        )

    def test_scalar_mul_deferred_is_distributive(self):
        """(p0+p1)*s == p0*s + p1*s — scalar mul IS distributive.

        Unlike exp/pow, scalar mul commutes with reduction.  The engine
        CORRECTLY defers the all_reduce.  This oracle confirms the two
        orderings give the SAME answer, validating the deferral decision.
        """
        M, K, N = 4, 8, 4
        x_jax = make_jax_array(M, K, seed=6)
        w_jax = make_jax_array(K, N, seed=7)
        s_jax = make_jax_array(1, seed=8)
        s = float(s_jax[0])

        p0 = x_jax[:, : K // 2] @ w_jax[: K // 2, :]
        p1 = x_jax[:, K // 2 :] @ w_jax[K // 2 :, :]

        reduce_then_mul = (p0 + p1) * s   # correct ordering (all_reduce, then mul)
        mul_then_reduce = p0 * s + p1 * s  # deferred ordering (mul each shard, sum after)

        print(f"\n[neg-oracle scalar_mul] s={s:.4f}")
        print(f"[neg-oracle scalar_mul] reduce-then-mul (first row): {reduce_then_mul[0]}")
        print(f"[neg-oracle scalar_mul] mul-then-reduce (first row): {mul_then_reduce[0]}")
        print(f"[neg-oracle scalar_mul] max abs diff: "
              f"{float(np.max(np.abs(reduce_then_mul - mul_then_reduce))):.2e}")

        # Both orderings must agree — distributive property holds
        np.testing.assert_allclose(reduce_then_mul, mul_then_reduce, rtol=1e-5, atol=1e-5)

    def test_megatron_row_partial_gives_wrong_without_reduce(self):
        """In the Megatron block, the final row-parallel matmul produces partial
        sums.  Without all_reduce the result is just one shard's contribution,
        not the full answer.
        """
        M, H, H2 = 4, 8, 4
        x_jax = make_jax_array(M, H, seed=10)
        w1_jax = make_jax_array(H, H2, seed=11)
        w2_jax = make_jax_array(H2, H, seed=12)

        h_full = jnp.maximum(x_jax @ w1_jax, 0)  # [M, H2]

        # Each device sees half the hidden dim
        q0 = h_full[:, : H2 // 2] @ w2_jax[: H2 // 2, :]  # device-0 partial
        q1 = h_full[:, H2 // 2 :] @ w2_jax[H2 // 2 :, :]  # device-1 partial

        correct = q0 + q1           # all_reduce: full result
        wrong_d0_only = q0          # bug: returned device-0 shard without reduce

        print(f"\n[neg-oracle megatron] correct (first row): {correct[0]}")
        print(f"[neg-oracle megatron] wrong (d0 only, first row): {wrong_d0_only[0]}")
        print(f"[neg-oracle megatron] max abs diff: "
              f"{float(np.max(np.abs(correct - wrong_d0_only))):.4f}")

        assert not np.allclose(correct, wrong_d0_only, rtol=1e-3, atol=1e-3), (
            "Device-0 partial accidentally equals the full result — oracle needs different seeds"
        )

    def test_cast_narrowing_is_not_distributive(self):
        """Prove cast(narrowing) does NOT distribute over addition.

        cast(p0 + p1, float16) ≠ cast(p0, float16) + cast(p1, float16)
        when partial sums cross a float16 precision boundary.

        In float16, values in [2048, 4096) have step size 2.0 (only even
        integers representable). 2049 rounds DOWN to 2048. But 2049+1=2050
        is perfectly representable (even). So:

          cast(2049.0 + 1.0, f16) = cast(2050.0, f16) = 2050  ← correct
          cast(2049.0, f16) + cast(1.0, f16) = 2048 + 1 = 2049 ← wrong!

        This is the algebraic justification for removing CastOp's
        allows_partial_passthrough = True.
        """
        p0 = np.array([[2049.0, 2049.0]], dtype=np.float32)
        p1 = np.array([[1.0,    3.0   ]], dtype=np.float32)
        # sums: [2050.0, 2052.0] — both even, exactly representable in float16
        # individual casts: 2049 → 2048, 3 → 3, 1 → 1
        # wrong sums:       2048+1=2049,  2048+3=2051 — neither representable!

        correct = (p0 + p1).astype(np.float16)             # [2050, 2052]
        wrong   = p0.astype(np.float16) + p1.astype(np.float16)  # [2049, 2051]
        # Note: 2049 and 2051 can't be exactly represented in float16 (step=2 in
        # this range), so they round to 2048 and 2050 respectively when stored,
        # but the intermediate uint16 bit patterns differ from the correct result.

        print(f"\n[neg-oracle cast] correct cast(p0+p1, f16): {correct[0]}")
        print(f"[neg-oracle cast] wrong   cast(p0)+cast(p1): {wrong[0]}")
        print(f"[neg-oracle cast] max abs diff: {float(np.max(np.abs(correct.astype(np.float32) - wrong.astype(np.float32)))):.1f}")

        assert not np.array_equal(correct, wrong), (
            "Expected cast distributivity failure: cast(2049+1, f16) should ≠ "
            "cast(2049,f16)+cast(1,f16)"
        )

    def test_concat_partial_replicated_gives_wrong_without_reduce(self):
        """concat([partial_A, replicated_B]) is NOT safe to defer.

        Without all_reduce: each device has concat([partial_shard, B]),
        then all_reduce sums them = concat([A0+A1, 2*B]) → B is doubled!

        This proves that ConcatenateOp MUST NOT defer when only some inputs
        are partial — exactly what our fixed partial_passthrough_axes enforces.
        """
        M, N1, N2 = 4, 4, 4
        x_jax = make_jax_array(M, N1, seed=40)  # the partial operand
        b_jax = make_jax_array(M, N2, seed=41)  # the replicated operand

        # Simulate: x is partial (p0 + p1), b is replicated
        p0 = x_jax / 2.0
        p1 = x_jax / 2.0  # p0 + p1 = x_jax

        # Bug: defer concat across partial, then all_reduce
        # device0 computes: concat([p0, b]) → [M, N1+N2]
        # device1 computes: concat([p1, b]) → [M, N1+N2]
        # wrong all_reduce sums them: concat([p0+p1, b+b]) = concat([x, 2b])
        wrong = jnp.concatenate([p0 + p1, b_jax + b_jax], axis=1)  # B doubled!

        # Correct: all_reduce p0+p1 first, then concat
        correct = jnp.concatenate([x_jax, b_jax], axis=1)

        print(f"\n[neg-oracle concat] correct B columns (first row): {correct[0, N1:]}")
        print(f"[neg-oracle concat] wrong   B columns (first row): {wrong[0, N1:]}")
        print(f"[neg-oracle concat] max abs diff in B columns: "
              f"{float(np.max(np.abs(correct[:, N1:] - wrong[:, N1:]))):.4f}")

        assert not np.allclose(correct, wrong, rtol=1e-3, atol=1e-3), (
            "Replicated tensor B accidentally equals 2*B — oracle needs different seeds"
        )
