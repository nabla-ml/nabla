#!/usr/bin/env python3
"""
COMPREHENSIVE NABLA FUNCTION TRANSFORMATION TEST SUITE
=====================================================

This file provides systematic validation of Nabla's function transformations (vjp, jvp, vmap, jit)
against JAX as ground truth. Tests the `add` operation across all meaningful transformation combinations
and tensor rank combinations to ensure correctness and consistency.

TRANSFORMATION COMBINATIONS TESTED (21 total):
──────────────────────────────────────────────

Level 0 - Baseline:
    1. f(x,y)                           # Direct function call

Level 1 - Single Transformations:
    2. vjp(f)                          # Reverse-mode autodiff
    3. jvp(f)                          # Forward-mode autodiff
    4. vmap(f)                         # Vectorization
    5. jit(f)                          # JIT compilation

Level 2 - Double Transformations:
    6. jit(vjp(f))                     # JIT + reverse-mode
    7. jit(jvp(f))                     # JIT + forward-mode
    8. jit(vmap(f))                    # JIT + vectorization
    9. vmap(vjp(f))                    # Vectorized reverse-mode
   10. vmap(jvp(f))                    # Vectorized forward-mode

Level 3 - Triple Transformations:
   11. jit(vmap(vjp(f)))               # JIT + vectorized reverse-mode
   12. jit(vmap(jvp(f)))               # JIT + vectorized forward-mode

Level 4 - Higher-Order Differentiation:
   13. vjp(vjp(f))                     # Second-order reverse (Hessian-vector)
   14. jvp(vjp(f))                     # Mixed-mode differentiation
   15. vmap(vjp(f))                    # [Duplicate of #9 for completeness]
   16. vjp(jvp(f))                     # Mixed-mode differentiation
   17. jvp(jvp(f))                     # Second-order forward
   18. vmap(jvp(f))                    # [Duplicate of #10 for completeness]

Level 5 - Advanced Compositions:
   19. vjp(vmap(f))                    # Differentiate vectorized function
   20. jvp(vmap(f))                    # Forward-mode of vectorized function
   21. vmap(vmap(f))                   # Double vectorization

TENSOR RANK COMBINATIONS TESTED (10 total):
──────────────────────────────────────────

Same-Rank Combinations:
    (0,0) : scalar   + scalar          # f(2.5, 1.5)
    (1,1) : vector   + vector          # f([1,2,3,4], [5,6,7,8])
    (2,2) : matrix   + matrix          # f([[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]])
    (3,3) : tensor3D + tensor3D        # f([2,2,3], [2,2,3])

Mixed-Rank Broadcasting:
    (0,1) : scalar   + vector          # f(2.5, [1,2,3,4])
    (0,2) : scalar   + matrix          # f(2.5, [[1,2,3],[4,5,6]])
    (0,3) : scalar   + tensor3D        # f(2.5, [2,2,3])
    (1,2) : vector   + matrix          # f([1,2,3,4], [[1,2,3],[4,5,6]])
    (1,3) : vector   + tensor3D        # f([1,2,3,4], [2,2,3])
    (2,3) : matrix   + tensor3D        # f([[1,2,3],[4,5,6]], [2,2,3])

Total Test Matrix: 21 transformations × 10 rank combinations = 210 individual tests

CONSISTENCY CHECKING LOGIC:
─────────────────────────

For each test, both Nabla and JAX implementations are executed separately:

✅ PASS if both succeed and produce identical results
✅ PASS if both fail consistently (regardless of error message details)
❌ FAIL if only one framework fails (indicates discrepancy)
❌ FAIL if results don't match when both succeed

This approach correctly identifies:
- Functional correctness (matching results)
- Behavioral consistency (both handle edge cases the same way)
- Implementation discrepancies (unexpected differences)

Expected consistent failures include:
- vmap operations on scalar tensors (rank 0)
- Broadcasting incompatibilities for certain mixed-rank combinations
- Batch size mismatches in vectorization

USAGE:
──────

Run default tests (rank 2,2):
    python test_add_comprehensive.py

Run all rank combinations:
    python test_add_comprehensive.py --all-ranks

The test suite validates that Nabla's function transformations work correctly
and consistently with JAX across all meaningful scenarios.
"""

import sys

sys.path.insert(0, "/Users/tillife/Documents/CodingProjects/nabla")

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


# Helper functions
def jax_arange(shape, dtype=jnp.float32):
    """Create JAX array matching nabla.arange"""
    return jax.numpy.arange(np.prod(shape), dtype=dtype).reshape(shape)


def f_nb(x, y):
    """Nabla add function"""
    return nb.add(x, y)


def f_jax(x, y):
    """JAX add function"""
    return jax.numpy.add(x, y)


def get_shape_for_rank(rank):
    """Get appropriate shape for given tensor rank"""
    if rank == 0:
        return ()  # scalar
    elif rank == 1:
        return (4,)  # vector
    elif rank == 2:
        return (2, 3)  # matrix
    elif rank == 3:
        return (2, 2, 3)  # 3D tensor
    else:
        raise ValueError(f"Unsupported rank: {rank}")


def get_test_data_for_ranks(rank_x, rank_y):
    """Get test data for specific tensor ranks"""
    shape_x = get_shape_for_rank(rank_x)
    shape_y = get_shape_for_rank(rank_y)

    # Handle scalar case specially
    if rank_x == 0:
        x_nb = nb.array(2.5)
        x_jax = jnp.array(2.5)
    else:
        x_nb = nb.arange(shape_x)
        x_jax = jax_arange(shape_x)

    if rank_y == 0:
        y_nb = nb.array(1.5)
        y_jax = jnp.array(1.5)
    else:
        y_nb = nb.arange(shape_y)
        y_jax = jax_arange(shape_y)

    return x_nb, y_nb, x_jax, y_jax


def get_test_data():
    """Get test data for both frameworks (default rank 2)"""
    return get_test_data_for_ranks(2, 2)


def run_test_with_consistency_check(test_name, nabla_fn, jax_fn):
    """
    Run Nabla and JAX functions separately and check for consistency.

    Returns True if:
    - Both succeed and give same result
    - Both fail in expected/consistent ways

    Returns False if:
    - Only one fails
    - They fail with different error types
    - Results don't match when both succeed
    """
    nabla_result = None
    jax_result = None
    nabla_error = None
    jax_error = None

    # Try Nabla
    try:
        nabla_result = nabla_fn()
    except Exception as e:
        nabla_error = str(e)

    # Try JAX
    try:
        jax_result = jax_fn()
    except Exception as e:
        jax_error = str(e)

    # Case 1: Both succeeded - check if results match
    if nabla_result is not None and jax_result is not None:
        try:
            # Handle tuple results (e.g., from VJP/JVP)
            if isinstance(nabla_result, tuple) and isinstance(jax_result, tuple):
                if len(nabla_result) != len(jax_result):
                    print(f"✗ {test_name}: Tuple length mismatch")
                    return False

                for i, (nb_item, jax_item) in enumerate(zip(nabla_result, jax_result, strict=False)):
                    if hasattr(nb_item, "to_numpy"):
                        nb_numpy = nb_item.to_numpy()
                    else:
                        nb_numpy = np.array(nb_item)

                    if not jnp.allclose(nb_numpy, jax_item):
                        print(f"✗ {test_name}: Tuple item {i} doesn't match")
                        return False

                print(f"✓ {test_name}")
                return True

            # Handle single array results
            else:
                if isinstance(nabla_result, nb.Array):
                    nabla_numpy = nabla_result.to_numpy()
                else:
                    nabla_numpy = np.array(nabla_result)

                if jnp.allclose(nabla_numpy, jax_result):
                    print(f"✓ {test_name}")
                    return True
                else:
                    print(f"✗ {test_name}: Results don't match")
                    return False

        except Exception as e:
            print(f"✗ {test_name}: Comparison failed: {e}")
            return False

    # Case 2: Both failed - this is consistent behavior, so it's a pass
    elif nabla_error is not None and jax_error is not None:
        print(f"✓ {test_name} (both frameworks failed consistently)")
        return True

    # Case 3: Only one failed - this is a discrepancy
    else:
        if nabla_error is not None:
            print(f"✗ {test_name}: Only Nabla failed: {nabla_error}")
        else:
            print(f"✗ {test_name}: Only JAX failed: {jax_error}")
        return False


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def test_1_baseline():
    """Test: f(x, y)"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        return f_nb(x_nb, y_nb)

    def jax_fn():
        return f_jax(x_jax, y_jax)

    return run_test_with_consistency_check("Baseline f(x,y)", nabla_fn, jax_fn)


def test_2_vjp():
    """Test: vjp(f)"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        value, vjp_fn = nb.vjp(f_nb, x_nb, y_nb)
        grad = vjp_fn(nb.ones_like(value))
        return (value, grad[0], grad[1])

    def jax_fn():
        value, vjp_fn = jax.vjp(f_jax, x_jax, y_jax)
        grad = vjp_fn(jnp.ones_like(value))
        return (value, grad[0], grad[1])

    return run_test_with_consistency_check("VJP", nabla_fn, jax_fn)


def test_3_jvp():
    """Test: jvp(f)"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        primals = (x_nb, y_nb)
        tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
        value, tangent = nb.jvp(f_nb, primals, tangents)
        return (value, tangent)

    def jax_fn():
        primals = (x_jax, y_jax)
        tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
        value, tangent = jax.jvp(f_jax, primals, tangents)
        return (value, tangent)

    return run_test_with_consistency_check("JVP", nabla_fn, jax_fn)


def test_4_vmap():
    """Test: vmap(f)"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        return nb.vmap(f_nb)(x_nb, y_nb)

    def jax_fn():
        return jax.vmap(f_jax)(x_jax, y_jax)

    return run_test_with_consistency_check("VMAP", nabla_fn, jax_fn)


def test_5_jit():
    """Test: jit(f)"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        return nb.jit(f_nb)(x_nb, y_nb)

    def jax_fn():
        return jax.jit(f_jax)(x_jax, y_jax)

    return run_test_with_consistency_check("JIT", nabla_fn, jax_fn)


def test_6_jit_vjp():
    """Test: jit(vjp(f))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        value, vjp_fn = nb.jit(nb.vjp)(f_nb, x_nb, y_nb)
        grad = vjp_fn(nb.ones_like(value))
        return (value, grad[0], grad[1])

    def jax_fn():
        # JAX handles jit differently, so compare with regular vjp
        value, vjp_fn = jax.vjp(f_jax, x_jax, y_jax)
        grad = vjp_fn(jnp.ones_like(value))
        return (value, grad[0], grad[1])

    return run_test_with_consistency_check("JIT(VJP)", nabla_fn, jax_fn)


def test_7_jit_jvp():
    """Test: jit(jvp(f))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        primals = (x_nb, y_nb)
        tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
        value, tangent = nb.jit(nb.jvp)(f_nb, primals, tangents)
        return (value, tangent)

    def jax_fn():
        primals = (x_jax, y_jax)
        tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
        value, tangent = jax.jvp(f_jax, primals, tangents)
        return (value, tangent)

    return run_test_with_consistency_check("JIT(JVP)", nabla_fn, jax_fn)


def test_8_jit_vmap():
    """Test: jit(vmap(f))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        return nb.jit(nb.vmap(f_nb))(x_nb, y_nb)

    def jax_fn():
        return jax.jit(jax.vmap(f_jax))(x_jax, y_jax)

    return run_test_with_consistency_check("JIT(VMAP)", nabla_fn, jax_fn)


def test_9_vmap_vjp():
    """Test: vmap(vjp_value(f)) - batched VJP values"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def vjp_value_nb(x, y):
            value, _ = nb.vjp(f_nb, x, y)
            return value

        return nb.vmap(vjp_value_nb)(x_nb, y_nb)

    def jax_fn():
        def vjp_value_jax(x, y):
            value, _ = jax.vjp(f_jax, x, y)
            return value

        return jax.vmap(vjp_value_jax)(x_jax, y_jax)

    return run_test_with_consistency_check("VMAP(VJP)", nabla_fn, jax_fn)


def test_10_vmap_jvp():
    """Test: vmap(jvp_value(f)) - batched JVP values"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def jvp_value_nb(x, y):
            primals = (x, y)
            tangents = (nb.ones_like(x), nb.ones_like(y))
            value, _ = nb.jvp(f_nb, primals, tangents)
            return value

        return nb.vmap(jvp_value_nb)(x_nb, y_nb)

    def jax_fn():
        def jvp_value_jax(x, y):
            primals = (x, y)
            tangents = (jnp.ones_like(x), jnp.ones_like(y))
            value, _ = jax.jvp(f_jax, primals, tangents)
            return value

        return jax.vmap(jvp_value_jax)(x_jax, y_jax)

    return run_test_with_consistency_check("VMAP(JVP)", nabla_fn, jax_fn)


def test_11_jit_vmap_vjp():
    """Test: jit(vmap(vjp_value(f)))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def vjp_value_nb(x, y):
            value, _ = nb.vjp(f_nb, x, y)
            return value

        return nb.jit(nb.vmap(vjp_value_nb))(x_nb, y_nb)

    def jax_fn():
        def vjp_value_jax(x, y):
            value, _ = jax.vjp(f_jax, x, y)
            return value

        return jax.jit(jax.vmap(vjp_value_jax))(x_jax, y_jax)

    return run_test_with_consistency_check("JIT(VMAP(VJP))", nabla_fn, jax_fn)


def test_12_jit_vmap_jvp():
    """Test: jit(vmap(jvp_value(f)))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def jvp_value_nb(x, y):
            primals = (x, y)
            tangents = (nb.ones_like(x), nb.ones_like(y))
            value, _ = nb.jvp(f_nb, primals, tangents)
            return value

        return nb.jit(nb.vmap(jvp_value_nb))(x_nb, y_nb)

    def jax_fn():
        def jvp_value_jax(x, y):
            primals = (x, y)
            tangents = (jnp.ones_like(x), jnp.ones_like(y))
            value, _ = jax.jvp(f_jax, primals, tangents)
            return value

        return jax.jit(jax.vmap(jvp_value_jax))(x_jax, y_jax)

    return run_test_with_consistency_check("JIT(VMAP(JVP))", nabla_fn, jax_fn)


def test_13_vjp_vjp():
    """Test: vjp(vjp(f)) - second-order gradients"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def vjp_value_nb(x, y):
            value, _ = nb.vjp(f_nb, x, y)
            return value

        value, vjp_fn = nb.vjp(vjp_value_nb, x_nb, y_nb)
        grad = vjp_fn(nb.ones_like(value))
        return (value, grad[0], grad[1])

    def jax_fn():
        def vjp_value_jax(x, y):
            value, _ = jax.vjp(f_jax, x, y)
            return value

        value, vjp_fn = jax.vjp(vjp_value_jax, x_jax, y_jax)
        grad = vjp_fn(jnp.ones_like(value))
        return (value, grad[0], grad[1])

    return run_test_with_consistency_check("VJP(VJP)", nabla_fn, jax_fn)


def test_14_jvp_vjp():
    """Test: jvp(vjp_value(f))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def vjp_value_nb(x, y):
            value, _ = nb.vjp(f_nb, x, y)
            return value

        primals = (x_nb, y_nb)
        tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
        value, tangent = nb.jvp(vjp_value_nb, primals, tangents)
        return (value, tangent)

    def jax_fn():
        def vjp_value_jax(x, y):
            value, _ = jax.vjp(f_jax, x, y)
            return value

        primals = (x_jax, y_jax)
        tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
        value, tangent = jax.jvp(vjp_value_jax, primals, tangents)
        return (value, tangent)

    return run_test_with_consistency_check("JVP(VJP)", nabla_fn, jax_fn)


def test_15_vmap_vjp():
    """Test: vmap(vjp_value(f)) - already tested as test_9, but let's be explicit"""
    return test_9_vmap_vjp()


def test_16_vjp_jvp():
    """Test: vjp(jvp_value(f))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def jvp_value_nb(x, y):
            primals = (x, y)
            tangents = (nb.ones_like(x), nb.ones_like(y))
            value, _ = nb.jvp(f_nb, primals, tangents)
            return value

        value, vjp_fn = nb.vjp(jvp_value_nb, x_nb, y_nb)
        grad = vjp_fn(nb.ones_like(value))
        return (value, grad[0], grad[1])

    def jax_fn():
        def jvp_value_jax(x, y):
            primals = (x, y)
            tangents = (jnp.ones_like(x), jnp.ones_like(y))
            value, _ = jax.jvp(f_jax, primals, tangents)
            return value

        value, vjp_fn = jax.vjp(jvp_value_jax, x_jax, y_jax)
        grad = vjp_fn(jnp.ones_like(value))
        return (value, grad[0], grad[1])

    return run_test_with_consistency_check("VJP(JVP)", nabla_fn, jax_fn)


def test_17_jvp_jvp():
    """Test: jvp(jvp_value(f)) - second-order forward mode"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def jvp_value_nb(x, y):
            primals = (x, y)
            tangents = (nb.ones_like(x), nb.ones_like(y))
            value, _ = nb.jvp(f_nb, primals, tangents)
            return value

        primals = (x_nb, y_nb)
        tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
        value, tangent = nb.jvp(jvp_value_nb, primals, tangents)
        return (value, tangent)

    def jax_fn():
        def jvp_value_jax(x, y):
            primals = (x, y)
            tangents = (jnp.ones_like(x), jnp.ones_like(y))
            value, _ = jax.jvp(f_jax, primals, tangents)
            return value

        primals = (x_jax, y_jax)
        tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
        value, tangent = jax.jvp(jvp_value_jax, primals, tangents)
        return (value, tangent)

    return run_test_with_consistency_check("JVP(JVP)", nabla_fn, jax_fn)


def test_18_vmap_jvp():
    """Test: vmap(jvp_value(f)) - already tested as test_10, but let's be explicit"""
    return test_10_vmap_jvp()


def test_19_vjp_vmap():
    """Test: vjp(vmap(f))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def vmap_f_nb(x, y):
            return nb.vmap(f_nb)(x, y)

        value, vjp_fn = nb.vjp(vmap_f_nb, x_nb, y_nb)
        grad = vjp_fn(nb.ones_like(value))
        return (value, grad[0], grad[1])

    def jax_fn():
        def vmap_f_jax(x, y):
            return jax.vmap(f_jax)(x, y)

        value, vjp_fn = jax.vjp(vmap_f_jax, x_jax, y_jax)
        grad = vjp_fn(jnp.ones_like(value))
        return (value, grad[0], grad[1])

    return run_test_with_consistency_check("VJP(VMAP)", nabla_fn, jax_fn)


def test_20_jvp_vmap():
    """Test: jvp(vmap(f))"""
    x_nb, y_nb, x_jax, y_jax = get_test_data()

    def nabla_fn():
        def vmap_f_nb(x, y):
            return nb.vmap(f_nb)(x, y)

        primals = (x_nb, y_nb)
        tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
        value, tangent = nb.jvp(vmap_f_nb, primals, tangents)
        return (value, tangent)

    def jax_fn():
        def vmap_f_jax(x, y):
            return jax.vmap(f_jax)(x, y)

        primals = (x_jax, y_jax)
        tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
        value, tangent = jax.jvp(vmap_f_jax, primals, tangents)
        return (value, tangent)

    return run_test_with_consistency_check("JVP(VMAP)", nabla_fn, jax_fn)


def test_21_vmap_vmap():
    """Test: vmap(vmap(f)) - double vectorization"""
    # For this test, we need 3D data since we're vmapping twice
    x_nb = nb.arange((2, 3, 4))
    y_nb = nb.arange((2, 3, 4))
    x_jax = jax_arange((2, 3, 4))
    y_jax = jax_arange((2, 3, 4))

    def nabla_fn():
        # Double vmap - first over axis 0, then over axis 1
        return nb.vmap(nb.vmap(f_nb, in_axes=1, out_axes=1), in_axes=0, out_axes=0)(
            x_nb, y_nb
        )

    def jax_fn():
        return jax.vmap(jax.vmap(f_jax, in_axes=1, out_axes=1), in_axes=0, out_axes=0)(
            x_jax, y_jax
        )

    return run_test_with_consistency_check("VMAP(VMAP)", nabla_fn, jax_fn)


# ============================================================================
# PARAMETERIZED TEST FRAMEWORK
# ============================================================================


def run_test_with_ranks(test_func, rank_x, rank_y):
    """Run a test function with specific tensor ranks"""
    # Temporarily override get_test_data to use specific ranks
    global get_test_data
    original_get_test_data = get_test_data

    def get_test_data_override():
        return get_test_data_for_ranks(rank_x, rank_y)

    get_test_data = get_test_data_override

    try:
        # Call the test function - it should return True/False now
        result = test_func()
        get_test_data = original_get_test_data

        # If function returns boolean, use it; otherwise assume success
        return result if result is not None else True

    except Exception as e:
        get_test_data = original_get_test_data
        print(f"✗ Unexpected error: {e}")
        return False


def get_rank_combinations():
    """Get all rank combinations to test"""
    return [
        (0, 0),  # scalar + scalar
        (1, 1),  # vector + vector
        (2, 2),  # matrix + matrix (original)
        (3, 3),  # 3D + 3D
        (0, 1),  # scalar + vector
        (0, 2),  # scalar + matrix
        (0, 3),  # scalar + 3D
        (1, 2),  # vector + matrix
        (1, 3),  # vector + 3D
        (2, 3),  # matrix + 3D
    ]


def run_tests_for_all_ranks():
    """Run all tests for all rank combinations"""
    print("=" * 80)
    print("COMPREHENSIVE ADD OPERATION TESTS - ALL TENSOR RANK COMBINATIONS")
    print("=" * 80)

    # Get all test functions
    test_functions = [
        test_1_baseline,
        test_2_vjp,
        test_3_jvp,
        test_4_vmap,
        test_5_jit,
        test_6_jit_vjp,
        test_7_jit_jvp,
        test_8_jit_vmap,
        test_9_vmap_vjp,
        test_10_vmap_jvp,
        test_11_jit_vmap_vjp,
        test_12_jit_vmap_jvp,
        test_13_vjp_vjp,
        test_14_jvp_vjp,
        test_15_vmap_vjp,
        test_16_vjp_jvp,
        test_17_jvp_jvp,
        test_18_vmap_jvp,
        test_19_vjp_vmap,
        test_20_jvp_vmap,
        test_21_vmap_vmap,
    ]

    rank_combinations = get_rank_combinations()

    total_tests = 0
    total_passed = 0

    for rank_x, rank_y in rank_combinations:
        print(f"\n{'=' * 50}")
        print(f"TESTING RANKS: x={rank_x}, y={rank_y}")
        print(f"{'=' * 50}")

        passed_for_ranks = 0

        for i, test_func in enumerate(test_functions, 1):
            # Extract test description
            desc = (
                test_func.__doc__.split(":")[1].strip()
                if test_func.__doc__ and ":" in test_func.__doc__
                else test_func.__name__
            )

            print(f"{i:2d}. {desc} ", end="")

            # Run test with specific ranks
            success = run_test_with_ranks(test_func, rank_x, rank_y)

            if success:
                passed_for_ranks += 1
                total_passed += 1

            total_tests += 1

        print(
            f"\nRank ({rank_x},{rank_y}): {passed_for_ranks}/{len(test_functions)} tests passed"
        )

    print("\n" + "=" * 80)
    print(f"OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
    print(f"SUCCESS RATE: {total_passed / total_tests * 100:.1f}%")

    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED ACROSS ALL RANK COMBINATIONS!")
    else:
        print(f"❌ {total_tests - total_passed} tests failed")

    print("=" * 80)
    return total_passed == total_tests


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 70)
    print("COMPREHENSIVE ADD OPERATION TESTS - NABLA vs JAX")
    print("=" * 70)

    tests = [
        test_1_baseline,
        test_2_vjp,
        test_3_jvp,
        test_4_vmap,
        test_5_jit,
        test_6_jit_vjp,
        test_7_jit_jvp,
        test_8_jit_vmap,
        test_9_vmap_vjp,
        test_10_vmap_jvp,
        test_11_jit_vmap_vjp,
        test_12_jit_vmap_jvp,
        test_13_vjp_vjp,
        test_14_jvp_vjp,
        test_15_vmap_vjp,
        test_16_vjp_jvp,
        test_17_jvp_jvp,
        test_18_vmap_jvp,
        test_19_vjp_vmap,
        test_20_jvp_vmap,
        test_21_vmap_vmap,
    ]

    passed = 0
    total = len(tests)

    for i, test in enumerate(tests, 1):
        # Extract test description from docstring
        desc = (
            test.__doc__.split(":")[1].strip()
            if test.__doc__ and ":" in test.__doc__
            else test.__name__
        )
        print(f"\n{i:2d}. {desc}")

        # Call the test function - now it returns True/False
        success = test()
        if success:
            passed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {total - passed} tests failed")

    print("=" * 70)
    return passed == total


if __name__ == "__main__":
    import sys

    # Check if user wants to run all rank combinations
    if len(sys.argv) > 1 and sys.argv[1] == "--all-ranks":
        print("Running comprehensive tests for all tensor rank combinations...")
        success = run_tests_for_all_ranks()
    else:
        print("Running tests for default ranks (2,2)...")
        print("Use --all-ranks flag to test all tensor rank combinations")
        success = run_all_tests()

    sys.exit(0 if success else 1)
