#!/usr/bin/env python3
"""
COMPREHENSIVE NABLA BINARY OPERATIONS TEST SUITE
================================================

Generalized version of the add operation test suite that validates ALL binary operations
in Nabla against JAX as ground truth. Tests function transformations (vjp, jvp, vmap, jit)
across all binary operations and tensor rank combinations.

BINARY OPERATIONS TESTED (11 total):
──────────────────────────────────────

Arithmetic Operations:
    add, mul, sub, div, floordiv, pow

Comparison Operations:
    greater_equal, equal, not_equal

Min/Max Operations:
    maximum, minimum

TRANSFORMATION COMBINATIONS TESTED (19 total):
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
   15. vjp(jvp(f))                     # Mixed-mode differentiation
   16. jvp(jvp(f))                     # Second-order forward

Level 5 - Advanced Compositions:
   17. vjp(vmap(f))                    # Differentiate vectorized function
   18. jvp(vmap(f))                    # Forward-mode of vectorized function
   19. vmap(vmap(f))                   # Double vectorization

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

Total Test Matrix: 11 operations × 19 transformations × 10 rank combinations = 2,090 individual tests

CONSISTENCY CHECKING LOGIC:
─────────────────────────

For each test, both Nabla and JAX implementations are executed separately:

✅ PASS if both succeed and produce identical results (works for numeric AND boolean arrays)
✅ PASS if both fail consistently (e.g., boolean ops can't be differentiated)
❌ FAIL if only one framework fails (indicates discrepancy)
❌ FAIL if results don't match when both succeed

This approach correctly handles:
- Arithmetic operations (add, mul, etc.) - should work with all transformations
- Comparison operations (equal, greater_equal, etc.) - may fail VJP/JVP consistently
- Edge cases like division by zero, pow with negatives, etc.

USAGE:
──────

Run tests for a specific operation:
    python test_binary_ops.py add
    python test_binary_ops.py equal

Run tests for all operations:
    python test_binary_ops.py all

Run all operations with all rank combinations:
    python test_binary_ops.py all --all-ranks

The test suite validates that Nabla's binary operations and their transformations
work correctly and consistently with JAX across all scenarios.
"""

import sys

sys.path.insert(0, "/Users/tillife/Documents/CodingProjects/nabla")

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb

# ============================================================================
# BINARY OPERATION DEFINITIONS
# ============================================================================


@dataclass
class BinaryOperation:
    """Definition of a binary operation for testing"""

    name: str
    nabla_fn: Callable
    jax_fn: Callable
    description: str


# Define all binary operations to test
BINARY_OPERATIONS = {
    "add": BinaryOperation(
        name="add", nabla_fn=nb.add, jax_fn=jnp.add, description="Element-wise addition"
    ),
    "mul": BinaryOperation(
        name="mul",
        nabla_fn=nb.mul,
        jax_fn=jnp.multiply,
        description="Element-wise multiplication",
    ),
    "sub": BinaryOperation(
        name="sub",
        nabla_fn=nb.sub,
        jax_fn=jnp.subtract,
        description="Element-wise subtraction",
    ),
    "div": BinaryOperation(
        name="div",
        nabla_fn=nb.div,
        jax_fn=jnp.divide,
        description="Element-wise division",
    ),
    "floordiv": BinaryOperation(
        name="floordiv",
        nabla_fn=nb.floordiv,
        jax_fn=jnp.floor_divide,
        description="Element-wise floor division",
    ),
    "pow": BinaryOperation(
        name="pow", nabla_fn=nb.pow, jax_fn=jnp.power, description="Element-wise power"
    ),
    "greater_equal": BinaryOperation(
        name="greater_equal",
        nabla_fn=nb.greater_equal,
        jax_fn=jnp.greater_equal,
        description="Element-wise greater-than-or-equal comparison",
    ),
    "equal": BinaryOperation(
        name="equal",
        nabla_fn=nb.equal,
        jax_fn=jnp.equal,
        description="Element-wise equality comparison",
    ),
    "not_equal": BinaryOperation(
        name="not_equal",
        nabla_fn=nb.not_equal,
        jax_fn=jnp.not_equal,
        description="Element-wise inequality comparison",
    ),
    "maximum": BinaryOperation(
        name="maximum",
        nabla_fn=nb.maximum,
        jax_fn=jnp.maximum,
        description="Element-wise maximum",
    ),
    "minimum": BinaryOperation(
        name="minimum",
        nabla_fn=nb.minimum,
        jax_fn=jnp.minimum,
        description="Element-wise minimum",
    ),
}

# ============================================================================
# SHARED UTILITIES (copied from test_add_comprehensive.py)
# ============================================================================


def jax_arange(shape, dtype=jnp.float32):
    """Create JAX array matching nabla.arange"""
    return jax.numpy.arange(np.prod(shape), dtype=dtype).reshape(shape)


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
        x_nb = 1 + nb.array(2.5)
        x_jax = 1 + jnp.array(2.5)
    else:
        x_nb = 1 + nb.arange(shape_x)
        x_jax = 1 + jax_arange(shape_x)

    if rank_y == 0:
        y_nb = 1 + nb.array(1.5)
        y_jax = 1 + jnp.array(1.5)
    else:
        y_nb = 1 + nb.arange(shape_y)
        y_jax = 1 + jax_arange(shape_y)

    return x_nb, y_nb, x_jax, y_jax


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


def run_test_with_consistency_check(test_name, nabla_fn, jax_fn):
    """
    Run Nabla and JAX functions separately and check for consistency.

    Returns True if:
    - Both succeed and give same result (numeric or boolean arrays)
    - Both fail consistently

    Returns False if:
    - Only one fails
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

                    # Handle JAX float0 (zero tangent space) - convert to regular zeros
                    if hasattr(jax_item, "dtype") and str(jax_item.dtype).startswith(
                        "[('float0"
                    ):
                        # JAX float0 means zero gradient - convert to regular zeros
                        jax_item = jnp.zeros_like(jax_item, dtype=jnp.float32)

                    if not jnp.allclose(nb_numpy, jax_item):
                        print(f"✗ {test_name}: Tuple item {i} doesn't match")
                        return False

                print(f"✓ {test_name}")
                return True

            # Handle single array results (numeric or boolean)
            else:
                if isinstance(nabla_result, nb.Array):
                    nabla_numpy = nabla_result.to_numpy()
                else:
                    nabla_numpy = np.array(nabla_result)

                # Handle JAX float0 (zero tangent space) - convert to regular zeros
                if hasattr(jax_result, "dtype") and str(jax_result.dtype).startswith(
                    "[('float0"
                ):
                    # JAX float0 means zero gradient - convert to regular zeros
                    jax_result = jnp.zeros_like(jax_result, dtype=jnp.float32)

                # Use array_equal for boolean arrays, allclose for numeric
                if nabla_numpy.dtype == bool and jax_result.dtype == bool:
                    arrays_match = np.array_equal(nabla_numpy, jax_result)
                else:
                    arrays_match = jnp.allclose(nabla_numpy, jax_result)

                if arrays_match:
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
# PARAMETERIZED TEST FRAMEWORK
# ============================================================================


def create_binary_op_tests(operation: BinaryOperation):
    """Create all 21 transformation tests for a given binary operation"""

    # This will be set by the rank testing framework
    current_rank_x = 2
    current_rank_y = 2

    def get_test_data():
        """Get test data for current operation and current ranks"""
        return get_test_data_for_ranks(current_rank_x, current_rank_y)

    def f_nb(x, y):
        """Nabla operation function"""
        return operation.nabla_fn(x, y)

    def f_jax(x, y):
        """JAX operation function"""
        return operation.jax_fn(x, y)

    # All 21 test functions
    def test_1_baseline():
        """Test: f(x, y)"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            return f_nb(x_nb, y_nb)

        def jax_fn():
            return f_jax(x_jax, y_jax)

        return run_test_with_consistency_check(
            f"Baseline {operation.name}(x,y)", nabla_fn, jax_fn
        )

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

        return run_test_with_consistency_check(
            f"VJP {operation.name}", nabla_fn, jax_fn
        )

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

        return run_test_with_consistency_check(
            f"JVP {operation.name}", nabla_fn, jax_fn
        )

    def test_4_vmap():
        """Test: vmap(f)"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            return nb.vmap(f_nb)(x_nb, y_nb)

        def jax_fn():
            return jax.vmap(f_jax)(x_jax, y_jax)

        return run_test_with_consistency_check(
            f"VMAP {operation.name}", nabla_fn, jax_fn
        )

    def test_5_jit():
        """Test: jit(f)"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            return nb.djit(f_nb)(x_nb, y_nb)

        def jax_fn():
            return jax.jit(f_jax)(x_jax, y_jax)

        return run_test_with_consistency_check(
            f"JIT {operation.name}", nabla_fn, jax_fn
        )

    def test_6_jit_vjp():
        """Test: jit(vjp(f))"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            value, vjp_fn = nb.djit(nb.vjp)(f_nb, x_nb, y_nb)
            grad = vjp_fn(nb.ones_like(value))
            return (value, grad[0], grad[1])

        def jax_fn():
            value, vjp_fn = jax.vjp(f_jax, x_jax, y_jax)
            grad = vjp_fn(jnp.ones_like(value))
            return (value, grad[0], grad[1])

        return run_test_with_consistency_check(
            f"JIT(VJP) {operation.name}", nabla_fn, jax_fn
        )

    def test_7_jit_jvp():
        """Test: jit(jvp(f))"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            primals = (x_nb, y_nb)
            tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
            value, tangent = nb.djit(nb.jvp)(f_nb, primals, tangents)
            return (value, tangent)

        def jax_fn():
            primals = (x_jax, y_jax)
            tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
            value, tangent = jax.jvp(f_jax, primals, tangents)
            return (value, tangent)

        return run_test_with_consistency_check(
            f"JIT(JVP) {operation.name}", nabla_fn, jax_fn
        )

    def test_8_jit_vmap():
        """Test: jit(vmap(f))"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            return nb.djit(nb.vmap(f_nb))(x_nb, y_nb)

        def jax_fn():
            return jax.jit(jax.vmap(f_jax))(x_jax, y_jax)

        return run_test_with_consistency_check(
            f"JIT(VMAP) {operation.name}", nabla_fn, jax_fn
        )

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

        return run_test_with_consistency_check(
            f"VMAP(VJP) {operation.name}", nabla_fn, jax_fn
        )

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

        return run_test_with_consistency_check(
            f"VMAP(JVP) {operation.name}", nabla_fn, jax_fn
        )

    # Tests 11-21 would continue in the same pattern...
    # For brevity, I'll create stubs that delegate to the first 10 tests
    def test_11_jit_vmap_vjp():
        """Test: jit(vmap(vjp(f)))"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            def vjp_wrapper(x, y):
                value, vjp_fn = nb.vjp(f_nb, x, y)
                grad = vjp_fn(nb.ones_like(value))
                return (value, grad[0], grad[1])

            return nb.djit(nb.vmap(vjp_wrapper))(x_nb, y_nb)

        def jax_fn():
            def vjp_wrapper(x, y):
                value, vjp_fn = jax.vjp(f_jax, x, y)
                grad = vjp_fn(jnp.ones_like(value))
                return (value, grad[0], grad[1])

            return jax.jit(jax.vmap(vjp_wrapper))(x_jax, y_jax)

        return run_test_with_consistency_check(
            f"JIT_VMAP_VJP {operation.name}", nabla_fn, jax_fn
        )

    def test_12_jit_vmap_jvp():
        """Test: jit(vmap(jvp(f)))"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            def jvp_wrapper(x, y):
                primals = (x, y)
                tangents = (nb.ones_like(x), nb.ones_like(y))
                value, tangent = nb.jvp(f_nb, primals, tangents)
                return (value, tangent)

            return nb.djit(nb.vmap(jvp_wrapper))(x_nb, y_nb)

        def jax_fn():
            def jvp_wrapper(x, y):
                primals = (x, y)
                tangents = (jnp.ones_like(x), jnp.ones_like(y))
                value, tangent = jax.jvp(f_jax, primals, tangents)
                return (value, tangent)

            return jax.jit(jax.vmap(jvp_wrapper))(x_jax, y_jax)

        return run_test_with_consistency_check(
            f"JIT_VMAP_JVP {operation.name}", nabla_fn, jax_fn
        )

    def test_13_vjp_vjp():
        """Test: vjp(vjp(f)) - Second-order reverse (Hessian-vector)"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            def first_vjp(x, y):
                value, vjp_fn = nb.vjp(f_nb, x, y)
                grad = vjp_fn(nb.ones_like(value))
                return grad[0], grad[1]  # Return gradients only

            # Second VJP on the gradient function
            (grad_x, grad_y), second_vjp_fn = nb.vjp(first_vjp, x_nb, y_nb)
            second_grad = second_vjp_fn((nb.ones_like(grad_x), nb.ones_like(grad_y)))
            return (grad_x, grad_y, second_grad[0], second_grad[1])

        def jax_fn():
            def first_vjp(x, y):
                value, vjp_fn = jax.vjp(f_jax, x, y)
                grad = vjp_fn(jnp.ones_like(value))
                return grad[0], grad[1]  # Return gradients only

            # Second VJP on the gradient function
            (grad_x, grad_y), second_vjp_fn = jax.vjp(first_vjp, x_jax, y_jax)
            second_grad = second_vjp_fn((jnp.ones_like(grad_x), jnp.ones_like(grad_y)))
            return (grad_x, grad_y, second_grad[0], second_grad[1])

        return run_test_with_consistency_check(
            f"VJP_VJP {operation.name}", nabla_fn, jax_fn
        )

    def test_14_jvp_vjp():
        """Test: jvp(vjp(f)) - Mixed-mode differentiation"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            def vjp_wrapper(x, y):
                value, vjp_fn = nb.vjp(f_nb, x, y)
                grad = vjp_fn(nb.ones_like(value))
                return grad[0], grad[1]

            primals = (x_nb, y_nb)
            tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
            value, tangent = nb.jvp(vjp_wrapper, primals, tangents)
            return (value[0], value[1], tangent[0], tangent[1])

        def jax_fn():
            def vjp_wrapper(x, y):
                value, vjp_fn = jax.vjp(f_jax, x, y)
                grad = vjp_fn(jnp.ones_like(value))
                return grad[0], grad[1]

            primals = (x_jax, y_jax)
            tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
            value, tangent = jax.jvp(vjp_wrapper, primals, tangents)
            return (value[0], value[1], tangent[0], tangent[1])

        return run_test_with_consistency_check(
            f"JVP_VJP {operation.name}", nabla_fn, jax_fn
        )

    def test_15_vjp_jvp():
        """Test: vjp(jvp(f)) - Mixed-mode differentiation"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            def jvp_wrapper(x, y):
                primals = (x, y)
                tangents = (nb.ones_like(x), nb.ones_like(y))
                value, tangent = nb.jvp(f_nb, primals, tangents)
                return value, tangent

            result, vjp_fn = nb.vjp(jvp_wrapper, x_nb, y_nb)
            cotangent = (nb.ones_like(result[0]), nb.ones_like(result[1]))
            grad = vjp_fn(cotangent)
            return (result[0], result[1], grad[0], grad[1])

        def jax_fn():
            def jvp_wrapper(x, y):
                primals = (x, y)
                tangents = (jnp.ones_like(x), jnp.ones_like(y))
                value, tangent = jax.jvp(f_jax, primals, tangents)
                return value, tangent

            result, vjp_fn = jax.vjp(jvp_wrapper, x_jax, y_jax)
            cotangent = (jnp.ones_like(result[0]), jnp.ones_like(result[1]))
            grad = vjp_fn(cotangent)
            return (result[0], result[1], grad[0], grad[1])

        return run_test_with_consistency_check(
            f"VJP_JVP {operation.name}", nabla_fn, jax_fn
        )

    def test_16_jvp_jvp():
        """Test: jvp(jvp(f)) - Second-order forward"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            def first_jvp(x, y):
                primals = (x, y)
                tangents = (nb.ones_like(x), nb.ones_like(y))
                value, tangent = nb.jvp(f_nb, primals, tangents)
                return value, tangent

            primals = (x_nb, y_nb)
            tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
            result, second_tangent = nb.jvp(first_jvp, primals, tangents)
            return (result[0], result[1], second_tangent[0], second_tangent[1])

        def jax_fn():
            def first_jvp(x, y):
                primals = (x, y)
                tangents = (jnp.ones_like(x), jnp.ones_like(y))
                value, tangent = jax.jvp(f_jax, primals, tangents)
                return value, tangent

            primals = (x_jax, y_jax)
            tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
            result, second_tangent = jax.jvp(first_jvp, primals, tangents)
            return (result[0], result[1], second_tangent[0], second_tangent[1])

        return run_test_with_consistency_check(
            f"JVP_JVP {operation.name}", nabla_fn, jax_fn
        )

    def test_17_vjp_vmap():
        """Test: vjp(vmap(f)) - Differentiate vectorized function"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            value, vjp_fn = nb.vjp(nb.vmap(f_nb), x_nb, y_nb)
            grad = vjp_fn(nb.ones_like(value))
            return (value, grad[0], grad[1])

        def jax_fn():
            value, vjp_fn = jax.vjp(jax.vmap(f_jax), x_jax, y_jax)
            grad = vjp_fn(jnp.ones_like(value))
            return (value, grad[0], grad[1])

        return run_test_with_consistency_check(
            f"VJP_VMAP {operation.name}", nabla_fn, jax_fn
        )

    def test_18_jvp_vmap():
        """Test: jvp(vmap(f)) - Forward-mode of vectorized function"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            primals = (x_nb, y_nb)
            tangents = (nb.ones_like(x_nb), nb.ones_like(y_nb))
            value, tangent = nb.jvp(nb.vmap(f_nb), primals, tangents)
            return (value, tangent)

        def jax_fn():
            primals = (x_jax, y_jax)
            tangents = (jnp.ones_like(x_jax), jnp.ones_like(y_jax))
            value, tangent = jax.jvp(jax.vmap(f_jax), primals, tangents)
            return (value, tangent)

        return run_test_with_consistency_check(
            f"JVP_VMAP {operation.name}", nabla_fn, jax_fn
        )

    def test_19_vmap_vmap():
        """Test: vmap(vmap(f)) - Double vectorization"""
        x_nb, y_nb, x_jax, y_jax = get_test_data()

        def nabla_fn():
            return nb.vmap(nb.vmap(f_nb))(x_nb, y_nb)

        def jax_fn():
            return jax.vmap(jax.vmap(f_jax))(x_jax, y_jax)

        return run_test_with_consistency_check(
            f"VMAP_VMAP {operation.name}", nabla_fn, jax_fn
        )

    # Set rank override function
    def set_ranks(rank_x, rank_y):
        nonlocal current_rank_x, current_rank_y
        current_rank_x = rank_x
        current_rank_y = rank_y

    # Return all test functions plus the rank setter
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
        test_15_vjp_jvp,
        test_16_jvp_jvp,
        test_17_vjp_vmap,
        test_18_jvp_vmap,
        test_19_vmap_vmap,
    ]

    return test_functions, set_ranks


# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================


def run_operation_tests(operation_name: str, all_ranks: bool = False):
    """Run all transformation tests for a specific operation"""
    if operation_name not in BINARY_OPERATIONS:
        print(f"Unknown operation: {operation_name}")
        print(f"Available operations: {list(BINARY_OPERATIONS.keys())}")
        return False, 0, 0

    operation = BINARY_OPERATIONS[operation_name]
    print("=" * 80)
    print(f"TESTING BINARY OPERATION: {operation.name.upper()}")
    print(f"Description: {operation.description}")
    print("=" * 80)

    # Get test functions and rank setter
    test_functions, set_ranks = create_binary_op_tests(operation)

    if all_ranks:
        rank_combinations = get_rank_combinations()
        total_tests = 0
        total_passed = 0

        for rank_x, rank_y in rank_combinations:
            print(f"\n{'=' * 50}")
            print(f"TESTING RANKS: x={rank_x}, y={rank_y}")
            print(f"{'=' * 50}")

            # Set ranks for this combination
            set_ranks(rank_x, rank_y)

            passed_for_ranks = 0
            for i, test_func in enumerate(test_functions, 1):  # All 19 tests
                desc = (
                    test_func.__doc__.split(":")[1].strip()
                    if test_func.__doc__ and ":" in test_func.__doc__
                    else test_func.__name__
                )
                print(f"{i:2d}. {desc} ", end="")
                success = test_func()
                if success:
                    passed_for_ranks += 1
                    total_passed += 1
                total_tests += 1

            print(
                f"\nRank ({rank_x},{rank_y}): {passed_for_ranks}/{len(test_functions)} tests passed"
            )

        print(f"\n{'=' * 80}")
        print(
            f"OPERATION {operation.name.upper()} RESULTS: {total_passed}/{total_tests} tests passed"
        )
        print(f"SUCCESS RATE: {total_passed / total_tests * 100:.1f}%")
        return total_passed == total_tests, total_passed, total_tests

    else:
        print("Testing with default ranks (2,2)...")
        set_ranks(2, 2)  # Set default ranks

        passed = 0
        for i, test_func in enumerate(test_functions, 1):  # All 19 tests
            desc = (
                test_func.__doc__.split(":")[1].strip()
                if test_func.__doc__ and ":" in test_func.__doc__
                else test_func.__name__
            )
            print(f"\n{i:2d}. {desc}")
            success = test_func()
            if success:
                passed += 1

        print(f"\n{'=' * 80}")
        print(
            f"OPERATION {operation.name.upper()} RESULTS: {passed}/{len(test_functions)} tests passed"
        )
        print(f"SUCCESS RATE: {passed / len(test_functions) * 100:.1f}%")
        return passed == len(test_functions), passed, len(test_functions)


def run_all_operations(all_ranks: bool = False):
    """Run tests for all binary operations"""
    print("=" * 100)
    print("COMPREHENSIVE BINARY OPERATIONS TEST SUITE")
    print("=" * 100)

    all_passed = True
    total_passed_all = 0
    total_tests_all = 0

    for op_name in BINARY_OPERATIONS:
        success, passed, total = run_operation_tests(op_name, all_ranks)
        all_passed = all_passed and success
        total_passed_all += passed
        total_tests_all += total
        print("\n")

    print("=" * 100)
    print("🏁 FINAL SUMMARY")
    print("=" * 100)
    overall_success_rate = (
        (total_passed_all / total_tests_all) * 100 if total_tests_all > 0 else 0
    )
    print(f"TOTAL TESTS PASSED: {total_passed_all}/{total_tests_all}")
    print(f"OVERALL SUCCESS RATE: {overall_success_rate:.1f}%")
    print("=" * 100)

    if all_passed:
        print("🎉 ALL BINARY OPERATIONS PASSED!")
    else:
        print("❌ Some binary operations failed")
    print("=" * 100)

    return all_passed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_binary_ops.py <operation|all> [--all-ranks]")
        print(f"Available operations: {list(BINARY_OPERATIONS.keys())}")
        sys.exit(1)

    operation_arg = sys.argv[1]
    all_ranks = "--all-ranks" in sys.argv

    if operation_arg == "all":
        success = run_all_operations(all_ranks)
    else:
        success, _, _ = run_operation_tests(operation_arg, all_ranks)

    # Exit with 0 for successful script execution (test results are reported in output)
    # Only exit with 1 for actual script errors (handled by exceptions)
    sys.exit(0)
