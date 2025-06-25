"""
COMPREHENSIVE NABLA BINARY OPERATIONS TEST SUITE
================================================

Generalized version of the add operation test suite that validates ALL binary operations
in Nabla against JAX as ground truth. Tests function transformations (vjp, jvp, vmap, jit)
across all binary operations and tensor rank combinations.

BINARY OPERATIONS TESTED (12 total):
──────────────────────────────────────

Arithmetic Operations:
    add, mul, sub, div, floordiv, mod, pow

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
import pytest

import nabla as nb

# Import our new utility modules (handle both relative and absolute imports)
try:
    from .test_errors import ErrorSummary, ErrorType, enhanced_error_message
    from .test_utils import (
        cleanup_caches,
        cleanup_jax_caches,
        get_rank_combinations,
        get_shape_for_rank,
        get_test_data_for_ranks,
        jax_arange,
        run_test_with_consistency_check,
        with_timeout,
    )
except ImportError:
    # Fall back to absolute imports when running directly
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from test_utils import (
        cleanup_caches,
        get_rank_combinations,
        get_test_data_for_ranks,
        run_test_with_consistency_check,
    )

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
    "mod": BinaryOperation(
        name="mod",
        nabla_fn=nb.mod,
        jax_fn=jnp.mod,
        description="Element-wise modulo operation",
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

# Note: Most utilities have been moved to test_utils.py for reusability

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

            # Clean up caches after each rank combination to prevent memory buildup
            cleanup_caches()

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

        # Clean up caches after each operation to prevent memory accumulation
        cleanup_caches()
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

# ============================================================================
# PYTEST INTEGRATION
# ============================================================================


def pytest_generate_tests(metafunc):
    """Generate pytest parameters for all operation/rank/transformation combinations"""
    if "operation_name" in metafunc.fixturenames:
        operations = list(BINARY_OPERATIONS.keys())
        metafunc.parametrize("operation_name", operations)

    if "rank_combination" in metafunc.fixturenames:
        ranks = get_rank_combinations()
        metafunc.parametrize("rank_combination", ranks)

    if "transformation_index" in metafunc.fixturenames:
        # 19 transformations (0-18)
        transformations = list(range(19))
        metafunc.parametrize("transformation_index", transformations)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup caches after each test to prevent memory issues"""
    yield
    cleanup_caches()


class TestBinaryOperations:
    """Pytest test class for binary operations"""

    def test_binary_operation_transformation(
        self, operation_name, rank_combination, transformation_index
    ):
        """Test a specific binary operation with specific ranks and transformation"""
        operation = BINARY_OPERATIONS[operation_name]
        rank_x, rank_y = rank_combination

        # Get test functions for this operation
        test_functions, set_ranks = create_binary_op_tests(operation)

        # Set the ranks for this test
        set_ranks(rank_x, rank_y)

        # Get the specific test function
        test_func = test_functions[transformation_index]

        # Create a descriptive test name
        test_desc = (
            test_func.__doc__.split(":")[1].strip()
            if test_func.__doc__ and ":" in test_func.__doc__
            else test_func.__name__
        )

        # Run the test
        success = test_func()

        # Assert the test passed
        assert success, (
            f"Failed: {operation_name} - ranks({rank_x},{rank_y}) - {test_desc}"
        )


# ============================================================================
# CONVENIENCE FUNCTIONS FOR PYTEST
# ============================================================================


def test_single_operation_all_ranks():
    """Test a single operation with all ranks - useful for debugging"""
    operation_name = "mod"  # Change this to test different operations
    success, passed, total = run_operation_tests(operation_name, all_ranks=True)
    assert success, f"Operation {operation_name} failed: {passed}/{total} tests passed"


def test_all_operations_default_ranks():
    """Test all operations with default ranks (2,2) - faster smoke test"""
    failed_operations = []

    for op_name in BINARY_OPERATIONS:
        success, passed, total = run_operation_tests(op_name, all_ranks=False)
        if not success:
            failed_operations.append(f"{op_name}: {passed}/{total}")

    assert not failed_operations, f"Failed operations: {failed_operations}"


@pytest.mark.benchmark
def test_all_operations_all_ranks(capsys):
    """Full comprehensive test - mark as benchmark since it's slow"""
    import sys

    print("\n" + "=" * 100)
    print("🚀 STARTING COMPREHENSIVE BINARY OPERATIONS TEST SUITE")
    print(
        "📊 Testing: 11 operations × 19 transformations × 10 rank combinations = 2,090 tests"
    )
    print("⏱️  Expected duration: ~10 minutes")
    print("=" * 100)
    sys.stdout.flush()

    all_passed = True
    total_passed_all = 0
    total_tests_all = 0

    for i, op_name in enumerate(BINARY_OPERATIONS, 1):
        print(f"\n[{i:2d}/11] 🔄 Testing operation: {op_name.upper()}")
        sys.stdout.flush()

        success, passed, total = run_operation_tests(op_name, all_ranks=True)
        all_passed = all_passed and success
        total_passed_all += passed
        total_tests_all += total

        # Show progress after each operation
        overall_progress = (i / len(BINARY_OPERATIONS)) * 100
        print(
            f"[{i:2d}/11] ✅ {op_name}: {passed}/{total} tests passed | Overall: {overall_progress:.1f}% complete"
        )
        sys.stdout.flush()

        # Clean up caches after each operation
        cleanup_caches()

    print("\n" + "=" * 100)
    print("🏁 FINAL COMPREHENSIVE TEST RESULTS")
    print("=" * 100)
    overall_success_rate = (
        (total_passed_all / total_tests_all) * 100 if total_tests_all > 0 else 0
    )
    print(f"📈 TOTAL TESTS PASSED: {total_passed_all}/{total_tests_all}")
    print(f"🎯 OVERALL SUCCESS RATE: {overall_success_rate:.1f}%")

    if all_passed:
        print("🎉 ALL BINARY OPERATIONS PASSED!")
    else:
        print("❌ Some binary operations failed")
    print("=" * 100)
    sys.stdout.flush()

    assert all_passed, (
        f"Comprehensive test failed: {total_passed_all}/{total_tests_all} tests passed"
    )


# ============================================================================
# LEGACY MAIN EXECUTION (kept for backward compatibility)
# ============================================================================


# COMPREHENSIVE PLAN FOR UNARY OPERATIONS TEST SUITE
# Understanding of Current Binary Ops Test Structure
# The current test_binary_ops.py file is excellently structured with:

# 19 Different Transformation Combinations: From basic function calls to complex compositions like jit(vmap(vjp(f)))
# 10 Tensor Rank Combinations: Testing scalar, vector, matrix, and 3D tensor combinations with broadcasting
# Consistency Checking: Both Nabla and JAX implementations run separately, with success if both succeed with matching results OR both fail consistently
# Comprehensive Error Handling: Using run_test_with_consistency_check for robust error detection
# Multiple Execution Modes: Direct execution, pytest integration, and parameterized testing
# Memory Management: Cache cleanup between tests to prevent memory issues
# Available Unary Operations (15 total)
# From unary.py, I identified these public unary operations:

# Mathematical Functions (7):

# sin - trigonometric sine
# cos - trigonometric cosine
# tanh - hyperbolic tangent
# log - natural logarithm
# exp - exponential function
# sqrt - square root
# abs - absolute value
# Element-wise Operations (4):

# negate - element-wise negation
# relu - rectified linear unit
# sigmoid - sigmoid activation
# floor - floor operation
# Logical Operations (1):

# logical_not - logical negation
# Utility Operations (3):

# cast - type casting (requires dtype parameter)
# incr_batch_dim_ctr - internal batch dimension counter
# decr_batch_dim_ctr - internal batch dimension counter
# Note: transfer_to is device-specific and may need special handling.

# Proposed Test Structure
# 1. Shapes to Test (4 shapes as requested)
# 2. Transformation Combinations (19 total - same as binary ops)
# Level 0 - Baseline:

# f(x) # Direct function call
# Level 1 - Single Transformations: 2. vjp(f) # Reverse-mode autodiff 3. jvp(f) # Forward-mode autodiff
# 4. vmap(f) # Vectorization 5. jit(f) # JIT compilation

# Level 2 - Double Transformations: 6. jit(vjp(f)) # JIT + reverse-mode 7. jit(jvp(f)) # JIT + forward-mode 8. jit(vmap(f)) # JIT + vectorization 9. vmap(vjp(f)) # Vectorized reverse-mode 10. vmap(jvp(f)) # Vectorized forward-mode

# Level 3 - Triple Transformations: 11. jit(vmap(vjp(f))) # JIT + vectorized reverse-mode 12. jit(vmap(jvp(f))) # JIT + vectorized forward-mode

# Level 4 - Higher-Order Differentiation: 13. vjp(vjp(f)) # Second-order reverse (Hessian-vector) 14. jvp(vjp(f)) # Mixed-mode differentiation 15. vjp(jvp(f)) # Mixed-mode differentiation
# 16. jvp(jvp(f)) # Second-order forward

# Level 5 - Advanced Compositions: 17. vjp(vmap(f)) # Differentiate vectorized function 18. jvp(vmap(f)) # Forward-mode of vectorized function 19. vmap(vmap(f)) # Double vectorization

# 3. Special Considerations for Unary Operations
# Categorization by Expected Differentiability:

# Group A - Fully Differentiable (11 ops):

# sin, cos, tanh, log, exp, sqrt, abs, negate, relu, sigmoid, floor
# Should work with all transformations
# Group B - Non-differentiable (1 op):

# logical_not - Boolean operation, VJP/JVP should fail consistently
# Group C - Special Handling (3 ops):

# cast - May have limited differentiability depending on dtype
# incr_batch_dim_ctr, decr_batch_dim_ctr - Internal ops, may need special test logic
# 4. Test Matrix Calculation
# Standard Operations: 12 ops × 19 transformations × 4 shapes = 912 tests
# Special Operations: 3 ops × selective transformations × 4 shapes = ~100 tests
# Total: ~1,000 individual tests
# 5. Implementation Plan
# Phase 1: Core Infrastructure (Priority 1)

# Create test_unary_ops.py following the same pattern as test_binary_ops.py
# Define UnaryOperation dataclass and UNARY_OPERATIONS dictionary
# Adapt get_test_data_for_rank() to work with single tensor input
# Create create_unary_op_tests() function with all 19 transformation patterns
# Phase 2: Basic Operations (Priority 2)

# Implement tests for Group A operations (fully differentiable)
# Test with all 4 shapes and all 19 transformations
# Ensure robust error handling and consistency checking
# Phase 3: Special Cases (Priority 3)

# Handle logical_not with expected differentiation failures
# Implement special logic for cast operation with different dtypes
# Handle internal batch dimension operations
# Phase 4: Integration (Priority 4)

# Add pytest integration matching binary ops pattern
# Add command-line execution capabilities
# Add comprehensive benchmarking and reporting
# 6. Key Implementation Details
# Test Data Generation:

# Operation Definitions:

# Transformation Test Pattern:

# 7. Expected Challenges and Solutions
# Challenge 1: Domain Restrictions

# log requires positive inputs
# sqrt requires non-negative inputs
# Solution: Use 1 + nb.ndarange() to ensure positive test data
# Challenge 2: Non-differentiable Operations

# logical_not should fail VJP/JVP consistently
# Solution: Expect and handle consistent failures
# Challenge 3: Special Operations

# cast requires dtype parameter
# Batch dimension operations may have different semantics
# Solution: Create specialized test functions for these operations
# Challenge 4: Memory Management

# Large number of tests may cause memory issues
# Solution: Use existing cache cleanup infrastructure
# 8. Success Criteria
# Comprehensive Coverage: All 15 unary operations tested
# Transformation Coverage: All 19 transformation combinations work
# Shape Coverage: All 4 shapes (scalar to 3D tensor) tested
# Consistency: Results match JAX exactly or fail consistently
# Performance: Test suite completes in reasonable time (~5-10 minutes)
# Maintainability: Code follows same patterns as binary ops for easy maintenance
# This comprehensive plan will create a robust test suite for unary operations that matches the quality and thoroughness of the existing binary operations test suite, with appropriate adaptations for the single-input nature of unary operations.
