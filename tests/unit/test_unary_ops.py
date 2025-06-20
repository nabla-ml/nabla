"""
COMPREHENSIVE NABLA UNARY OPERATIONS TEST SUITE
================================================

This test suite validates ALL unary operations in Nabla against JAX as a ground truth.
It is a generalized version of the binary operations test suite, applying the same
rigorous testing methodology to unary functions. It covers function transformations
(vjp, jvp, vmap, jit) across all core unary operations and tensor ranks.

UNARY OPERATIONS TESTED (12 total):
──────────────────────────────────

Mathematical Functions:
    sin, cos, tanh, log, exp, sqrt, abs, floor

Element-wise Operations:
    negate, relu, sigmoid

Logical Operations:
    logical_not

TRANSFORMATION COMBINATIONS TESTED (19 total):
──────────────────────────────────────────────
(Same 19 combinations as the binary operations suite)

Level 0 - Baseline:
    1. f(x)

Level 1 - Single Transformations:
    2. vjp(f), 3. jvp(f), 4. vmap(f), 5. jit(f)

Level 2 - Double Transformations:
    6. jit(vjp(f)), 7. jit(jvp(f)), 8. jit(vmap(f)),
    9. vmap(vjp(f)), 10. vmap(jvp(f))

Level 3 - Triple Transformations:
    11. jit(vmap(vjp(f))), 12. jit(vmap(jvp(f)))

Level 4 - Higher-Order Differentiation:
    13. vjp(vjp(f)), 14. jvp(vjp(f)), 15. vjp(jvp(f)), 16. jvp(jvp(f))

Level 5 - Advanced Compositions:
    17. vjp(vmap(f)), 18. jvp(vmap(f)), 19. vmap(vmap(f))

TENSOR RANK COMBINATIONS TESTED (4 total):
──────────────────────────────────────────
    0: scalar   # f(2.5)
    1: vector   # f([1,2,3,4])
    2: matrix   # f([[1,2,3],[4,5,6]])
    3: tensor3D # f([2,2,3])

Total Test Matrix: 12 operations × 19 transformations × 4 rank combinations = 912 individual tests

CONSISTENCY CHECKING LOGIC:
─────────────────────────
The test logic is identical to the binary suite:

✅ PASS if both Nabla and JAX succeed and produce identical results.
✅ PASS if both fail consistently (e.g., differentiating boolean operations).
❌ FAIL if only one framework fails or if results do not match.

This robustly handles functions with domain restrictions (log, sqrt) and non-differentiable
points (relu, floor, abs).

USAGE:
──────
Run tests for a specific operation:
    pytest test_unary_ops.py -k "sin"
    python test_unary_ops.py relu

Run tests for all operations:
    pytest test_unary_ops.py
    python test_unary_ops.py all

Run all operations with all rank combinations (via command line):
    python test_unary_ops.py all --all-ranks
"""

import sys

# Assume nabla and test utilities are in the path
# sys.path.insert(0, "/path/to/nabla/project")
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

import nabla as nb

# Import utility modules
try:
    from .test_utils import (
        cleanup_caches,
        get_shape_for_rank,
        jax_arange,
        run_test_with_consistency_check,
    )
except ImportError:
    # Fallback for direct execution
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from test_utils import (
        cleanup_caches,
        get_shape_for_rank,
        jax_arange,
        run_test_with_consistency_check,
    )

# ============================================================================
# UTILITY FUNCTIONS FOR UNARY OPERATIONS
# ============================================================================


def get_ranks_to_test() -> list[int]:
    """Get all ranks to test for unary operations."""
    return [0, 1, 2, 3]  # scalar, vector, matrix, 3D tensor


def get_test_data_for_rank(
    rank: int, domain_positive: bool = False, dtype: str = "float32"
) -> tuple[nb.Array, jnp.ndarray]:
    """Get test data for specific tensor rank for unary operations."""
    shape = get_shape_for_rank(rank)

    if dtype == "bool":
        # Boolean data for logical operations
        if rank == 0:
            x_nb = nb.array(True)
            x_jax = jnp.array(True)
        else:
            # Create alternating boolean pattern using modulo and equal
            x_nb = nb.equal(nb.arange(shape) % 2, nb.array(0))
            x_jax = jax_arange(shape) % 2 == 0
    else:
        # Numeric data
        if rank == 0:
            base_value = 2.5 if domain_positive else 1.5
            x_nb = nb.array(base_value)
            x_jax = jnp.array(base_value)
        else:
            if domain_positive:
                # For log, sqrt - ensure positive values
                x_nb = 1 + nb.arange(shape)
                x_jax = 1 + jax_arange(shape)
            else:
                # Regular test data
                x_nb = nb.arange(shape) + 1
                x_jax = jax_arange(shape) + 1

    return x_nb, x_jax


def get_tangent_for_jvp(x_nb, x_jax, operation):
    """Generate appropriate tangents for JVP tests.

    For boolean operations, JAX requires float0 tangents, so we use zeros.
    For numeric operations, we use ones_like.
    """
    if operation.input_dtype == "bool":
        # For boolean operations, use zero tangents since derivatives should be zero
        # JAX requires float0 tangents for boolean primals
        tangent_nb = nb.zeros_like(x_nb).astype(nb.DType.float32)
        # JAX needs float0 dtype specifically
        import jax._src.dtypes as jdt

        tangent_jax = jnp.zeros_like(x_jax, dtype=jdt.float0)
    else:
        # For numeric operations, use ones_like tangents
        tangent_nb = nb.ones_like(x_nb)
        tangent_jax = jnp.ones_like(x_jax)

    return tangent_nb, tangent_jax


# ============================================================================
# UNARY OPERATION DEFINITIONS
# ============================================================================


@dataclass
class UnaryOperation:
    """Definition of a unary operation for testing."""

    name: str
    nabla_fn: Callable
    jax_fn: Callable
    description: str
    domain_positive: bool = False  # True for ops like log, sqrt
    input_dtype: str = "float32"  # "bool" for logical ops


# Define all unary operations to test
UNARY_OPERATIONS = {
    # Mathematical Functions
    "sin": UnaryOperation("sin", nb.sin, jnp.sin, "Trigonometric sine"),
    "cos": UnaryOperation("cos", nb.cos, jnp.cos, "Trigonometric cosine"),
    "tanh": UnaryOperation("tanh", nb.tanh, jnp.tanh, "Hyperbolic tangent"),
    "log": UnaryOperation(
        "log", nb.log, jnp.log, "Natural logarithm", domain_positive=True
    ),
    "exp": UnaryOperation("exp", nb.exp, jnp.exp, "Exponential function"),
    "sqrt": UnaryOperation(
        "sqrt", nb.sqrt, jnp.sqrt, "Square root", domain_positive=True
    ),
    "abs": UnaryOperation("abs", nb.abs, jnp.abs, "Absolute value"),
    "floor": UnaryOperation("floor", nb.floor, jnp.floor, "Floor operation"),
    # Element-wise Operations
    "negate": UnaryOperation(
        "negate", nb.negate, jnp.negative, "Element-wise negation"
    ),
    "relu": UnaryOperation(
        "relu", nb.relu, jax.nn.relu, "Rectified Linear Unit (ReLU)"
    ),
    "sigmoid": UnaryOperation(
        "sigmoid", nb.sigmoid, jax.nn.sigmoid, "Sigmoid activation"
    ),
    # Logical Operations
    "logical_not": UnaryOperation(
        "logical_not",
        nb.logical_not,
        jnp.logical_not,
        "Logical NOT",
        input_dtype="bool",
    ),
}

# ============================================================================
# PARAMETERIZED TEST FRAMEWORK
# ============================================================================


def create_unary_op_tests(operation: UnaryOperation):
    """Create all 19 transformation tests for a given unary operation."""

    current_rank = 2  # Default rank

    def get_test_data():
        """Get test data for the current operation and rank."""
        return get_test_data_for_rank(
            current_rank,
            domain_positive=operation.domain_positive,
            dtype=operation.input_dtype,
        )

    def f_nb(x):
        return operation.nabla_fn(x)

    def f_jax(x):
        return operation.jax_fn(x)

    # All 19 test functions, adapted for unary operations
    def test_1_baseline():
        """Test: f(x)"""
        x_nb, x_jax = get_test_data()
        return run_test_with_consistency_check(
            f"Baseline {operation.name}(x)", lambda: f_nb(x_nb), lambda: f_jax(x_jax)
        )

    def test_2_vjp():
        """Test: vjp(f)"""
        x_nb, x_jax = get_test_data()

        def nabla_fn():
            value, vjp_fn = nb.vjp(f_nb, x_nb)
            grad = vjp_fn(nb.ones_like(value))  # Nabla returns single value for unary
            return value, grad

        def jax_fn():
            value, vjp_fn = jax.vjp(f_jax, x_jax)
            grad = vjp_fn(jnp.ones_like(value))
            return value, grad[0]  # JAX returns tuple, extract single gradient

        return run_test_with_consistency_check(
            f"VJP {operation.name}", nabla_fn, jax_fn
        )

    def test_3_jvp():
        """Test: jvp(f)"""
        x_nb, x_jax = get_test_data()
        tangent_nb, tangent_jax = get_tangent_for_jvp(x_nb, x_jax, operation)

        def nabla_fn():
            return nb.jvp(f_nb, (x_nb,), (tangent_nb,))

        def jax_fn():
            return jax.jvp(f_jax, (x_jax,), (tangent_jax,))

        return run_test_with_consistency_check(
            f"JVP {operation.name}", nabla_fn, jax_fn
        )

    def test_4_vmap():
        """Test: vmap(f)"""
        x_nb, x_jax = get_test_data()
        return run_test_with_consistency_check(
            f"VMAP {operation.name}",
            lambda: nb.vmap(f_nb)(x_nb),
            lambda: jax.vmap(f_jax)(x_jax),
        )

    def test_5_jit():
        """Test: jit(f)"""
        x_nb, x_jax = get_test_data()
        return run_test_with_consistency_check(
            f"JIT {operation.name}",
            lambda: nb.djit(f_nb)(x_nb),
            lambda: jax.jit(f_jax)(x_jax),
        )

    def test_6_jit_vjp():
        """Test: jit(vjp(f))"""
        x_nb, x_jax = get_test_data()

        def vjp_wrapper(x):
            value, vjp_fn = nb.vjp(f_nb, x)
            grad = vjp_fn(nb.ones_like(value))  # Nabla returns single value
            return value, grad

        def vjp_wrapper_jax(x):
            value, vjp_fn = jax.vjp(f_jax, x)
            grad = vjp_fn(jnp.ones_like(value))
            return value, grad[0]  # JAX returns tuple, extract single gradient

        return run_test_with_consistency_check(
            f"JIT(VJP) {operation.name}",
            lambda: nb.djit(vjp_wrapper)(x_nb),
            lambda: jax.jit(vjp_wrapper_jax)(x_jax),
        )

    def test_7_jit_jvp():
        """Test: jit(jvp(f))"""
        x_nb, x_jax = get_test_data()
        tangent_nb, tangent_jax = get_tangent_for_jvp(x_nb, x_jax, operation)

        def jvp_wrapper(x, tangent):
            return nb.jvp(f_nb, (x,), (tangent,))

        def jvp_wrapper_jax(x, tangent):
            return jax.jvp(f_jax, (x,), (tangent,))

        return run_test_with_consistency_check(
            f"JIT(JVP) {operation.name}",
            lambda: nb.djit(jvp_wrapper)(x_nb, tangent_nb),
            lambda: jax.jit(jvp_wrapper_jax)(x_jax, tangent_jax),
        )

    def test_8_jit_vmap():
        """Test: jit(vmap(f))"""
        x_nb, x_jax = get_test_data()
        return run_test_with_consistency_check(
            f"JIT(VMAP) {operation.name}",
            lambda: nb.djit(nb.vmap(f_nb))(x_nb),
            lambda: jax.jit(jax.vmap(f_jax))(x_jax),
        )

    def test_9_vmap_vjp():
        """Test: vmap(vjp_value_and_grad(f))"""
        x_nb, x_jax = get_test_data()

        def vjp_wrapper(x):
            value, vjp_fn = nb.vjp(f_nb, x)
            grad = vjp_fn(nb.ones_like(value))  # Nabla returns single value
            return value, grad

        def vjp_wrapper_jax(x):
            value, vjp_fn = jax.vjp(f_jax, x)
            grad = vjp_fn(jnp.ones_like(value))
            return value, grad[0]  # JAX returns tuple, extract single gradient

        return run_test_with_consistency_check(
            f"VMAP(VJP) {operation.name}",
            lambda: nb.vmap(vjp_wrapper)(x_nb),
            lambda: jax.vmap(vjp_wrapper_jax)(x_jax),
        )

    def test_10_vmap_jvp():
        """Test: vmap(jvp_value_and_tangent(f))"""
        x_nb, x_jax = get_test_data()
        # For vmap operations, we need to use proper tangent generation
        if operation.input_dtype == "bool":
            # Use float32 tangents for Nabla, float0 tangents for JAX
            def jvp_wrapper(x):
                tangent = nb.zeros_like(x).astype(nb.DType.float32)
                return nb.jvp(f_nb, (x,), (tangent,))

            def jvp_wrapper_jax(x):
                import jax._src.dtypes as jdt

                tangent = jnp.zeros_like(x, dtype=jdt.float0)
                return jax.jvp(f_jax, (x,), (tangent,))
        else:
            # Use ones_like tangents for numeric operations
            def jvp_wrapper(x):
                return nb.jvp(f_nb, (x,), (nb.ones_like(x),))

            def jvp_wrapper_jax(x):
                return jax.jvp(f_jax, (x,), (jnp.ones_like(x),))

        return run_test_with_consistency_check(
            f"VMAP(JVP) {operation.name}",
            lambda: nb.vmap(jvp_wrapper)(x_nb),
            lambda: jax.vmap(jvp_wrapper_jax)(x_jax),
        )

    def test_11_jit_vmap_vjp():
        """Test: jit(vmap(vjp(f)))"""
        x_nb, x_jax = get_test_data()

        def vjp_wrapper(x):
            value, vjp_fn = nb.vjp(f_nb, x)
            grad = vjp_fn(nb.ones_like(value))  # Nabla returns single value
            return value, grad

        def vjp_wrapper_jax(x):
            value, vjp_fn = jax.vjp(f_jax, x)
            grad = vjp_fn(jnp.ones_like(value))
            return value, grad[0]  # JAX returns tuple, extract single gradient

        return run_test_with_consistency_check(
            f"JIT(VMAP(VJP)) {operation.name}",
            lambda: nb.djit(nb.vmap(vjp_wrapper))(x_nb),
            lambda: jax.jit(jax.vmap(vjp_wrapper_jax))(x_jax),
        )

    def test_12_jit_vmap_jvp():
        """Test: jit(vmap(jvp(f)))"""
        x_nb, x_jax = get_test_data()
        # Same pattern as test_10_vmap_jvp but with JIT
        if operation.input_dtype == "bool":

            def jvp_wrapper(x):
                tangent = nb.zeros_like(x).astype(nb.DType.float32)
                return nb.jvp(f_nb, (x,), (tangent,))

            def jvp_wrapper_jax(x):
                import jax._src.dtypes as jdt

                tangent = jnp.zeros_like(x, dtype=jdt.float0)
                return jax.jvp(f_jax, (x,), (tangent,))
        else:

            def jvp_wrapper(x):
                return nb.jvp(f_nb, (x,), (nb.ones_like(x),))

            def jvp_wrapper_jax(x):
                return jax.jvp(f_jax, (x,), (jnp.ones_like(x),))

        return run_test_with_consistency_check(
            f"JIT(VMAP(JVP)) {operation.name}",
            lambda: nb.djit(nb.vmap(jvp_wrapper))(x_nb),
            lambda: jax.jit(jax.vmap(jvp_wrapper_jax))(x_jax),
        )

    def test_13_vjp_vjp():
        """Test: vjp(vjp(f))"""
        x_nb, x_jax = get_test_data()

        def nabla_fn():
            def first_grad(x):
                _, vjp_fn = nb.vjp(f_nb, x)
                return vjp_fn(nb.ones_like(f_nb(x)))  # Nabla returns single value

            value, vjp_fn_2 = nb.vjp(first_grad, x_nb)
            grad2 = vjp_fn_2(nb.ones_like(value))  # Nabla returns single value
            return value, grad2

        def jax_fn():
            def first_grad(x):
                _, vjp_fn = jax.vjp(f_jax, x)
                grad = vjp_fn(jnp.ones_like(f_jax(x)))
                return grad[0]  # JAX returns tuple, extract single gradient

            value, vjp_fn_2 = jax.vjp(first_grad, x_jax)
            grad2 = vjp_fn_2(jnp.ones_like(value))
            return value, grad2[0]  # JAX returns tuple, extract single gradient

        return run_test_with_consistency_check(
            f"VJP(VJP) {operation.name}", nabla_fn, jax_fn
        )

    def test_14_jvp_vjp():
        """Test: jvp(vjp(f))"""
        x_nb, x_jax = get_test_data()
        tangent_nb, tangent_jax = get_tangent_for_jvp(x_nb, x_jax, operation)

        def nabla_fn():
            def first_grad(x):
                _, vjp_fn = nb.vjp(f_nb, x)
                return vjp_fn(nb.ones_like(f_nb(x)))  # Nabla returns single value

            return nb.jvp(first_grad, (x_nb,), (tangent_nb,))

        def jax_fn():
            def first_grad(x):
                _, vjp_fn = jax.vjp(f_jax, x)
                grad = vjp_fn(jnp.ones_like(f_jax(x)))
                return grad[0]  # JAX returns tuple, extract single gradient

            return jax.jvp(first_grad, (x_jax,), (tangent_jax,))

        return run_test_with_consistency_check(
            f"JVP(VJP) {operation.name}", nabla_fn, jax_fn
        )

    def test_15_vjp_jvp():
        """Test: vjp(jvp(f))"""
        x_nb, x_jax = get_test_data()
        tangent_nb, tangent_jax = get_tangent_for_jvp(x_nb, x_jax, operation)

        def nabla_fn():
            def jvp_wrapper(x):
                return nb.jvp(f_nb, (x,), (tangent_nb,))

            value, vjp_fn = nb.vjp(jvp_wrapper, x_nb)
            # Cotangent must match structure of primal output (value, tangent)
            cotangent = (nb.ones_like(value[0]), nb.ones_like(value[1]))
            grad = vjp_fn(cotangent)  # Nabla returns single value for unary
            return value, grad

        def jax_fn():
            def jvp_wrapper(x):
                return jax.jvp(f_jax, (x,), (tangent_jax,))

            value, vjp_fn = jax.vjp(jvp_wrapper, x_jax)
            cotangent = (jnp.ones_like(value[0]), jnp.ones_like(value[1]))
            grad = vjp_fn(cotangent)
            return value, grad[0]  # JAX returns tuple, extract single gradient

        return run_test_with_consistency_check(
            f"VJP(JVP) {operation.name}", nabla_fn, jax_fn
        )

    def test_16_jvp_jvp():
        """Test: jvp(jvp(f))"""
        x_nb, x_jax = get_test_data()
        tangent_nb, tangent_jax = get_tangent_for_jvp(x_nb, x_jax, operation)

        def nabla_fn():
            def first_jvp(x):
                return nb.jvp(f_nb, (x,), (tangent_nb,))

            return nb.jvp(first_jvp, (x_nb,), (tangent_nb,))

        def jax_fn():
            def first_jvp(x):
                return jax.jvp(f_jax, (x,), (tangent_jax,))

            return jax.jvp(first_jvp, (x_jax,), (tangent_jax,))

        return run_test_with_consistency_check(
            f"JVP(JVP) {operation.name}", nabla_fn, jax_fn
        )

    def test_17_vjp_vmap():
        """Test: vjp(vmap(f))"""
        x_nb, x_jax = get_test_data()
        vmapped_f_nb = nb.vmap(f_nb)
        vmapped_f_jax = jax.vmap(f_jax)

        def nabla_fn():
            value, vjp_fn = nb.vjp(vmapped_f_nb, x_nb)
            return value, vjp_fn(nb.ones_like(value))  # Nabla returns single value

        def jax_fn():
            value, vjp_fn = jax.vjp(vmapped_f_jax, x_jax)
            grad = vjp_fn(jnp.ones_like(value))
            return value, grad[0]  # JAX returns tuple, extract single gradient

        return run_test_with_consistency_check(
            f"VJP(VMAP) {operation.name}", nabla_fn, jax_fn
        )

    def test_18_jvp_vmap():
        """Test: jvp(vmap(f))"""
        x_nb, x_jax = get_test_data()
        tangent_nb, tangent_jax = get_tangent_for_jvp(x_nb, x_jax, operation)

        def nabla_fn():
            return nb.jvp(nb.vmap(f_nb), (x_nb,), (tangent_nb,))

        def jax_fn():
            return jax.jvp(jax.vmap(f_jax), (x_jax,), (tangent_jax,))

        return run_test_with_consistency_check(
            f"JVP(VMAP) {operation.name}", nabla_fn, jax_fn
        )

    def test_19_vmap_vmap():
        """Test: vmap(vmap(f))"""
        x_nb, x_jax = get_test_data()
        # vmap(vmap) requires at least a 2D input
        if len(x_nb.shape) < 2:
            pytest.skip("vmap(vmap) requires ndim >= 2")
        return run_test_with_consistency_check(
            f"VMAP(VMAP) {operation.name}",
            lambda: nb.vmap(nb.vmap(f_nb))(x_nb),
            lambda: jax.vmap(jax.vmap(f_jax))(x_jax),
        )

    def set_rank(rank):
        nonlocal current_rank
        current_rank = rank

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
    return test_functions, set_rank


# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================


def run_operation_tests(operation_name: str, all_ranks: bool = False):
    """Run all transformation tests for a specific unary operation."""
    if operation_name not in UNARY_OPERATIONS:
        print(f"Unknown operation: {operation_name}", file=sys.stderr)
        return False, 0, 0

    operation = UNARY_OPERATIONS[operation_name]
    print("=" * 80)
    print(f"TESTING UNARY OPERATION: {operation.name.upper()}")
    print("=" * 80)

    test_functions, set_rank_fn = create_unary_op_tests(operation)
    ranks_to_test = get_ranks_to_test() if all_ranks else [2]  # Default to rank 2

    total_passed = 0
    total_run = 0

    for rank in ranks_to_test:
        print(f"\n--- Testing Rank: {rank} ---")
        set_rank_fn(rank)
        passed_for_rank = 0
        for i, test_func in enumerate(test_functions):
            desc = (
                test_func.__doc__.split(":")[1].strip()
                if test_func.__doc__
                else test_func.__name__
            )
            print(f"  {i + 1:2d}. {desc:<25}", end="")

            try:
                success = test_func()
                if success:
                    passed_for_rank += 1
                total_run += 1
            except pytest.skip.Exception as e:
                print(f"SKIPPED ({e})")

        total_passed += passed_for_rank
        # total_run is implicitly len(test_functions) * len(ranks_to_test) excluding skips

    num_tests = len(test_functions) * len(ranks_to_test)
    print(f"\n{'=' * 80}")
    print(
        f"OPERATION {operation.name.upper()} RESULTS: {total_passed}/{total_run} tests passed"
    )
    return total_passed == total_run, total_passed, total_run


def run_all_operations(all_ranks: bool = False):
    """Run tests for all unary operations."""
    print("=" * 100)
    print("COMPREHENSIVE UNARY OPERATIONS TEST SUITE")
    print("=" * 100)

    overall_success = True
    total_passed_all, total_run_all = 0, 0

    for op_name in UNARY_OPERATIONS:
        success, passed, run = run_operation_tests(op_name, all_ranks)
        overall_success &= success
        total_passed_all += passed
        total_run_all += run
        cleanup_caches()

    print("\n" + "=" * 100)
    print("🏁 FINAL SUMMARY")
    print("=" * 100)
    success_rate = (total_passed_all / total_run_all * 100) if total_run_all > 0 else 0
    print(f"TOTAL TESTS PASSED: {total_passed_all}/{total_run_all}")
    print(f"OVERALL SUCCESS RATE: {success_rate:.1f}%")
    if overall_success:
        print("🎉 ALL UNARY OPERATIONS PASSED!")
    else:
        print("❌ SOME UNARY OPERATIONS FAILED")
    print("=" * 100)
    return overall_success


# ============================================================================
# PYTEST INTEGRATION
# ============================================================================


def pytest_generate_tests(metafunc):
    """Generate pytest parameters for all combinations."""
    if "operation_name" in metafunc.fixturenames:
        metafunc.parametrize("operation_name", list(UNARY_OPERATIONS.keys()))
    if "rank" in metafunc.fixturenames:
        metafunc.parametrize("rank", get_ranks_to_test())
    if "transformation_index" in metafunc.fixturenames:
        metafunc.parametrize("transformation_index", list(range(19)))


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup caches after each individual pytest test."""
    yield
    cleanup_caches()


class TestUnaryOperations:
    """Pytest test class for all unary operations."""

    def test_unary_operation_transformation(
        self, operation_name, rank, transformation_index
    ):
        """Test a specific unary operation with a specific rank and transformation."""
        operation = UNARY_OPERATIONS[operation_name]
        test_functions, set_rank_fn = create_unary_op_tests(operation)

        set_rank_fn(rank)
        test_func = test_functions[transformation_index]

        test_desc = (
            test_func.__doc__.split(":")[1].strip()
            if test_func.__doc__
            else test_func.__name__
        )

        try:
            success = test_func()
            assert success, f"Failed: {operation_name} - rank({rank}) - {test_desc}"
        except pytest.skip.Exception as e:
            pytest.skip(f"{operation_name} - rank({rank}) - {test_desc}: {e}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <operation_name|all> [--all-ranks]")
        print(f"Available operations: {list(UNARY_OPERATIONS.keys())}")
        sys.exit(1)

    op_arg = sys.argv[1]
    all_ranks_arg = "--all-ranks" in sys.argv

    if op_arg == "all":
        run_all_operations(all_ranks=all_ranks_arg)
    else:
        run_operation_tests(op_arg, all_ranks=all_ranks_arg)
