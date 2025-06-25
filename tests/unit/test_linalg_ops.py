# test_linalg_ops.py

"""
COMPREHENSIVE NABLA LINEAR ALGEBRA OPERATIONS TEST SUITE
=========================================================
This test suite validates core linear algebra operations in Nabla against JAX as a
ground truth. It applies a rigorous testing methodology, covering 19 function
transformations (vjp, jvp, vmap, jit, and their compositions) across various
operation-specific configurations.

The framework is designed to be modular and can be extended to handle complex
operations with numerous keyword arguments.

LINEAR ALGEBRA OPERATIONS TESTED:
──────────────────────────────────
- matmul

(NOTE: conv2d and conv2d_transpose tests are defined but commented out
 as the operations are not yet implemented in Nabla. The framework is
 ready for their inclusion.)

TRANSFORMATION COMBINATIONS TESTED (19 total):
──────────────────────────────────────────────
(Same 19 combinations as the unary/binary/reduction suites)
Level 0: f(x, y)
Level 1: vjp, jvp, vmap, jit
Level 2: jit(vjp), jit(jvp), jit(vmap), vmap(vjp), vmap(jvp)
Level 3: jit(vmap(vjp)), jit(vmap(jvp))
Level 4: vjp(vjp), jvp(vjp), vjp(jvp), jvp(jvp)
Level 5: vjp(vmap), jvp(vmap), vmap(vmap)

CONSISTENCY CHECKING LOGIC:
─────────────────────────
Each test runs both the Nabla and JAX versions of an operation.
✅ PASS if both succeed and results match, or if both fail consistently.
❌ FAIL if one fails and the other succeeds, or if results differ.

This ensures Nabla's behavior, including its differentiation rules and JIT
compilation, is consistent with the industry-standard JAX.

USAGE:
──────
Run all linalg op tests:
    pytest test_linalg_ops.py
    python test_linalg_ops.py all

Run a specific operation:
    pytest test_linalg_ops.py -k "matmul"
    python test_linalg_ops.py matmul
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb

# Import utility modules
try:
    from .test_utils import (
        cleanup_caches,
        run_test_with_consistency_check,
    )
except ImportError:
    # Fallback for direct execution
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from test_utils import (
        cleanup_caches,
        run_test_with_consistency_check,
    )

# ============================================================================
# JAX HELPERS FOR NABLA-EQUIVALENT OPERATIONS
# ============================================================================


def jax_matmul_wrapper(x, y):
    """JAX wrapper to mimic Nabla's matmul which implicitly handles 1D vectors."""
    # Nabla's matmul reshapes 1D vectors to 2D before the operation.
    # We replicate this to ensure the ground truth is identical.
    x_rank, y_rank = len(x.shape), len(y.shape)
    x_jax, y_jax = x, y

    squeeze_axes = []
    if x_rank == 1:
        x_jax = jnp.expand_dims(x, 0)
        squeeze_axes.append(-2)
    if y_rank == 1:
        y_jax = jnp.expand_dims(y, 1)
        squeeze_axes.append(-1)

    res = jnp.matmul(x_jax, y_jax)

    if squeeze_axes:
        # Squeeze the dimensions that were added
        res = jnp.squeeze(res, axis=tuple(squeeze_axes))
    return res


# ============================================================================
# LINALG OPERATION & CONFIGURATION DEFINITIONS
# ============================================================================


@dataclass
class LinAlgConfig:
    """Defines a specific configuration for a linear algebra operation."""

    description: str
    # Shapes of the primals (e.g., input and filter)
    primal_shapes: tuple[tuple[int, ...], ...]
    # Keyword arguments for the operation
    params: dict


@dataclass
class LinAlgOperation:
    """Definition of a linear algebra operation for testing."""

    name: str
    nabla_fn: Callable
    jax_fn: Callable
    description: str
    configs: list[LinAlgConfig]


def get_test_data_for_shapes(
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[nb.Array, ...], tuple[jnp.ndarray, ...]]:
    """Get test data (Nabla and JAX) for a set of primal shapes."""
    nabla_primals = []
    jax_primals = []
    for i, shape in enumerate(shapes):
        # Use different constants to avoid trivial gradients
        # CORRECTED: nb.arange takes a shape tuple directly.
        nabla_primals.append((nb.arange(shape) + i + 1).astype(nb.DType.float32) * 0.1)
        # JAX requires the number of elements, then a reshape. This was already correct.
        jax_primals.append(
            (jnp.arange(np.prod(shape)).reshape(shape) + i + 1).astype("float32") * 0.1
        )
    return tuple(nabla_primals), tuple(jax_primals)


def get_linalg_op_args(
    config: LinAlgConfig,
) -> tuple[tuple[tuple, dict], tuple[tuple, dict]]:
    """Helper to generate full (primals, kwargs) tuples for Nabla and JAX."""
    primals_nb, primals_jax = get_test_data_for_shapes(config.primal_shapes)
    return ((primals_nb, config.params), (primals_jax, config.params))


# Define all linear algebra operations to test
LINALG_OPERATIONS = {
    "matmul": LinAlgOperation(
        name="matmul",
        nabla_fn=nb.matmul,
        jax_fn=jax_matmul_wrapper,  # Wrapper ensures 1D vectors are handled like Nabla
        description="Matrix multiplication.",
        configs=[
            LinAlgConfig("Vector @ Vector", ((4,), (4,)), {}),
            LinAlgConfig("Matrix @ Vector", ((3, 4), (4,)), {}),
            LinAlgConfig("Vector @ Matrix", ((4,), (4, 5)), {}),
            LinAlgConfig("Matrix @ Matrix", ((3, 4), (4, 5)), {}),
            LinAlgConfig("Batched Matmul", ((10, 3, 4), (10, 4, 5)), {}),
        ],
    ),
    # The framework is ready for more operations like conv2d once implemented.
    # "conv2d": LinAlgOperation(...),
    # "conv2d_transpose": LinAlgOperation(...),
}


# ============================================================================
# PARAMETERIZED TEST FRAMEWORK (Adapted from test_view_ops.py)
# ============================================================================


def create_op_tests(operation: LinAlgOperation):
    """Create all 19 transformation tests for a given linear algebra operation."""
    # These will be updated by the test runner for each configuration.
    current_config = operation.configs[0]

    def get_configured_functions():
        """
        Core helper that uses `partial` to create clean, tensor-only functions.
        This adapts functions with keyword arguments to the f(*primals) style
        needed by the test harness.
        """
        (primals_nb, params_nb), (primals_jax, params_jax) = get_linalg_op_args(
            current_config
        )

        f_nb = partial(operation.nabla_fn, **params_nb)
        f_jax = partial(operation.jax_fn, **params_jax)

        return f_nb, primals_nb, f_jax, primals_jax

    # The following 19 test functions are adapted from test_binary_ops.py to handle
    # multiple primals (input, filter) and their gradients correctly.

    def test_1_baseline():
        """Test: f(x, y)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "Baseline", lambda: f_nb(*primals_nb), lambda: f_jax(*primals_jax)
        )

    def test_2_vjp():
        """Test: vjp(f)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def nabla_fn():
            val, vjp_fn = nb.vjp(f_nb, *primals_nb)
            grad = vjp_fn(nb.ones_like(val))
            return val, grad

        def jax_fn():
            val, vjp_fn = jax.vjp(f_jax, *primals_jax)
            grad = vjp_fn(jnp.ones_like(val))
            return val, grad

        return run_test_with_consistency_check("VJP", nabla_fn, jax_fn)

    def test_3_jvp():
        """Test: jvp(f)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb = tuple(nb.ones_like(p) for p in primals_nb)
        tangents_jax = tuple(jnp.ones_like(p) for p in primals_jax)
        return run_test_with_consistency_check(
            "JVP",
            lambda: nb.jvp(f_nb, primals_nb, tangents_nb),
            lambda: jax.jvp(f_jax, primals_jax, tangents_jax),
        )

    def test_4_vmap():
        """Test: vmap(f)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "VMAP",
            lambda: nb.vmap(f_nb)(*primals_nb),
            lambda: jax.vmap(f_jax)(*primals_jax),
        )

    def test_5_jit():
        """Test: jit(f)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "JIT",
            lambda: nb.djit(f_nb)(*primals_nb),
            lambda: jax.jit(f_jax)(*primals_jax),
        )

    def test_6_jit_vjp():
        """Test: jit(vjp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def vjp_wrapper_nb(*args):
            val, vjp_fn = nb.vjp(f_nb, *args)
            return val, vjp_fn(nb.ones_like(val))

        def vjp_wrapper_jax(*args):
            val, vjp_fn = jax.vjp(f_jax, *args)
            return val, vjp_fn(jnp.ones_like(val))

        return run_test_with_consistency_check(
            "JIT(VJP)",
            lambda: nb.djit(vjp_wrapper_nb)(*primals_nb),
            lambda: jax.jit(vjp_wrapper_jax)(*primals_jax),
        )

    def test_7_jit_jvp():
        """Test: jit(jvp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb = tuple(nb.ones_like(p) for p in primals_nb)
        tangents_jax = tuple(jnp.ones_like(p) for p in primals_jax)
        jitted_jvp_nb = nb.djit(lambda p, t: nb.jvp(f_nb, p, t))
        jitted_jvp_jax = jax.jit(lambda p, t: jax.jvp(f_jax, p, t))
        return run_test_with_consistency_check(
            "JIT(JVP)",
            lambda: jitted_jvp_nb(primals_nb, tangents_nb),
            lambda: jitted_jvp_jax(primals_jax, tangents_jax),
        )

    def test_8_jit_vmap():
        """Test: jit(vmap(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "JIT(VMAP)",
            lambda: nb.djit(nb.vmap(f_nb))(*primals_nb),
            lambda: jax.jit(jax.vmap(f_jax))(*primals_jax),
        )

    def test_9_vmap_vjp():
        """Test: vmap(vjp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def vjp_wrapper_nb(*args):
            val, vjp_fn = nb.vjp(f_nb, *args)
            return val, vjp_fn(nb.ones_like(val))

        def vjp_wrapper_jax(*args):
            val, vjp_fn = jax.vjp(f_jax, *args)
            return val, vjp_fn(jnp.ones_like(val))

        return run_test_with_consistency_check(
            "VMAP(VJP)",
            lambda: nb.vmap(vjp_wrapper_nb)(*primals_nb),
            lambda: jax.vmap(vjp_wrapper_jax)(*primals_jax),
        )

    def test_10_vmap_jvp():
        """Test: vmap(jvp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def jvp_wrapper_nb(*args):
            tangents = tuple(nb.ones_like(p) for p in args)
            return nb.jvp(f_nb, args, tangents)

        def jvp_wrapper_jax(*args):
            tangents = tuple(jnp.ones_like(p) for p in args)
            return jax.jvp(f_jax, args, tangents)

        return run_test_with_consistency_check(
            "VMAP(JVP)",
            lambda: nb.vmap(jvp_wrapper_nb)(*primals_nb),
            lambda: jax.vmap(jvp_wrapper_jax)(*primals_jax),
        )

    def test_11_jit_vmap_vjp():
        """Test: jit(vmap(vjp(f)))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def vjp_wrapper_nb(*args):
            val, vjp_fn = nb.vjp(f_nb, *args)
            return val, vjp_fn(nb.ones_like(val))

        def vjp_wrapper_jax(*args):
            val, vjp_fn = jax.vjp(f_jax, *args)
            return val, vjp_fn(jnp.ones_like(val))

        return run_test_with_consistency_check(
            "JIT(VMAP(VJP))",
            lambda: nb.djit(nb.vmap(vjp_wrapper_nb))(*primals_nb),
            lambda: jax.jit(jax.vmap(vjp_wrapper_jax))(*primals_jax),
        )

    def test_12_jit_vmap_jvp():
        """Test: jit(vmap(jvp(f)))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def jvp_wrapper_nb(*args):
            tangents = tuple(nb.ones_like(p) for p in args)
            return nb.jvp(f_nb, args, tangents)

        def jvp_wrapper_jax(*args):
            tangents = tuple(jnp.ones_like(p) for p in args)
            return jax.jvp(f_jax, args, tangents)

        return run_test_with_consistency_check(
            "JIT(VMAP(JVP))",
            lambda: nb.djit(nb.vmap(jvp_wrapper_nb))(*primals_nb),
            lambda: jax.jit(jax.vmap(jvp_wrapper_jax))(*primals_jax),
        )

    def test_13_vjp_vjp():
        """Test: vjp(vjp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def first_grad_nb(*args):
            _val, vjp_fn = nb.vjp(f_nb, *args)
            return vjp_fn(nb.ones_like(f_nb(*args)))

        def first_grad_jax(*args):
            _val, vjp_fn = jax.vjp(f_jax, *args)
            return vjp_fn(jnp.ones_like(f_jax(*args)))

        def nabla_fn():
            val, vjp_fn_2 = nb.vjp(first_grad_nb, *primals_nb)
            cotan_ones = tuple(nb.ones_like(c) for c in val)
            return val, vjp_fn_2(cotan_ones)

        def jax_fn():
            val, vjp_fn_2 = jax.vjp(first_grad_jax, *primals_jax)
            cotan_ones = tuple(jnp.ones_like(c) for c in val)
            return val, vjp_fn_2(cotan_ones)

        return run_test_with_consistency_check("VJP(VJP)", nabla_fn, jax_fn)

    def test_14_jvp_vjp():
        """Test: jvp(vjp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb = tuple(nb.ones_like(p) for p in primals_nb)
        tangents_jax = tuple(jnp.ones_like(p) for p in primals_jax)

        def first_grad_nb(*args):
            _val, vjp_fn = nb.vjp(f_nb, *args)
            return vjp_fn(nb.ones_like(f_nb(*args)))

        def first_grad_jax(*args):
            _val, vjp_fn = jax.vjp(f_jax, *args)
            return vjp_fn(jnp.ones_like(f_jax(*args)))

        return run_test_with_consistency_check(
            "JVP(VJP)",
            lambda: nb.jvp(first_grad_nb, primals_nb, tangents_nb),
            lambda: jax.jvp(first_grad_jax, primals_jax, tangents_jax),
        )

    def test_15_vjp_jvp():
        """Test: vjp(jvp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb = tuple(nb.ones_like(p) for p in primals_nb)
        tangents_jax = tuple(jnp.ones_like(p) for p in primals_jax)

        def jvp_wrapper_nb(*args):
            return nb.jvp(f_nb, args, tangents_nb)

        def jvp_wrapper_jax(*args):
            return jax.jvp(f_jax, args, tangents_jax)

        def nabla_fn():
            val, vjp_fn = nb.vjp(jvp_wrapper_nb, *primals_nb)
            cotan = (nb.ones_like(val[0]), nb.ones_like(val[1]))
            return val, vjp_fn(cotan)

        def jax_fn():
            val, vjp_fn = jax.vjp(jvp_wrapper_jax, *primals_jax)
            cotan = (jnp.ones_like(val[0]), jnp.ones_like(val[1]))
            return val, vjp_fn(cotan)

        return run_test_with_consistency_check("VJP(JVP)", nabla_fn, jax_fn)

    def test_16_jvp_jvp():
        """Test: jvp(jvp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb = tuple(nb.ones_like(p) for p in primals_nb)
        tangents_jax = tuple(jnp.ones_like(p) for p in primals_jax)

        def jvp_wrapper_nb(*args):
            return nb.jvp(f_nb, args, tangents_nb)

        def jvp_wrapper_jax(*args):
            return jax.jvp(f_jax, args, tangents_jax)

        return run_test_with_consistency_check(
            "JVP(JVP)",
            lambda: nb.jvp(jvp_wrapper_nb, primals_nb, tangents_nb),
            lambda: jax.jvp(jvp_wrapper_jax, primals_jax, tangents_jax),
        )

    def test_17_vjp_vmap():
        """Test: vjp(vmap(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        vmapped_f_nb, vmapped_f_jax = nb.vmap(f_nb), jax.vmap(f_jax)

        def nabla_fn():
            val, vjp_fn = nb.vjp(vmapped_f_nb, *primals_nb)
            return val, vjp_fn(nb.ones_like(val))

        def jax_fn():
            val, vjp_fn = jax.vjp(vmapped_f_jax, *primals_jax)
            return val, vjp_fn(jnp.ones_like(val))

        return run_test_with_consistency_check("VJP(VMAP)", nabla_fn, jax_fn)

    def test_18_jvp_vmap():
        """Test: jvp(vmap(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb = tuple(nb.ones_like(p) for p in primals_nb)
        tangents_jax = tuple(jnp.ones_like(p) for p in primals_jax)
        vmapped_f_nb, vmapped_f_jax = nb.vmap(f_nb), jax.vmap(f_jax)
        return run_test_with_consistency_check(
            "JVP(VMAP)",
            lambda: nb.jvp(vmapped_f_nb, primals_nb, tangents_nb),
            lambda: jax.jvp(vmapped_f_jax, primals_jax, tangents_jax),
        )

    def test_19_vmap_vmap():
        """Test: vmap(vmap(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        for p in primals_nb:
            if len(p.shape) < 2:
                pytest.skip("vmap(vmap) requires ndim >= 2 for all primals")
        return run_test_with_consistency_check(
            "VMAP(VMAP)",
            lambda: nb.vmap(nb.vmap(f_nb))(*primals_nb),
            lambda: jax.vmap(jax.vmap(f_jax))(*primals_jax),
        )

    def set_config(config):
        nonlocal current_config
        current_config = config

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
    return test_functions, set_config


# ============================================================================
# MAIN EXECUTION LOGIC & PYTEST INTEGRATION
# ============================================================================


def run_operation_tests(operation_name: str):
    """Run all transformation tests for a specific linalg operation."""
    if operation_name not in LINALG_OPERATIONS:
        print(f"Unknown operation: {operation_name}", file=sys.stderr)
        return False, 0, 0

    operation = LINALG_OPERATIONS[operation_name]
    print("=" * 80)
    print(f"TESTING LINALG OPERATION: {operation.name.upper()}")
    print("=" * 80)

    test_functions, set_config_fn = create_op_tests(operation)
    total_passed, total_run = 0, 0
    failed_tests = []

    for config in operation.configs:
        print(f"\n  > Config: {config.description}")
        set_config_fn(config)

        for i, test_func in enumerate(test_functions):
            desc = test_func.__doc__.split(":")[1].strip()
            print(f"    {i + 1:2d}. {desc:<15}", end="")

            try:
                success = test_func()
                total_run += 1
                if success:
                    total_passed += 1
                else:
                    failed_tests.append(f"Config '{config.description}', Test '{desc}'")
            except pytest.skip.Exception as e:
                print(f"SKIPPED ({e})")
            except Exception as e:
                print(f"ERROR ({type(e).__name__})")
                total_run += 1
                failed_tests.append(
                    f"Config '{config.description}', Test '{desc}' (ERROR: {e})"
                )
            finally:
                cleanup_caches()

    print(f"\n{'=' * 80}")
    print(
        f"OPERATION {operation.name.upper()} RESULTS: {total_passed}/{total_run} tests passed"
    )
    if failed_tests:
        print("--- FAILED TESTS ---")
        for failed in failed_tests:
            print(f"  - {failed}")

    return total_passed == total_run and total_run > 0, total_passed, total_run


def run_all_operations():
    """Run tests for all linear algebra operations."""
    print("=" * 100)
    print("COMPREHENSIVE LINEAR ALGEBRA OPERATIONS TEST SUITE")
    print("=" * 100)
    overall_success = True
    total_passed_all, total_run_all = 0, 0

    for op_name in LINALG_OPERATIONS:
        success, passed, run = run_operation_tests(op_name)
        overall_success &= success
        total_passed_all += passed
        total_run_all += run
        cleanup_caches()

    print("\n" + "=" * 100)
    print("🏁 FINAL SUMMARY")
    print("=" * 100)
    rate = (total_passed_all / total_run_all * 100) if total_run_all > 0 else 0
    print(f"TOTAL TESTS PASSED: {total_passed_all}/{total_run_all}")
    print(f"OVERALL SUCCESS RATE: {rate:.1f}%")
    if overall_success:
        print("🎉 ALL LINALG OPERATIONS PASSED!")
    else:
        print("❌ SOME LINALG OPERATIONS FAILED")
    print("=" * 100)
    return overall_success


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Fixture to clean up caches after each individual test function."""
    yield
    cleanup_caches()


# Create a flat list of all test combinations for pytest parametrization
ALL_TEST_CASES = []
for op_name, op_def in LINALG_OPERATIONS.items():
    for config in op_def.configs:
        for trans_idx in range(19):
            test_id = f"{op_name}-cfg_{config.description.replace(' ', '_').replace('@', '_at_')}-trans{trans_idx}"
            ALL_TEST_CASES.append(pytest.param(op_name, config, trans_idx, id=test_id))


@pytest.mark.parametrize("operation_name, config, transformation_index", ALL_TEST_CASES)
def test_linalg_operation_transformation(operation_name, config, transformation_index):
    """Pytest entry point for a specific linalg op/config/transform."""
    operation = LINALG_OPERATIONS[operation_name]
    test_functions, set_config_fn = create_op_tests(operation)
    set_config_fn(config)
    test_func = test_functions[transformation_index]
    test_desc = test_func.__doc__.split(":")[1].strip()

    try:
        success = test_func()
        assert success, (
            f"Failed: {operation_name} - config('{config.description}') - transform({test_desc})"
        )
    except pytest.skip.Exception as e:
        pytest.skip(
            f"SKIPPED: {operation_name} - config('{config.description}') - transform({test_desc}): {e}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <operation_name|all>")
        print(f"Available operations: {list(LINALG_OPERATIONS.keys())}")
        sys.exit(1)

    op_arg = sys.argv[1]
    if op_arg == "all":
        run_all_operations()
    else:
        run_operation_tests(op_arg)
