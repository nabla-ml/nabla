"""
COMPREHENSIVE NABLA REDUCTION OPERATIONS TEST SUITE
===================================================
... (docstring remains the same) ...
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

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
# REDUCTION OPERATION & CONFIGURATION DEFINITIONS
# ============================================================================


@dataclass
class ReductionOperation:
    """Definition of a reduction operation for testing."""

    name: str
    nabla_fn: Callable
    jax_fn: Callable
    description: str
    is_differentiable: bool = True


@dataclass
class ReductionConfig:
    """Defines a specific reduction configuration (axes, keep_dims)."""

    description: str
    axes: int | tuple[int, ...] | None
    keep_dims: bool
    min_rank: int  # Minimum tensor rank this config is valid for


# Define all reduction operations to test
REDUCTION_OPERATIONS = {
    "sum": ReductionOperation(
        "sum", nb.sum, jnp.sum, "Sum of elements over given axes."
    ),
    "mean": ReductionOperation(
        "mean", nb.mean, jnp.mean, "Mean of elements over given axes."
    ),
    "max": ReductionOperation(
        "max", nb.max, jnp.max, "Maximum of elements over given axes."
    ),
    "argmax": ReductionOperation(
        "argmax",
        nb.argmax,
        jnp.argmax,
        "Indices of maximum elements",
        is_differentiable=False,
    ),
}

# Define all reduction configurations to test
REDUCTION_CONFIGS = [
    # Rank 1+
    ReductionConfig("Reduce all, keep", axes=None, keep_dims=True, min_rank=1),
    ReductionConfig("Reduce all, no keep", axes=None, keep_dims=False, min_rank=1),
    ReductionConfig("Reduce axis 0, keep", axes=0, keep_dims=True, min_rank=1),
    ReductionConfig("Reduce axis 0, no keep", axes=0, keep_dims=False, min_rank=1),
    # Rank 2+
    ReductionConfig("Reduce axis 1, keep", axes=1, keep_dims=True, min_rank=2),
    ReductionConfig("Reduce axis 1, no keep", axes=1, keep_dims=False, min_rank=2),
    ReductionConfig("Reduce axes (0,1), keep", axes=(0, 1), keep_dims=True, min_rank=2),
    # Rank 3+
    ReductionConfig("Reduce axis 2, keep", axes=2, keep_dims=True, min_rank=3),
    ReductionConfig(
        "Reduce axes (0,2), no keep", axes=(0, 2), keep_dims=False, min_rank=3
    ),
]


def get_ranks_to_test() -> list[int]:
    """Get all ranks to test for reduction operations."""
    return [1, 2, 3]  # vector, matrix, 3D tensor


def get_test_data_for_rank(rank: int) -> tuple[nb.Array, jnp.ndarray]:
    """Get test data for a specific tensor rank."""
    shape = get_shape_for_rank(rank)
    x_nb = (nb.arange(shape) - 5).astype(nb.DType.float32)  # Include negative values
    x_jax = (jax_arange(shape) - 5).astype("float32")
    return x_nb, x_jax


# ============================================================================
# PARAMETERIZED TEST FRAMEWORK
# ============================================================================


def create_reduction_op_tests(operation: ReductionOperation):
    """Create all 19 transformation tests for a given reduction operation."""

    # These will be set by the test runner for each specific test case
    current_rank = 2  # Default rank
    current_config = REDUCTION_CONFIGS[0]  # Default config

    def get_test_data():
        return get_test_data_for_rank(current_rank)

    def get_configured_functions():
        axes, keep_dims = current_config.axes, current_config.keep_dims
        f_nb = partial(operation.nabla_fn, axes=axes, keep_dims=keep_dims)

        if operation.name == "argmax" and isinstance(axes, tuple):

            def jax_unsupported_fn(x):
                raise NotImplementedError(
                    "JAX/Nabla do not support tuple axes for argmax"
                )

            f_jax = jax_unsupported_fn
        else:
            f_jax = partial(operation.jax_fn, axis=axes, keepdims=keep_dims)

        return f_nb, f_jax

    def test_1_baseline():
        """Test: f(x)"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "Baseline", lambda: f_nb(x_nb), lambda: f_jax(x_jax)
        )

    def test_2_vjp():
        """Test: vjp(f)"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()

        def nabla_fn():
            value, vjp_fn = nb.vjp(f_nb, x_nb)
            return value, vjp_fn(nb.ones_like(value))

        def jax_fn():
            value, vjp_fn = jax.vjp(f_jax, x_jax)
            return value, vjp_fn(jnp.ones_like(value))[0]

        return run_test_with_consistency_check("VJP", nabla_fn, jax_fn)

    def test_3_jvp():
        """Test: jvp(f)"""
        x_nb, x_jax = get_test_data()
        tangent_nb, tangent_jax = nb.ones_like(x_nb), jnp.ones_like(x_jax)
        f_nb, f_jax = get_configured_functions()

        def nabla_fn():
            return nb.jvp(f_nb, (x_nb,), (tangent_nb,))

        def jax_fn():
            return jax.jvp(f_jax, (x_jax,), (tangent_jax,))

        return run_test_with_consistency_check("JVP", nabla_fn, jax_fn)

    def test_4_vmap():
        """Test: vmap(f)"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "VMAP", lambda: nb.vmap(f_nb)(x_nb), lambda: jax.vmap(f_jax)(x_jax)
        )

    def test_5_jit():
        """Test: jit(f)"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "JIT", lambda: nb.djit(f_nb)(x_nb), lambda: jax.jit(f_jax)(x_jax)
        )

    def test_6_jit_vjp():
        """Test: jit(vjp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()

        def vjp_wrapper(fn, x):
            value, vjp_fn = nb.vjp(fn, x)
            return value, vjp_fn(nb.ones_like(value))

        def vjp_wrapper_jax(fn, x):
            value, vjp_fn = jax.vjp(fn, x)
            return value, vjp_fn(jnp.ones_like(value))[0]

        return run_test_with_consistency_check(
            "JIT(VJP)",
            lambda: nb.djit(partial(vjp_wrapper, f_nb))(x_nb),
            lambda: jax.jit(partial(vjp_wrapper_jax, f_jax))(x_jax),
        )

    def test_7_jit_jvp():
        """Test: jit(jvp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        tangent_nb, tangent_jax = nb.ones_like(x_nb), jnp.ones_like(x_jax)

        def jvp_wrapper_nb(fn, x, t):
            return nb.jvp(fn, (x,), (t,))

        def jvp_wrapper_jax(fn, x, t):
            return jax.jvp(fn, (x,), (t,))

        return run_test_with_consistency_check(
            "JIT(JVP)",
            lambda: nb.djit(partial(jvp_wrapper_nb, f_nb))(x_nb, tangent_nb),
            lambda: jax.jit(partial(jvp_wrapper_jax, f_jax))(x_jax, tangent_jax),
        )

    def test_8_jit_vmap():
        """Test: jit(vmap(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "JIT(VMAP)",
            lambda: nb.djit(nb.vmap(f_nb))(x_nb),
            lambda: jax.jit(jax.vmap(f_jax))(x_jax),
        )

    def test_9_vmap_vjp():
        """Test: vmap(vjp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()

        def vjp_wrapper(fn, x):
            value, vjp_fn = nb.vjp(fn, x)
            return value, vjp_fn(nb.ones_like(value))

        def vjp_wrapper_jax(fn, x):
            value, vjp_fn = jax.vjp(fn, x)
            return value, vjp_fn(jnp.ones_like(value))[0]

        return run_test_with_consistency_check(
            "VMAP(VJP)",
            lambda: nb.vmap(partial(vjp_wrapper, f_nb))(x_nb),
            lambda: jax.vmap(partial(vjp_wrapper_jax, f_jax))(x_jax),
        )

    def test_10_vmap_jvp():
        """Test: vmap(jvp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()

        def jvp_wrapper_nb(fn, x):
            t = nb.ones_like(x)
            return nb.jvp(fn, (x,), (t,))

        def jvp_wrapper_jax(fn, x):
            t = jnp.ones_like(x)
            return jax.jvp(fn, (x,), (t,))

        return run_test_with_consistency_check(
            "VMAP(JVP)",
            lambda: nb.vmap(partial(jvp_wrapper_nb, f_nb))(x_nb),
            lambda: jax.vmap(partial(jvp_wrapper_jax, f_jax))(x_jax),
        )

    def test_11_jit_vmap_vjp():
        """Test: jit(vmap(vjp(f)))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()

        def vjp_wrapper(fn, x):
            value, vjp_fn = nb.vjp(fn, x)
            return value, vjp_fn(nb.ones_like(value))

        def vjp_wrapper_jax(fn, x):
            value, vjp_fn = jax.vjp(fn, x)
            return value, vjp_fn(jnp.ones_like(value))[0]

        vmapped_vjp_nb = nb.vmap(partial(vjp_wrapper, f_nb))
        vmapped_vjp_jax = jax.vmap(partial(vjp_wrapper_jax, f_jax))

        return run_test_with_consistency_check(
            "JIT(VMAP(VJP))",
            lambda: nb.djit(vmapped_vjp_nb)(x_nb),
            lambda: jax.jit(vmapped_vjp_jax)(x_jax),
        )

    def test_12_jit_vmap_jvp():
        """Test: jit(vmap(jvp(f)))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()

        def jvp_wrapper_nb(fn, x):
            return nb.jvp(fn, (x,), (nb.ones_like(x),))

        def jvp_wrapper_jax(fn, x):
            t = jnp.ones_like(x)
            return jax.jvp(fn, (x,), (t,))

        vmapped_jvp_nb = nb.vmap(partial(jvp_wrapper_nb, f_nb))
        vmapped_jvp_jax = jax.vmap(partial(jvp_wrapper_jax, f_jax))

        return run_test_with_consistency_check(
            "JIT(VMAP(JVP))",
            lambda: nb.djit(vmapped_jvp_nb)(x_nb),
            lambda: jax.jit(vmapped_jvp_jax)(x_jax),
        )

    def test_13_vjp_vjp():
        """Test: vjp(vjp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()

        def first_grad_nb(x):
            _, vjp_fn = nb.vjp(f_nb, x)
            return vjp_fn(nb.ones_like(f_nb(x)))

        def first_grad_jax(x):
            _, vjp_fn = jax.vjp(f_jax, x)
            return vjp_fn(jnp.ones_like(f_jax(x)))[0]

        def nabla_fn():
            value, vjp_fn_2 = nb.vjp(first_grad_nb, x_nb)
            return value, vjp_fn_2(nb.ones_like(value))

        def jax_fn():
            value, vjp_fn_2 = jax.vjp(first_grad_jax, x_jax)
            return value, vjp_fn_2(jnp.ones_like(value))[0]

        return run_test_with_consistency_check("VJP(VJP)", nabla_fn, jax_fn)

    def test_14_jvp_vjp():
        """Test: jvp(vjp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        tangent_nb, tangent_jax = nb.ones_like(x_nb), jnp.ones_like(x_jax)

        def first_grad_nb(x):
            _, vjp_fn = nb.vjp(f_nb, x)
            return vjp_fn(nb.ones_like(f_nb(x)))

        def first_grad_jax(x):
            _, vjp_fn = jax.vjp(f_jax, x)
            return vjp_fn(jnp.ones_like(f_jax(x)))[0]

        return run_test_with_consistency_check(
            "JVP(VJP)",
            lambda: nb.jvp(first_grad_nb, (x_nb,), (tangent_nb,)),
            lambda: jax.jvp(first_grad_jax, (x_jax,), (tangent_jax,)),
        )

    def test_15_vjp_jvp():
        """Test: vjp(jvp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        tangent_nb, tangent_jax = nb.ones_like(x_nb), jnp.ones_like(x_jax)

        def jvp_wrapper_nb(x):
            return nb.jvp(f_nb, (x,), (tangent_nb,))

        def jvp_wrapper_jax(x):
            return jax.jvp(f_jax, (x,), (tangent_jax,))

        def nabla_fn():
            val, vjp_fn = nb.vjp(jvp_wrapper_nb, x_nb)
            cotan = (nb.ones_like(val[0]), nb.ones_like(val[1]))
            return val, vjp_fn(cotan)

        def jax_fn():
            val, vjp_fn = jax.vjp(jvp_wrapper_jax, x_jax)
            cotan = (jnp.ones_like(val[0]), jnp.ones_like(val[1]))
            return val, vjp_fn(cotan)[0]

        return run_test_with_consistency_check("VJP(JVP)", nabla_fn, jax_fn)

    def test_16_jvp_jvp():
        """Test: jvp(jvp(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        tangent_nb, tangent_jax = nb.ones_like(x_nb), jnp.ones_like(x_jax)

        def jvp_wrapper_nb(x):
            return nb.jvp(f_nb, (x,), (tangent_nb,))

        def jvp_wrapper_jax(x):
            return jax.jvp(f_jax, (x,), (tangent_jax,))

        return run_test_with_consistency_check(
            "JVP(JVP)",
            lambda: nb.jvp(jvp_wrapper_nb, (x_nb,), (tangent_nb,)),
            lambda: jax.jvp(jvp_wrapper_jax, (x_jax,), (tangent_jax,)),
        )

    def test_17_vjp_vmap():
        """Test: vjp(vmap(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        vmapped_f_nb, vmapped_f_jax = nb.vmap(f_nb), jax.vmap(f_jax)

        def nabla_fn():
            val, vjp_fn = nb.vjp(vmapped_f_nb, x_nb)
            return val, vjp_fn(nb.ones_like(val))

        def jax_fn():
            val, vjp_fn = jax.vjp(vmapped_f_jax, x_jax)
            return val, vjp_fn(jnp.ones_like(val))[0]

        return run_test_with_consistency_check("VJP(VMAP)", nabla_fn, jax_fn)

    def test_18_jvp_vmap():
        """Test: jvp(vmap(f))"""
        x_nb, x_jax = get_test_data()
        f_nb, f_jax = get_configured_functions()
        tangent_nb, tangent_jax = nb.ones_like(x_nb), jnp.ones_like(x_jax)
        vmapped_f_nb = nb.vmap(f_nb)
        vmapped_f_jax = jax.vmap(f_jax)

        def nabla_fn():
            return nb.jvp(vmapped_f_nb, (x_nb,), (tangent_nb,))

        def jax_fn():
            return jax.jvp(vmapped_f_jax, (x_jax,), (tangent_jax,))

        return run_test_with_consistency_check("JVP(VMAP)", nabla_fn, jax_fn)

    def test_19_vmap_vmap():
        """Test: vmap(vmap(f))"""
        x_nb, x_jax = get_test_data()
        if len(x_nb.shape) < 2:
            pytest.skip("vmap(vmap) requires ndim >= 2")

        f_nb, f_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "VMAP(VMAP)",
            lambda: nb.vmap(nb.vmap(f_nb))(x_nb),
            lambda: jax.vmap(jax.vmap(f_jax))(x_jax),
        )

    def set_rank_and_config(rank, config):
        nonlocal current_rank, current_config
        current_rank = rank
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
    return test_functions, set_rank_and_config


# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================


def run_operation_tests(operation_name: str, all_configs: bool = False):
    """Run all transformation tests for a specific reduction operation."""
    if operation_name not in REDUCTION_OPERATIONS:
        print(f"Unknown operation: {operation_name}", file=sys.stderr)
        return False, 0, 0

    operation = REDUCTION_OPERATIONS[operation_name]
    print("=" * 80)
    print(f"TESTING REDUCTION OPERATION: {operation.name.upper()}")
    print("=" * 80)

    test_functions, set_rank_and_config_fn = create_reduction_op_tests(operation)
    ranks_to_test = get_ranks_to_test() if all_configs else [2]

    total_passed, total_run = 0, 0
    failed_tests = []

    for rank in ranks_to_test:
        print(f"\n--- Testing Rank: {rank} ---")
        valid_configs = [c for c in REDUCTION_CONFIGS if rank >= c.min_rank]
        if not all_configs:
            # For single run, pick a representative config.
            valid_configs = [c for c in valid_configs if c.axes == 0 and c.keep_dims][
                :1
            ]

        for config in valid_configs:
            print(f"\n  > Config: {config.description}")
            set_rank_and_config_fn(rank, config)

            for i, test_func in enumerate(test_functions):
                desc = test_func.__doc__.split(":")[1].strip()
                print(f"    {i + 1:2d}. {desc:<15}", end="")

                try:
                    success = test_func()
                    # A test that completes is always counted as "run".
                    total_run += 1
                    if success:
                        total_passed += 1
                    else:
                        failed_tests.append(
                            f"Rank {rank}, Config '{config.description}', Test '{desc}'"
                        )

                except pytest.skip.Exception as e:
                    # This test was not run, so we DO NOT increment total_run.
                    print(f"SKIPPED ({e})")

                except Exception as e:
                    # An errored test was "run" but failed.
                    # We print the exception type for better debugging.
                    print(f"ERROR ({type(e).__name__}: {e})")
                    total_run += 1
                    failed_tests.append(
                        f"Rank {rank}, Config '{config.description}', Test '{desc}' (ERROR)"
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


def run_all_operations(all_configs: bool = False):
    """Run tests for all reduction operations."""
    print("=" * 100)
    print("COMPREHENSIVE REDUCTION OPERATIONS TEST SUITE")
    print("=" * 100)

    overall_success = True
    total_passed_all, total_run_all = 0, 0

    for op_name in REDUCTION_OPERATIONS:
        success, passed, run = run_operation_tests(op_name, all_configs)
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
        print("🎉 ALL REDUCTION OPERATIONS PASSED!")
    else:
        print("❌ SOME REDUCTION OPERATIONS FAILED")
    print("=" * 100)
    return overall_success


# ============================================================================
# PYTEST INTEGRATION
# ============================================================================
@pytest.fixture(scope="module", autouse=True)
def manage_caches():
    """Fixture to clean up caches before and after the test module."""
    cleanup_caches()
    yield
    cleanup_caches()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Fixture to clean up caches after each individual test function."""
    yield
    cleanup_caches()


TRANSFORMATION_INDICES = list(range(19))
CONFIG_INDICES = list(range(len(REDUCTION_CONFIGS)))
RANKS = get_ranks_to_test()
OPERATION_NAMES = list(REDUCTION_OPERATIONS.keys())


@pytest.mark.parametrize("operation_name", OPERATION_NAMES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("config_index", CONFIG_INDICES)
@pytest.mark.parametrize("transformation_index", TRANSFORMATION_INDICES)
class TestReductionOperations:
    """Pytest test class for all reduction operations."""

    def test_reduction_operation_transformation(
        self, operation_name, rank, config_index, transformation_index
    ):
        """Test a specific reduction op with a specific rank, config, and transform."""
        operation = REDUCTION_OPERATIONS[operation_name]
        config = REDUCTION_CONFIGS[config_index]

        if rank < config.min_rank:
            pytest.skip(
                f"Config '{config.description}' requires rank >= {config.min_rank}, but got rank {rank}"
            )

        test_functions, set_rank_and_config_fn = create_reduction_op_tests(operation)
        set_rank_and_config_fn(rank, config)
        test_func = test_functions[transformation_index]
        test_desc = test_func.__doc__.split(":")[1].strip()

        try:
            success = test_func()
            assert success, (
                f"Failed: {operation_name} - rank({rank}) - "
                f"config('{config.description}') - transform({test_desc})"
            )
        except pytest.skip.Exception as e:
            pytest.skip(f"SKIPPED: {operation_name} - rank({rank}): {e}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <operation_name|all> [--all-configs]")
        print(f"Available operations: {list(REDUCTION_OPERATIONS.keys())}")
        sys.exit(1)

    op_arg = sys.argv[1]
    all_configs_arg = "--all-configs" in sys.argv

    if op_arg == "all":
        run_all_operations(all_configs=all_configs_arg)
    else:
        run_operation_tests(op_arg, all_configs=all_configs_arg)
