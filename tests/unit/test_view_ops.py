"""
COMPREHENSIVE NABLA VIEW OPERATIONS TEST SUITE
==============================================
This test suite validates ALL view and shape manipulation operations in Nabla against
JAX as a ground truth. It applies a rigorous testing methodology, covering 19
function transformations (vjp, jvp, vmap, jit, and their compositions) across
various tensor ranks and operation-specific configurations.

The framework is designed to handle the complexity of view operations, which often
have diverse function signatures (e.g., multiple keyword arguments, list inputs).
It uses a flexible configuration system to test each operation under meaningful
scenarios.

VIEW OPERATIONS TESTED:
───────────────────────
- transpose, permute, reshape, broadcast_to, squeeze, unsqueeze,
- array_slice, pad, concatenate, stack, move_axis_to_front

TRANSFORMATION COMBINATIONS TESTED (19 total):
──────────────────────────────────────────────
(Same 19 combinations as the unary/binary/reduction suites)
Level 0: f(x)
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
Run all view op tests:
    pytest test_view_ops.py
    python test_view_ops.py all

Run a specific operation:
    pytest test_view_ops.py -k "transpose"
    python test_view_ops.py reshape
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
# JAX HELPERS FOR NABLA-EQUIVALENT OPERATIONS
# ============================================================================


def jax_transpose_wrapper(x, axis_1, axis_2):
    """JAX wrapper to mimic Nabla's transpose(x, axis_1, axis_2)."""
    rank = len(x.shape)
    if rank < 2:
        return x  # Transpose is a no-op for scalars and vectors
    axes = list(range(rank))
    axis_1_pos = axis_1 if axis_1 >= 0 else rank + axis_1
    axis_2_pos = axis_2 if axis_2 >= 0 else rank + axis_2
    axes[axis_1_pos], axes[axis_2_pos] = axes[axis_2_pos], axes[axis_1_pos]
    return jnp.transpose(x, axes=axes)


def jax_pad_inverse_slice(arr, slices, target_shape):
    """JAX equivalent of Nabla's pad, which is an inverse slice."""
    return jnp.zeros(target_shape, dtype=arr.dtype).at[tuple(slices)].set(arr)


def jax_unsqueeze_wrapper(x, axes):
    """Wrapper to make jnp.expand_dims handle a list of axes like Nabla."""
    res = x
    # JAX requires axes to be sorted for sequential expansion
    for axis in sorted(axes):
        res = jnp.expand_dims(res, axis=axis)
    return res


def jax_squeeze_wrapper(x, axes):
    """Wrapper to handle the `axes` keyword for jnp.squeeze."""
    return jnp.squeeze(x, axis=tuple(axes))


def jax_slice_wrapper(x, slices):
    """Wrapper to handle slice arguments for JAX functions."""
    return x[tuple(slices)]


def jax_move_axis_wrapper(x, source, destination):
    """JAX equivalent of np.moveaxis."""
    return jnp.moveaxis(x, source, destination)


# ============================================================================
# VIEW OPERATION & CONFIGURATION DEFINITIONS
# ============================================================================


@dataclass
class ViewConfig:
    """Defines a specific configuration for a view operation."""

    description: str
    params: dict
    min_rank: int
    shape_generator: Callable[[int], tuple[int, ...]] | None = None


@dataclass
class ViewOperation:
    """Definition of a view operation for testing."""

    name: str
    nabla_fn: Callable
    jax_fn: Callable
    get_args: Callable[[int, ViewConfig], tuple[tuple[tuple, dict], tuple[tuple, dict]]]
    description: str
    configs: list[ViewConfig]
    is_differentiable: bool = True
    is_list_input: bool = False


def get_ranks_to_test() -> list[int]:
    """Get all ranks to test for view operations."""
    return [1, 2, 3]


def get_test_data_for_rank(
    rank: int, shape_generator: Callable | None = None
) -> tuple[nb.Array, jnp.ndarray]:
    """Get test data for a specific tensor rank."""
    shape = shape_generator(rank) if shape_generator else get_shape_for_rank(rank)
    x_nb = (nb.ndarange(shape) + 1).astype(nb.DType.float32)
    x_jax = (jax_arange(shape) + 1).astype("float32")
    return x_nb, x_jax


# Define all view operations to test
VIEW_OPERATIONS = {
    "transpose": ViewOperation(
        name="transpose",
        nabla_fn=nb.transpose,
        jax_fn=jax_transpose_wrapper,
        get_args=lambda r, c: (
            ((get_test_data_for_rank(r)[0],), c.params),
            ((get_test_data_for_rank(r)[1],), c.params),
        ),
        description="Transpose two axes of a tensor.",
        configs=[
            ViewConfig(
                "Swap last two axes (2D)", {"axis_1": -2, "axis_2": -1}, min_rank=2
            ),
            ViewConfig(
                "Swap first and last (3D)", {"axis_1": 0, "axis_2": -1}, min_rank=3
            ),
        ],
    ),
    "permute": ViewOperation(
        name="permute",
        nabla_fn=nb.permute,
        jax_fn=jnp.transpose,
        get_args=lambda r, c: (
            ((get_test_data_for_rank(r)[0],), c.params),
            ((get_test_data_for_rank(r)[1],), c.params),
        ),
        description="Permute tensor dimensions.",
        configs=[
            ViewConfig("Reverse 2D", {"axes": (1, 0)}, min_rank=2),
            ViewConfig("Reverse 3D", {"axes": (2, 1, 0)}, min_rank=3),
        ],
    ),
    "move_axis_to_front": ViewOperation(
        name="move_axis_to_front",
        nabla_fn=nb.move_axis_to_front,
        jax_fn=lambda x, axis: jax_move_axis_wrapper(x, axis, 0),
        get_args=lambda r, c: (
            ((get_test_data_for_rank(r)[0],), c.params),
            ((get_test_data_for_rank(r)[1],), c.params),
        ),
        description="Move an axis to the front.",
        configs=[ViewConfig("Move axis 1 to front", {"axis": 1}, min_rank=2)],
    ),
    "reshape": ViewOperation(
        name="reshape",
        nabla_fn=nb.reshape,
        jax_fn=jnp.reshape,
        get_args=lambda r, c: (
            (
                (get_test_data_for_rank(r, c.shape_generator)[0],),
                {
                    "shape": c.params["shape_fn"](
                        c.shape_generator(r)
                        if c.shape_generator
                        else get_shape_for_rank(r)
                    )
                },
            ),
            (
                (get_test_data_for_rank(r, c.shape_generator)[1],),
                {
                    "shape": c.params["shape_fn"](
                        c.shape_generator(r)
                        if c.shape_generator
                        else get_shape_for_rank(r)
                    )
                },
            ),
        ),
        description="Reshape a tensor to a new shape.",
        configs=[
            ViewConfig(
                "Flatten", {"shape_fn": lambda s: (int(np.prod(s)),)}, min_rank=2
            ),
            ViewConfig(
                "Unflatten",
                {"shape_fn": lambda s: (2, int(np.prod(s) // 2))},
                min_rank=1,
                shape_generator=lambda r: (10,),
            ),
        ],
    ),
    "broadcast_to": ViewOperation(
        name="broadcast_to",
        nabla_fn=nb.broadcast_to,
        jax_fn=jnp.broadcast_to,
        get_args=lambda r, c: (
            (
                (get_test_data_for_rank(r, c.shape_generator)[0],),
                {
                    "shape": c.params["shape_fn"](
                        c.shape_generator(r)
                        if c.shape_generator
                        else get_shape_for_rank(r)
                    )
                },
            ),
            (
                (get_test_data_for_rank(r, c.shape_generator)[1],),
                {
                    "shape": c.params["shape_fn"](
                        c.shape_generator(r)
                        if c.shape_generator
                        else get_shape_for_rank(r)
                    )
                },
            ),
        ),
        description="Broadcast a tensor to a new shape.",
        configs=[
            ViewConfig(
                "Broadcast vector to matrix",
                shape_generator=lambda r: (3,),
                params={"shape_fn": lambda s: (2, s[0])},
                min_rank=1,
            ),
            ViewConfig(
                "Add leading dimension",
                shape_generator=lambda r: (2, 3),
                params={"shape_fn": lambda s: (4, s[0], s[1])},
                min_rank=2,
            ),
        ],
    ),
    "squeeze": ViewOperation(
        name="squeeze",
        nabla_fn=nb.squeeze,
        jax_fn=jax_squeeze_wrapper,
        get_args=lambda r, c: (
            ((get_test_data_for_rank(r, c.shape_generator)[0],), c.params),
            ((get_test_data_for_rank(r, c.shape_generator)[1],), c.params),
        ),
        description="Remove dimensions of size 1.",
        configs=[
            ViewConfig(
                "Squeeze middle axis",
                {"axes": [1]},
                min_rank=3,
                shape_generator=lambda r: (2, 1, 3),
            ),
            ViewConfig(
                "Squeeze first axis",
                {"axes": [0]},
                min_rank=2,
                shape_generator=lambda r: (1, 2, 3),
            ),
        ],
    ),
    "unsqueeze": ViewOperation(
        name="unsqueeze",
        nabla_fn=nb.unsqueeze,
        jax_fn=jax_unsqueeze_wrapper,
        get_args=lambda r, c: (
            ((get_test_data_for_rank(r)[0],), c.params),
            ((get_test_data_for_rank(r)[1],), c.params),
        ),
        description="Add a new dimension of size 1.",
        configs=[
            ViewConfig("Unsqueeze in middle", {"axes": [1]}, min_rank=2),
            ViewConfig("Unsqueeze at end", {"axes": [-1]}, min_rank=1),
        ],
    ),
    "array_slice": ViewOperation(
        name="array_slice",
        nabla_fn=nb.array_slice,
        jax_fn=jax_slice_wrapper,
        get_args=lambda r, c: (
            ((get_test_data_for_rank(r, lambda r: (5, 6, 7))[0],), c.params),
            ((get_test_data_for_rank(r, lambda r: (5, 6, 7))[1],), c.params),
        ),
        description="Slice a tensor.",
        configs=[
            ViewConfig(
                "Slice with step",
                {"slices": [slice(None), slice(0, None, 2)]},
                min_rank=2,
            ),
            ViewConfig(
                "Slice with negative indices",
                {"slices": [slice(1, -1), slice(None)]},
                min_rank=2,
            ),
        ],
    ),
    "pad": ViewOperation(
        name="pad",
        nabla_fn=nb.pad,
        jax_fn=jax_pad_inverse_slice,
        get_args=lambda r, c: (
            ((get_test_data_for_rank(r, c.shape_generator)[0],), c.params),
            ((get_test_data_for_rank(r, c.shape_generator)[1],), c.params),
        ),
        description="Pad a tensor (inverse slice).",
        configs=[
            ViewConfig(
                "Pad matrix",
                {"slices": [slice(1, 3), slice(2, 4)], "target_shape": (4, 5)},
                min_rank=2,
                shape_generator=lambda r: (2, 2),
            ),
        ],
    ),
    "concatenate": ViewOperation(
        name="concatenate",
        nabla_fn=nb.concatenate,
        jax_fn=jnp.concatenate,
        is_list_input=True,
        get_args=lambda r, c: (
            (
                (
                    [
                        get_test_data_for_rank(r)[0],
                        get_test_data_for_rank(r)[0] + 10,
                    ],
                ),
                c.params,
            ),
            (
                (
                    [
                        get_test_data_for_rank(r)[1],
                        get_test_data_for_rank(r)[1] + 10,
                    ],
                ),
                c.params,
            ),
        ),
        description="Concatenate tensors along an axis.",
        configs=[ViewConfig("Concat axis 1", {"axis": 1}, min_rank=2)],
    ),
    "stack": ViewOperation(
        name="stack",
        nabla_fn=nb.stack,
        jax_fn=jnp.stack,
        is_list_input=True,
        get_args=lambda r, c: (
            (
                (
                    [
                        get_test_data_for_rank(r)[0],
                        get_test_data_for_rank(r)[0] + 10,
                    ],
                ),
                c.params,
            ),
            (
                (
                    [
                        get_test_data_for_rank(r)[1],
                        get_test_data_for_rank(r)[1] + 10,
                    ],
                ),
                c.params,
            ),
        ),
        description="Stack tensors along a new axis.",
        configs=[ViewConfig("Stack axis 1", {"axis": 1}, min_rank=1)],
    ),
}

# ============================================================================
# PARAMETERIZED TEST FRAMEWORK
# ============================================================================


def create_op_tests(operation: ViewOperation):
    """Create all 19 transformation tests for a given view operation."""
    current_rank = 2
    current_config = operation.configs[0]

    def get_configured_functions():
        """
        Core helper that uses `partial` to create clean, tensor-only functions.
        It handles standard f(*args) functions and adapts list-based functions
        (like concatenate) to fit the test harness.
        """
        (primals_nb_orig, params_nb), (primals_jax_orig, params_jax) = (
            operation.get_args(current_rank, current_config)
        )

        f_nb_partial = partial(operation.nabla_fn, **params_nb)
        f_jax_partial = partial(operation.jax_fn, **params_jax)

        if operation.is_list_input:
            # Adapt list-input functions to the f(*args) pattern.
            # Original primals: ([arr1, arr2],) -> New primals: (arr1, arr2)
            primals_nb = tuple(primals_nb_orig[0])
            primals_jax = tuple(primals_jax_orig[0])
            # Wrap the function to accept *args and bundle them into a list.
            f_nb = lambda *tensors: f_nb_partial(list(tensors))
            f_jax = lambda *tensors: f_jax_partial(list(tensors))
        else:
            # Standard case
            f_nb, f_jax = f_nb_partial, f_jax_partial
            primals_nb, primals_jax = primals_nb_orig, primals_jax_orig

        return f_nb, primals_nb, f_jax, primals_jax

    def test_1_baseline():
        """Test: f(x, *params)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "Baseline",
            lambda: f_nb(*primals_nb),
            lambda: f_jax(*primals_jax),
        )

    def test_2_vjp():
        """Test: vjp(f)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def nabla_fn():
            value, vjp_fn = nb.vjp(f_nb, *primals_nb)
            grad = vjp_fn(nb.ones_like(value))
            # Normalize to tuple for consistent comparison
            return value, grad if isinstance(grad, tuple) else (grad,)

        def jax_fn():
            value, vjp_fn = jax.vjp(f_jax, *primals_jax)
            grad = vjp_fn(jnp.ones_like(value))
            return value, grad

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
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
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

    # --- Full suite of 19 tests ---
    def test_6_jit_vjp():
        """Test: jit(vjp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def vjp_wrapper_nb(*args):
            val, vjp_fn = nb.vjp(f_nb, *args)
            grad = vjp_fn(nb.ones_like(val))
            return val, grad if isinstance(grad, tuple) else (grad,)

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
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "JIT(VMAP)",
            lambda: nb.djit(nb.vmap(f_nb))(*primals_nb),
            lambda: jax.jit(jax.vmap(f_jax))(*primals_jax),
        )

    def test_9_vmap_vjp():
        """Test: vmap(vjp(f))"""
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def vjp_wrapper_nb(*args):
            val, vjp_fn = nb.vjp(f_nb, *args)
            grad = vjp_fn(nb.ones_like(val))
            return val, grad if isinstance(grad, tuple) else (grad,)

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
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
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
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def vjp_wrapper_nb(*args):
            val, vjp_fn = nb.vjp(f_nb, *args)
            grad = vjp_fn(nb.ones_like(val))
            return val, grad if isinstance(grad, tuple) else (grad,)

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
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
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
            out = f_nb(*args)
            grad = vjp_fn(nb.ones_like(out))
            return grad if isinstance(grad, tuple) else (grad,)

        def first_grad_jax(*args):
            _val, vjp_fn = jax.vjp(f_jax, *args)
            out = f_jax(*args)
            grad = vjp_fn(jnp.ones_like(out))
            return grad  # JAX vjp always returns a tuple of gradients

        def nabla_fn():
            val, vjp_fn_2 = nb.vjp(first_grad_nb, *primals_nb)
            # Cotangent must be a tuple matching the output of first_grad_nb
            cotan_ones = tuple(nb.ones_like(c) for c in val)
            grad2 = vjp_fn_2(cotan_ones)
            return val, grad2 if isinstance(grad2, tuple) else (grad2,)

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
            out = f_nb(*args)
            grad = vjp_fn(nb.ones_like(out))
            return grad if isinstance(grad, tuple) else (grad,)

        def first_grad_jax(*args):
            _val, vjp_fn = jax.vjp(f_jax, *args)
            out = f_jax(*args)
            return vjp_fn(jnp.ones_like(out))

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
            grad = vjp_fn(cotan)
            return val, grad if isinstance(grad, tuple) else (grad,)

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
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        vmapped_f_nb, vmapped_f_jax = nb.vmap(f_nb), jax.vmap(f_jax)

        def nabla_fn():
            val, vjp_fn = nb.vjp(vmapped_f_nb, *primals_nb)
            grad = vjp_fn(nb.ones_like(val))
            return val, grad if isinstance(grad, tuple) else (grad,)

        def jax_fn():
            val, vjp_fn = jax.vjp(vmapped_f_jax, *primals_jax)
            return val, vjp_fn(jnp.ones_like(val))

        return run_test_with_consistency_check("VJP(VMAP)", nabla_fn, jax_fn)

    def test_18_jvp_vmap():
        """Test: jvp(vmap(f))"""
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
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
        if operation.name in ["pad"]:
            pytest.skip(f"VMAP is non-trivial for op '{operation.name}'")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        for p in primals_nb:
            if len(p.shape) < 2:
                pytest.skip("vmap(vmap) requires ndim >= 2 for all primals")

        return run_test_with_consistency_check(
            "VMAP(VMAP)",
            lambda: nb.vmap(nb.vmap(f_nb))(*primals_nb),
            lambda: jax.vmap(jax.vmap(f_jax))(*primals_jax),
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
# MAIN EXECUTION LOGIC & PYTEST INTEGRATION
# ============================================================================


def run_operation_tests(operation_name: str, all_configs: bool = False):
    """Run all transformation tests for a specific view operation."""
    if operation_name not in VIEW_OPERATIONS:
        print(f"Unknown operation: {operation_name}", file=sys.stderr)
        return False, 0, 0

    operation = VIEW_OPERATIONS[operation_name]
    print("=" * 80)
    print(f"TESTING VIEW OPERATION: {operation.name.upper()}")
    print("=" * 80)

    test_functions, set_rank_and_config_fn = create_op_tests(operation)
    ranks_to_test = get_ranks_to_test() if all_configs else [2, 3]

    total_passed, total_run = 0, 0
    failed_tests = []

    for rank in ranks_to_test:
        print(f"\n--- Testing Rank: {rank} ---")
        valid_configs = [c for c in operation.configs if rank >= c.min_rank]
        if not all_configs and valid_configs:
            # Pick a representative config for this rank
            valid_configs = [valid_configs[-1]]

        if not valid_configs:
            print(f"  > No valid configs for rank {rank}")
            continue

        for config in valid_configs:
            print(f"\n  > Config: {config.description}")
            set_rank_and_config_fn(rank, config)

            for i, test_func in enumerate(test_functions):
                desc = (
                    test_func.__doc__.split(":")[1].strip()
                    if test_func.__doc__
                    else f"Test {i + 1}"
                )
                print(f"    {i + 1:2d}. {desc:<15}", end="")

                try:
                    success = test_func()
                    total_run += 1
                    if success:
                        total_passed += 1
                    else:
                        failed_tests.append(
                            f"Rank {rank}, Config '{config.description}', Test '{desc}'"
                        )
                except pytest.skip.Exception as e:
                    print(f"SKIPPED ({e})")
                except Exception as e:
                    # For easier debugging, print the full error in the runner

                    print(f"ERROR ({type(e).__name__})")
                    # traceback.print_exc() # Uncomment for full stack trace
                    total_run += 1
                    failed_tests.append(
                        f"Rank {rank}, Config '{config.description}', Test '{desc}' (ERROR: {e})"
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
    """Run tests for all view operations."""
    print("=" * 100)
    print("COMPREHENSIVE VIEW OPERATIONS TEST SUITE")
    print("=" * 100)

    overall_success = True
    total_passed_all, total_run_all = 0, 0

    for op_name in VIEW_OPERATIONS:
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
        print("🎉 ALL VIEW OPERATIONS PASSED!")
    else:
        print("❌ SOME VIEW OPERATIONS FAILED")
    print("=" * 100)
    return overall_success


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Fixture to clean up caches after each individual test function."""
    yield
    cleanup_caches()


# Create a flat list of all test combinations for pytest parametrization
ALL_TEST_CASES = []
for op_name, op_def in VIEW_OPERATIONS.items():
    for rank in get_ranks_to_test():
        for config_idx, config in enumerate(op_def.configs):
            if rank >= config.min_rank:
                for trans_idx in range(19):
                    # Use a descriptive ID for easier test selection with -k
                    test_id = f"{op_name}-rank{rank}-cfg{config.description.replace(' ', '_')}-trans{trans_idx}"
                    ALL_TEST_CASES.append(
                        pytest.param(op_name, rank, config, trans_idx, id=test_id)
                    )


@pytest.mark.parametrize(
    "operation_name, rank, config, transformation_index", ALL_TEST_CASES
)
def test_view_operation_transformation(
    operation_name, rank, config, transformation_index
):
    """Pytest entry point for a specific view op/rank/config/transform."""
    operation = VIEW_OPERATIONS[operation_name]
    test_functions, set_rank_and_config_fn = create_op_tests(operation)
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
        pytest.skip(
            f"SKIPPED: {operation_name} - rank({rank}) - transform({test_desc}): {e}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <operation_name|all> [--all-configs]")
        print(f"Available operations: {list(VIEW_OPERATIONS.keys())}")
        sys.exit(1)

    op_arg = sys.argv[1]
    all_configs_arg = "--all-configs" in sys.argv

    if op_arg == "all":
        run_all_operations(all_configs=all_configs_arg)
    else:
        run_operation_tests(op_arg, all_configs=all_configs_arg)
