"""
============================================================
COMPREHENSIVE AND UNIFIED NABLA OPERATIONS TEST SUITE
============================================================

This single, unified test suite validates ALL core operations in Nabla against JAX
as a ground truth. It consolidates the previously separate test suites for unary,
binary, reduction, view, and linear algebra operations into one powerful and
maintainable framework.

The suite is built on a flexible abstraction that can describe any operation,
regardless of its arity or parameter signature.

OPERATIONS TESTED (40 total):
─────────────────────────────
- UNARY (12): sin, cos, tanh, log, exp, sqrt, abs, floor, negate, relu,
              sigmoid, logical_not
- BINARY (12): add, mul, sub, div, floordiv, mod, pow, greater_equal, equal,
               not_equal, maximum, minimum
- REDUCTION (4): sum, mean, max, argmax
- VIEW (11): transpose, permute, reshape, broadcast_to, squeeze, unsqueeze,
             tensor_slice, pad, concatenate, stack, move_axis_to_front
- LINALG (1): matmul

TRANSFORMATION COMBINATIONS TESTED (19 total):
──────────────────────────────────────────────
The same 19 rigorous transformation combinations are applied to every valid
operation configuration:
Level 0: f(x, ...)
Level 1: vjp, jvp, vmap, jit
Level 2: jit(vjp), jit(jvp), jit(vmap), vmap(vjp), vmap(jvp)
Level 3: jit(vmap(vjp)), jit(vmap(jvp))
Level 4: vjp(vjp), jvp(vjp), vjp(jvp), jvp(jvp)
Level 5: vjp(vmap), jvp(vmap), vmap(vmap)
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nabla as nb

# Assume nabla and test utilities are in the path
try:
    from .test_utils import (
        cleanup_caches,
        get_shape_for_rank,
        run_test_with_consistency_check,
    )
except ImportError:
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from test_utils import (
        cleanup_caches,
        get_shape_for_rank,
        run_test_with_consistency_check,
    )

# ============================================================================
# UNIFIED ABSTRACTION
# ============================================================================


@dataclass
class OpConfig:
    """A unified configuration for a single test scenario of any operation."""

    description: str
    params: dict = field(default_factory=dict)
    ranks: tuple[int, ...] | None = None
    primal_shapes: tuple[tuple[int, ...], ...] | None = None
    is_list_input: bool = False
    domain_positive: bool = False
    input_dtype: str = "float32"
    use_stable_floats: bool = False


@dataclass
class Operation:
    """A unified definition for any operation to be tested."""

    name: str
    op_type: str
    nabla_fn: Callable
    jax_fn: Callable
    configs: list[OpConfig]
    get_args: Callable[[OpConfig], tuple[tuple, tuple]]


# ============================================================================
# JAX HELPER FUNCTIONS (Consolidated)
# ============================================================================


def jax_transpose_wrapper(x, axis_1, axis_2):
    rank = len(x.shape)
    if rank < 2:
        return x
    axes = list(range(rank))
    axis_1_pos, axis_2_pos = (
        (axis_1 if axis_1 >= 0 else rank + axis_1),
        (axis_2 if axis_2 >= 0 else rank + axis_2),
    )
    axes[axis_1_pos], axes[axis_2_pos] = axes[axis_2_pos], axes[axis_1_pos]
    return jnp.transpose(x, axes=axes)


def jax_pad_inverse_slice(arr, slices, target_shape):
    return jnp.zeros(target_shape, dtype=arr.dtype).at[tuple(slices)].set(arr)


def jax_unsqueeze_wrapper(x, axes):
    res = x
    for axis in sorted(axes):
        res = jnp.expand_dims(res, axis=axis)
    return res


def jax_squeeze_wrapper(x, axes):
    return jnp.squeeze(x, axis=tuple(axes))


def jax_slice_wrapper(x, slices):
    return x[tuple(slices)]


def jax_move_axis_wrapper(x, source, destination):
    return jnp.moveaxis(x, source, destination)


def jax_matmul_wrapper(x, y):
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
        res = jnp.squeeze(res, axis=tuple(squeeze_axes))
    return res


# ============================================================================
# UNIFIED TEST DATA AND ARGUMENT GENERATION
# ============================================================================


def get_test_data_for_shapes(
    shapes: tuple[tuple[int, ...], ...], config: OpConfig
) -> tuple[tuple[nb.Tensor, ...], tuple[jnp.ndarray, ...]]:
    nabla_primals, jax_primals = [], []
    for i, shape in enumerate(shapes):
        num_elements = int(np.prod(shape)) if shape else 1
        if config.input_dtype == "bool":
            jax_base = jax.numpy.arange(num_elements)
            nb_val, jax_val = (
                (
                    nb.equal(nb.arange(num_elements).reshape(shape) % 2, 0),
                    (jax_base.reshape(shape) % 2 == 0),
                )
                if shape
                else (nb.tensor(True), jnp.array(True))
            )
        else:  # Numeric
            if not shape:
                base_val = 2.5 if config.domain_positive else 1.5
                nb_val, jax_val = (
                    nb.tensor(base_val, dtype=nb.DType.float32),
                    jnp.array(base_val, dtype="float32"),
                )
            else:
                nb_base = nb.arange(num_elements, dtype=nb.DType.float32)
                jax_base = jax.numpy.arange(num_elements, dtype="float32")

                offset = 1.0 if config.domain_positive else float(i + 1)
                nb_val = (nb_base + offset).reshape(shape)
                jax_val = (jax_base + offset).reshape(shape)

                if not config.use_stable_floats:
                    nb_val *= 0.1
                    jax_val *= 0.1

        nabla_primals.append(nb_val)
        jax_primals.append(jax_val)
    return tuple(nabla_primals), tuple(jax_primals)


def get_tangents_for_jvp(primals_nb, primals_jax, config: OpConfig):
    if config.input_dtype == "bool":
        import jax._src.dtypes as jdt

        tangents_nb = tuple(
            nb.zeros_like(p).astype(nb.DType.float32) for p in primals_nb
        )
        tangents_jax = tuple(jnp.zeros_like(p, dtype=jdt.float0) for p in primals_jax)
    else:
        tangents_nb = tuple(nb.ones_like(p) for p in primals_nb)
        tangents_jax = tuple(jnp.ones_like(p) for p in primals_jax)
    return tangents_nb, tangents_jax


def standard_get_args(config: OpConfig):
    """Argument factory for simple ops where params are static."""
    shapes = config.primal_shapes or tuple(get_shape_for_rank(r) for r in config.ranks)
    primals_nb, primals_jax = get_test_data_for_shapes(shapes, config)
    return (primals_nb, config.params), (primals_jax, config.params)


def get_reshape_args(config: OpConfig):
    """Argument factory for reshape, where the `shape` param depends on the input."""
    (primals_nb, _), (primals_jax, _) = standard_get_args(config)
    if config.description == "Flatten":
        target_shape = (int(np.prod(primals_nb[0].shape)),)
    else:  # Unflatten
        target_shape = (2, 4, int(np.prod(primals_nb[0].shape) // 8))
    params = {"shape": target_shape}
    return (primals_nb, params), (primals_jax, params)


def get_broadcast_to_args(config: OpConfig):
    """Argument factory for broadcast_to, where `shape` depends on input."""
    (primals_nb, _), (primals_jax, _) = standard_get_args(config)
    if "Vec_to_Mat" in config.description:
        target_shape = (2, *primals_nb[0].shape)
    else:  # Mat_to_Ten3D
        target_shape = (4, *primals_nb[0].shape)
    params = {"shape": target_shape}
    return (primals_nb, params), (primals_jax, params)


# ============================================================================
# MASTER OPERATION DEFINITIONS
# ============================================================================

ALL_OPERATIONS = {}

# --- UNARY OPERATIONS ---
unary_ops_data = [
    ("sin", nb.sin, jnp.sin, False),
    ("cos", nb.cos, jnp.cos, False),
    ("tanh", nb.tanh, jnp.tanh, False),
    ("log", nb.log, jnp.log, True),
    ("exp", nb.exp, jnp.exp, False),
    ("sqrt", nb.sqrt, jnp.sqrt, True),
    ("abs", nb.abs, jnp.abs, False),
    ("floor", nb.floor, jnp.floor, False),
    ("negate", nb.negate, jnp.negative, False),
    ("relu", nb.relu, jax.nn.relu, False),
    ("sigmoid", nb.sigmoid, jax.nn.sigmoid, False),
]
for name, nb_fn, jax_fn, domain_pos in unary_ops_data:
    configs = [
        OpConfig(f"Rank{r}", ranks=(r,), domain_positive=domain_pos)
        for r in [0, 1, 2, 3]
    ]
    ALL_OPERATIONS[name] = Operation(
        name, "UNARY", nb_fn, jax_fn, configs, standard_get_args
    )

ALL_OPERATIONS["logical_not"] = Operation(
    "logical_not",
    "UNARY",
    nb.logical_not,
    jnp.logical_not,
    [OpConfig(f"Rank{r}", ranks=(r,), input_dtype="bool") for r in [0, 1, 2, 3]],
    standard_get_args,
)

# --- BINARY OPERATIONS ---
binary_ops_data = [
    ("add", nb.add, jnp.add),
    ("mul", nb.mul, jnp.multiply),
    ("sub", nb.sub, jnp.subtract),
    ("div", nb.div, jnp.divide),
    ("floordiv", nb.floordiv, jnp.floor_divide),
    ("mod", nb.mod, jnp.mod),
    ("pow", nb.pow, jnp.power),
    ("greater_equal", nb.greater_equal, jnp.greater_equal),
    ("equal", nb.equal, jnp.equal),
    ("not_equal", nb.not_equal, jnp.not_equal),
    ("maximum", nb.maximum, jnp.maximum),
    ("minimum", nb.minimum, jnp.minimum),
]
binary_rank_combos = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]
for name, nb_fn, jax_fn in binary_ops_data:
    use_stable = name in ["div", "floordiv", "mod"]
    configs = [
        OpConfig(f"Ranks({r1},{r2})", ranks=(r1, r2), use_stable_floats=use_stable)
        for r1, r2 in binary_rank_combos
    ]
    ALL_OPERATIONS[name] = Operation(
        name, "BINARY", nb_fn, jax_fn, configs, standard_get_args
    )

# --- REDUCTION OPERATIONS ---
reduction_params = [
    {"desc": "all-keep", "axes": None, "keep_dims": True, "min_rank": 1},
    {"desc": "all-nokeep", "axes": None, "keep_dims": False, "min_rank": 1},
    {"desc": "axis0-keep", "axes": 0, "keep_dims": True, "min_rank": 1},
    {"desc": "axis0-nokeep", "axes": 0, "keep_dims": False, "min_rank": 1},
    {"desc": "axis1-keep", "axes": 1, "keep_dims": True, "min_rank": 2},
    {"desc": "axis1-nokeep", "axes": 1, "keep_dims": False, "min_rank": 2},
    {"desc": "axes_0,1-keep", "axes": (0, 1), "keep_dims": True, "min_rank": 2},
    {"desc": "axis2-keep", "axes": 2, "keep_dims": True, "min_rank": 3},
    {"desc": "axes_0,2-nokeep", "axes": (0, 2), "keep_dims": False, "min_rank": 3},
]
for name, nb_fn, jax_fn in [
    ("sum", nb.sum, jnp.sum),
    ("mean", nb.mean, jnp.mean),
    ("max", nb.max, jnp.max),
    ("argmax", nb.argmax, jnp.argmax),
]:
    configs = []
    for rank in [1, 2, 3]:
        for p in reduction_params:
            if rank >= p["min_rank"]:
                if name == "argmax" and isinstance(p["axes"], tuple):
                    continue
                configs.append(
                    OpConfig(
                        f"Rank{rank}-{p['desc']}",
                        ranks=(rank,),
                        params={"axes": p["axes"], "keep_dims": p["keep_dims"]},
                    )
                )
    ALL_OPERATIONS[name] = Operation(
        name, "REDUCTION", nb_fn, jax_fn, configs, standard_get_args
    )

# --- VIEW OPERATIONS ---
view_ops_data = [
    Operation(
        "transpose",
        "VIEW",
        nb.transpose,
        jax_transpose_wrapper,
        [
            OpConfig(
                "Rank2-Swap_last_two", ranks=(2,), params={"axis_1": -2, "axis_2": -1}
            ),
            OpConfig(
                "Rank3-Swap_first_last", ranks=(3,), params={"axis_1": 0, "axis_2": -1}
            ),
        ],
        standard_get_args,
    ),
    Operation(
        "permute",
        "VIEW",
        nb.permute,
        jnp.transpose,
        [
            OpConfig("Rank2-Reverse", ranks=(2,), params={"axes": (1, 0)}),
            OpConfig("Rank3-Reverse", ranks=(3,), params={"axes": (2, 1, 0)}),
        ],
        standard_get_args,
    ),
    Operation(
        "move_axis_to_front",
        "VIEW",
        nb.move_axis_to_front,
        lambda x, axis: jax_move_axis_wrapper(x, axis, 0),
        [
            OpConfig("Rank2-Move_axis_1", ranks=(2,), params={"axis": 1}),
        ],
        standard_get_args,
    ),
    Operation(
        "reshape",
        "VIEW",
        nb.reshape,
        jnp.reshape,
        [
            OpConfig("Flatten", ranks=(3,)),
            OpConfig("Unflatten", primal_shapes=((2 * 4 * 8,),)),
        ],
        get_reshape_args,
    ),
    Operation(
        "broadcast_to",
        "VIEW",
        nb.broadcast_to,
        jnp.broadcast_to,
        [
            OpConfig("Vec_to_Mat", primal_shapes=((3,),)),
            OpConfig("Mat_to_Ten3D", primal_shapes=((2, 3),)),
        ],
        get_broadcast_to_args,
    ),
    Operation(
        "squeeze",
        "VIEW",
        nb.squeeze,
        jax_squeeze_wrapper,
        [
            OpConfig(
                "Squeeze_middle", primal_shapes=((2, 1, 3),), params={"axes": [1]}
            ),
        ],
        standard_get_args,
    ),
    Operation(
        "unsqueeze",
        "VIEW",
        nb.unsqueeze,
        jax_unsqueeze_wrapper,
        [
            OpConfig("Unsqueeze_middle", ranks=(2,), params={"axes": [1]}),
        ],
        standard_get_args,
    ),
    Operation(
        "tensor_slice",
        "VIEW",
        nb.tensor_slice,
        jax_slice_wrapper,
        [
            OpConfig(
                "Slice_with_step",
                primal_shapes=((5, 6, 7),),
                params={"slices": [slice(None), slice(0, None, 2)]},
            ),
        ],
        standard_get_args,
    ),
    Operation(
        "pad",
        "VIEW",
        nb.pad,
        jax_pad_inverse_slice,
        [
            OpConfig(
                "Pad_matrix",
                primal_shapes=((2, 2),),
                params={"slices": [slice(1, 3), slice(2, 4)], "target_shape": (4, 5)},
            ),
        ],
        standard_get_args,
    ),
    Operation(
        "concatenate",
        "VIEW",
        nb.concatenate,
        jnp.concatenate,
        [
            OpConfig(
                "Concat_axis1", ranks=(2, 2), params={"axis": 1}, is_list_input=True
            ),
        ],
        standard_get_args,
    ),
    Operation(
        "stack",
        "VIEW",
        nb.stack,
        jnp.stack,
        [
            OpConfig(
                "Stack_axis1", ranks=(2, 2), params={"axis": 1}, is_list_input=True
            ),
        ],
        standard_get_args,
    ),
]
for op in view_ops_data:
    ALL_OPERATIONS[op.name] = op

# --- LINALG OPERATIONS ---
ALL_OPERATIONS["matmul"] = Operation(
    "matmul",
    "LINALG",
    nb.matmul,
    jax_matmul_wrapper,
    [
        OpConfig("Vector_@_Vector", primal_shapes=((4,), (4,))),
        OpConfig("Matrix_@_Vector", primal_shapes=((3, 4), (4,))),
        OpConfig("Vector_@_Matrix", primal_shapes=((4,), (4, 5))),
        OpConfig("Matrix_@_Matrix", primal_shapes=((3, 4), (4, 5))),
        OpConfig("Batched_Matmul", primal_shapes=((10, 3, 4), (10, 4, 5))),
    ],
    standard_get_args,
)

# ============================================================================
# GENERIC TEST FACTORY
# ============================================================================


def create_op_tests(operation: Operation):
    """Creates all 19 transformation tests for a given unified operation."""
    current_config: OpConfig = operation.configs[0]

    def get_configured_functions():
        (primals_nb_orig, params_nb), (primals_jax_orig, params_jax_raw) = (
            operation.get_args(current_config)
        )

        f_nb_partial = partial(operation.nabla_fn, **params_nb)

        if operation.name == "permute":
            f_jax_partial = partial(operation.jax_fn, **params_jax_raw)
        elif operation.name in ["squeeze", "unsqueeze", "move_axis_to_front"]:
            arg_name = "axes" if operation.name != "move_axis_to_front" else "axis"
            f_jax_partial = lambda x: operation.jax_fn(x, params_jax_raw[arg_name])
        else:
            params_jax = {
                k.replace("axes", "axis").replace("keep_dims", "keepdims"): v
                for k, v in params_jax_raw.items()
            }
            f_jax_partial = partial(operation.jax_fn, **params_jax)

        if current_config.is_list_input:
            f_nb = lambda *tensors: f_nb_partial(list(tensors))
            f_jax = lambda *tensors: f_jax_partial(list(tensors))
            return f_nb, primals_nb_orig, f_jax, primals_jax_orig
        else:
            return f_nb_partial, primals_nb_orig, f_jax_partial, primals_jax_orig

    def test_1_baseline():
        """Test: f(x)"""
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
            return val, grad if isinstance(grad, tuple) else (grad,)

        def jax_fn():
            val, vjp_fn = jax.vjp(f_jax, *primals_jax)
            return val, vjp_fn(jnp.ones_like(val))

        return run_test_with_consistency_check("VJP", nabla_fn, jax_fn)

    def test_3_jvp():
        """Test: jvp(f)"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb, tangents_jax = get_tangents_for_jvp(
            primals_nb, primals_jax, current_config
        )
        return run_test_with_consistency_check(
            "JVP",
            lambda: nb.jvp(f_nb, primals_nb, tangents_nb),
            lambda: jax.jvp(f_jax, primals_jax, tangents_jax),
        )

    def test_4_vmap():
        """Test: vmap(f)"""
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
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
        tangents_nb, tangents_jax = get_tangents_for_jvp(
            primals_nb, primals_jax, current_config
        )
        jitted_jvp_nb = nb.djit(lambda p, t: nb.jvp(f_nb, p, t))
        jitted_jvp_jax = jax.jit(lambda p, t: jax.jvp(f_jax, p, t))
        return run_test_with_consistency_check(
            "JIT(JVP)",
            lambda: jitted_jvp_nb(primals_nb, tangents_nb),
            lambda: jitted_jvp_jax(primals_jax, tangents_jax),
        )

    def test_8_jit_vmap():
        """Test: jit(vmap(f))"""
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        return run_test_with_consistency_check(
            "JIT(VMAP)",
            lambda: nb.djit(nb.vmap(f_nb))(*primals_nb),
            lambda: jax.jit(jax.vmap(f_jax))(*primals_jax),
        )

    def test_9_vmap_vjp():
        """Test: vmap(vjp(f))"""
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
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
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def jvp_wrapper_nb(*args):
            tangents, _ = get_tangents_for_jvp(args, args, current_config)
            return nb.jvp(f_nb, args, tangents)

        def jvp_wrapper_jax(*args):
            _, tangents = get_tangents_for_jvp(args, args, current_config)
            return jax.jvp(f_jax, args, tangents)

        return run_test_with_consistency_check(
            "VMAP(JVP)",
            lambda: nb.vmap(jvp_wrapper_nb)(*primals_nb),
            lambda: jax.vmap(jvp_wrapper_jax)(*primals_jax),
        )

    def test_11_jit_vmap_vjp():
        """Test: jit(vmap(vjp(f)))"""
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
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
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()

        def jvp_wrapper_nb(*args):
            tangents, _ = get_tangents_for_jvp(args, args, current_config)
            return nb.jvp(f_nb, args, tangents)

        def jvp_wrapper_jax(*args):
            _, tangents = get_tangents_for_jvp(args, args, current_config)
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
            grad = vjp_fn(nb.ones_like(f_nb(*args)))
            return grad if isinstance(grad, tuple) else (grad,)

        def first_grad_jax(*args):
            _val, vjp_fn = jax.vjp(f_jax, *args)
            return vjp_fn(jnp.ones_like(f_jax(*args)))

        def nabla_fn():
            val, vjp_fn_2 = nb.vjp(first_grad_nb, *primals_nb)
            cotan = tuple(nb.ones_like(c) for c in val)
            grad2 = vjp_fn_2(cotan)
            return val, grad2 if isinstance(grad2, tuple) else (grad2,)

        def jax_fn():
            val, vjp_fn_2 = jax.vjp(first_grad_jax, *primals_jax)
            cotan = tuple(jnp.ones_like(c) for c in val)
            return val, vjp_fn_2(cotan)

        return run_test_with_consistency_check("VJP(VJP)", nabla_fn, jax_fn)

    def test_14_jvp_vjp():
        """Test: jvp(vjp(f))"""
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb, tangents_jax = get_tangents_for_jvp(
            primals_nb, primals_jax, current_config
        )

        def first_grad_nb(*args):
            _val, vjp_fn = nb.vjp(f_nb, *args)
            grad = vjp_fn(nb.ones_like(f_nb(*args)))
            return grad if isinstance(grad, tuple) else (grad,)

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
        tangents_nb, tangents_jax = get_tangents_for_jvp(
            primals_nb, primals_jax, current_config
        )

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
        tangents_nb, tangents_jax = get_tangents_for_jvp(
            primals_nb, primals_jax, current_config
        )

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
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
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
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        tangents_nb, tangents_jax = get_tangents_for_jvp(
            primals_nb, primals_jax, current_config
        )
        vmapped_f_nb, vmapped_f_jax = nb.vmap(f_nb), jax.vmap(f_jax)
        return run_test_with_consistency_check(
            "JVP(VMAP)",
            lambda: nb.jvp(vmapped_f_nb, primals_nb, tangents_nb),
            lambda: jax.jvp(vmapped_f_jax, primals_jax, tangents_jax),
        )

    def test_19_vmap_vmap():
        """Test: vmap(vmap(f))"""
        if operation.name == "pad":
            pytest.skip("vmap for pad is non-trivial")
        f_nb, primals_nb, f_jax, primals_jax = get_configured_functions()
        for p in primals_nb:
            if len(p.shape) < 2:
                pytest.skip("vmap(vmap) requires ndim >= 2 for all primals")
        return run_test_with_consistency_check(
            "VMAP(VMAP)",
            lambda: nb.vmap(nb.vmap(f_nb))(*primals_nb),
            lambda: jax.vmap(jax.vmap(f_jax))(*primals_jax),
        )

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

    def set_config(config):
        nonlocal current_config
        current_config = config

    return test_functions, set_config


# ============================================================================
# PYTEST INTEGRATION & EXECUTION
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    cleanup_caches()


ALL_TEST_CASES = []
for op_name, op_def in ALL_OPERATIONS.items():
    for config in op_def.configs:
        for trans_idx in range(19):
            test_id = f"{op_name}-{config.description.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}-trans{trans_idx}"
            ALL_TEST_CASES.append(pytest.param(op_name, config, trans_idx, id=test_id))


@pytest.mark.parametrize("operation_name, config, transformation_index", ALL_TEST_CASES)
def test_unified_operation_transformation(operation_name, config, transformation_index):
    """Pytest entry point for a specific op/config/transform combination."""
    operation = ALL_OPERATIONS[operation_name]
    test_functions, set_config_fn = create_op_tests(operation)

    set_config_fn(config)
    test_func = test_functions[transformation_index]
    test_desc = (
        test_func.__doc__.split(":")[1].strip()
        if test_func.__doc__
        else f"transform_{transformation_index}"
    )

    try:
        success = test_func()
        assert success, (
            f"Failed: {operation_name} - config('{config.description}') - transform({test_desc})"
        )
    except pytest.skip.Exception as e:
        pytest.skip(
            f"SKIPPED: {operation_name} - config('{config.description}') - transform({test_desc}): {e}"
        )


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================


def run_operation_tests(operation_name: str):
    """Runs all configurations and transformations for a specific operation."""
    if operation_name not in ALL_OPERATIONS:
        print(f"Unknown operation: {operation_name}", file=sys.stderr)
        return False, 0, 0

    operation = ALL_OPERATIONS[operation_name]
    print("=" * 80)
    print(f"TESTING {operation.op_type} OPERATION: {operation.name.upper()}")
    print("=" * 80)

    test_functions, set_config_fn = create_op_tests(operation)
    total_passed, total_run = 0, 0
    failed_tests = []

    for config in operation.configs:
        print(f"\n  > Config: {config.description}")
        set_config_fn(config)

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
                    failed_tests.append(f"Config '{config.description}', Test '{desc}'")
            except pytest.skip.Exception as e:
                print(f"SKIPPED ({e})")
            except Exception as e:
                import traceback

                print(f"ERROR ({type(e).__name__})")
                traceback.print_exc()
                total_run += 1
                failed_tests.append(
                    f"Config '{config.description}', Test '{desc}' (ERROR: {e})"
                )
            finally:
                cleanup_caches()

    print(
        f"\n{'=' * 80}\nOPERATION {operation.name.upper()} RESULTS: {total_passed}/{total_run} tests passed"
    )
    if failed_tests:
        print("--- FAILED TESTS ---\n" + "\n".join(f"  - {f}" for f in failed_tests))

    return total_passed == total_run and total_run > 0, total_passed, total_run


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <operation_name|all>")
        print(f"Available operations: {list(ALL_OPERATIONS.keys())}")
        sys.exit(1)

    op_arg = sys.argv[1]
    if op_arg == "all":
        print("=" * 100)
        print("RUNNING FULL UNIFIED TEST SUITE")
        print("=" * 100)
        overall_success = True
        total_passed_all, total_run_all = 0, 0
        for op_name in sorted(ALL_OPERATIONS.keys()):
            success, passed, run = run_operation_tests(op_name)
            overall_success &= success
            total_passed_all += passed
            total_run_all += run
        print("\n" + "=" * 100)
        print("🏁 FINAL SUMMARY")
        print("=" * 100)
        rate = (total_passed_all / total_run_all * 100) if total_run_all > 0 else 0
        print(f"TOTAL TESTS PASSED: {total_passed_all}/{total_run_all}")
        print(f"OVERALL SUCCESS RATE: {rate:.1f}%")
        print("🎉 ALL TESTS PASSED!" if overall_success else "❌ SOME TESTS FAILED")
        print("=" * 100)
    else:
        run_operation_tests(op_arg)
