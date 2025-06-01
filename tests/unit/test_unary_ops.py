# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Unit tests for unary operations."""

import pytest

from tests.unit.test_utils import (
    JAX_AVAILABLE,
    SIMPLE_UNARY_SHAPES,
    UNARY_SHAPES,
    allclose_recursive,
    generate_test_data,
    requires_jax,
)

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

import nabla as nb

# Unary operation configurations
UNARY_OPERATIONS = [
    (
        "negate",
        nb.negate,
        jnp.negative if JAX_AVAILABLE else None,
        lambda i, d: {},
        "- (prefix)",
    ),
    ("exp", nb.exp, jnp.exp if JAX_AVAILABLE else None, lambda i, d: {}, None),
    (
        "log",
        nb.log,
        jnp.log if JAX_AVAILABLE else None,
        lambda i, d: {"ensure_positive": True},
        None,
    ),
    ("sin", nb.sin, jnp.sin if JAX_AVAILABLE else None, lambda i, d: {}, None),
    ("cos", nb.cos, jnp.cos if JAX_AVAILABLE else None, lambda i, d: {}, None),
    (
        "relu",
        nb.relu,
        (lambda x: jnp.maximum(x, 0)) if JAX_AVAILABLE else None,
        lambda i, d: {},
        None,
    ),
    (
        "transpose",
        nb.transpose,
        (lambda x: jnp.transpose(x)) if JAX_AVAILABLE else None,
        lambda i, d: {},
        None,
    ),
    (
        "reduce_sum",
        nb.reduce_sum,
        (lambda x: jnp.sum(x)) if JAX_AVAILABLE else None,
        lambda i, d: {},
        None,
    ),
]


@requires_jax
class TestUnaryOperations:
    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, op_symbol", UNARY_OPERATIONS
    )
    def test_unary_operation_values(
        self, op_name, nb_func, jax_func, constraints_fn, op_symbol, dtype, tolerances
    ):
        """Test unary operation values against JAX reference."""
        rtol, atol = tolerances["value"]

        shapes_to_test = UNARY_SHAPES
        # Special handling for operations that require at least 2D
        if op_name == "transpose":
            shapes_to_test = [(s[0], s[1]) for s in UNARY_SHAPES if len(s[0]) >= 2]

        for shape_a, desc in shapes_to_test:
            a_np = generate_test_data(shape_a, dtype, **constraints_fn(0, dtype))
            a_nb, a_jax = nb.Array.from_numpy(a_np), jnp.array(a_np)

            result_nb = nb_func(a_nb)
            result_jax = jax_func(a_jax)

            assert allclose_recursive(result_nb, result_jax, rtol, atol), (
                f"Failure for {op_name} with shape {shape_a} ({desc}), dtype {dtype}"
            )

    def test_negate_operator_overload(self, dtype, tolerances):
        """Test unary minus operator overload."""
        rtol, atol = tolerances["value"]
        shape = (3,)

        a_np = generate_test_data(shape, dtype)
        a_nb = nb.Array.from_numpy(a_np)

        result_nb_op = -a_nb  # Operator overload
        result_nb_func = nb.negate(a_nb)  # Function call
        result_jax = -jnp.array(a_np)  # JAX reference

        assert allclose_recursive(result_nb_op, result_jax, rtol, atol), (
            "Unary minus operator mismatch"
        )
        assert allclose_recursive(result_nb_func, result_jax, rtol, atol), (
            "Negate function mismatch"
        )
        assert allclose_recursive(result_nb_op, result_nb_func, rtol, atol), (
            "Operator vs function mismatch for negate"
        )

    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, op_symbol", UNARY_OPERATIONS
    )
    def test_unary_vjp(
        self, op_name, nb_func, jax_func, constraints_fn, op_symbol, dtype, tolerances
    ):
        """Test VJP for unary operations."""
        rtol_grad, atol_grad = tolerances["gradient"]

        shapes_config = SIMPLE_UNARY_SHAPES
        # Special handling for operations that require at least 2D
        if op_name == "transpose":
            shapes_config = [s for s in SIMPLE_UNARY_SHAPES if len(s[0]) >= 2]

        for shapes_tuple in shapes_config:
            primals_np = [
                generate_test_data(s, dtype, **constraints_fn(i, dtype))
                for i, s in enumerate(shapes_tuple)
            ]
            primals_nb = [nb.Array.from_numpy(p) for p in primals_np]
            primals_jax = [jnp.array(p) for p in primals_np]

            # Determine output shape for cotangent
            dummy_jax_out = jax_func(primals_jax[0])
            cotangent_np = generate_test_data(dummy_jax_out.shape, dummy_jax_out.dtype)
            cotangent_nb = nb.Array.from_numpy(cotangent_np)
            cotangent_jax = jnp.array(cotangent_np)

            # Nabla VJP
            def nabla_op(inputs):
                return [nb_func(inputs[0])]
            outputs_nb, vjp_fn_nb = nb.vjp(nabla_op, primals_nb)
            grads_nb = vjp_fn_nb([cotangent_nb])

            # JAX VJP
            def jax_op(x):
                return jax_func(x)
            _, vjp_fn_jax = jax.vjp(jax_op, *primals_jax)
            grads_jax = vjp_fn_jax(cotangent_jax)

            assert len(grads_nb) == 1, f"Expected 1 gradient, got {len(grads_nb)}"
            assert allclose_recursive(
                grads_nb[0], grads_jax[0], rtol_grad, atol_grad
            ), f"VJP grad mismatch for {op_name}, shapes {shapes_tuple}"

    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, op_symbol", UNARY_OPERATIONS
    )
    def test_unary_jvp(
        self, op_name, nb_func, jax_func, constraints_fn, op_symbol, dtype, tolerances
    ):
        """Test JVP for unary operations."""
        rtol_val, atol_val = tolerances["value"]
        rtol_tan, atol_tan = tolerances["gradient"]

        shapes_config = SIMPLE_UNARY_SHAPES
        # Special handling for operations that require at least 2D
        if op_name == "transpose":
            shapes_config = [s for s in SIMPLE_UNARY_SHAPES if len(s[0]) >= 2]

        for shapes_tuple in shapes_config:
            primals_np = [
                generate_test_data(s, dtype, **constraints_fn(i, dtype))
                for i, s in enumerate(shapes_tuple)
            ]
            tangents_np = [generate_test_data(s, dtype) for s in shapes_tuple]

            primals_nb = tuple(nb.Array.from_numpy(p) for p in primals_np)
            tangents_nb = tuple(nb.Array.from_numpy(t) for t in tangents_np)
            primals_jax = tuple(jnp.array(p) for p in primals_np)
            tangents_jax = tuple(jnp.array(t) for t in tangents_np)

            # Nabla JVP
            def nabla_op(inputs):
                return [nb_func(inputs[0])]
            primal_out_nb, tangent_out_nb = nb.jvp(
                nabla_op, list(primals_nb), list(tangents_nb)
            )

            # JAX JVP
            def jax_op(x):
                return jax_func(x)
            primal_out_jax, tangent_out_jax = jax.jvp(jax_op, primals_jax, tangents_jax)

            assert allclose_recursive(
                primal_out_nb[0], primal_out_jax, rtol_val, atol_val
            ), f"JVP primal mismatch for {op_name}, shapes {shapes_tuple}"
            assert allclose_recursive(
                tangent_out_nb[0], tangent_out_jax, rtol_tan, atol_tan
            ), f"JVP tangent mismatch for {op_name}, shapes {shapes_tuple}"
