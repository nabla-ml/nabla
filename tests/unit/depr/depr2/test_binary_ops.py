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

"""Unit tests for binary operations."""

import pytest

from tests.unit.test_utils import (
    BINARY_SHAPES_BROADCAST,
    JAX_AVAILABLE,
    SIMPLE_BINARY_SHAPES,
    allclose_recursive,
    generate_test_data,
    requires_jax,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp

import nabla as nb

# Binary operation configurations
BINARY_OPERATIONS = [
    ("add", nb.add, jnp.add if JAX_AVAILABLE else None, lambda i, d: {}, "+"),
    ("sub", nb.sub, jnp.subtract if JAX_AVAILABLE else None, lambda i, d: {}, "-"),
    (
        "mul",
        nb.mul,
        jnp.multiply if JAX_AVAILABLE else None,
        lambda i, d: ({"for_binary_op_rhs": True} if i == 1 else {}),
        "*",
    ),
    (
        "div",
        nb.div,
        jnp.divide if JAX_AVAILABLE else None,
        lambda i, d: ({"for_binary_op_rhs": True} if i == 1 else {}),
        "/",
    ),
    (
        "power",
        nb.pow,
        jnp.power if JAX_AVAILABLE else None,
        lambda i, d: (
            {"ensure_positive": True}
            if i == 0
            else {"for_binary_op_rhs": True}
            if i == 1
            else {}
        ),
        None,
    ),
]


@requires_jax
class TestBinaryOperations:
    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, op_symbol", BINARY_OPERATIONS
    )
    def test_binary_operation_values(
        self, op_name, nb_func, jax_func, constraints_fn, op_symbol, dtype, tolerances
    ):
        """Test binary operation values against JAX reference."""
        rtol, atol = tolerances["value"]

        for shape_a, shape_b, desc in BINARY_SHAPES_BROADCAST:
            a_np = generate_test_data(shape_a, dtype, **constraints_fn(0, dtype))
            b_np = generate_test_data(shape_b, dtype, **constraints_fn(1, dtype))

            a_nb, a_jax = nb.Array.from_numpy(a_np), jnp.array(a_np)
            b_nb, b_jax = nb.Array.from_numpy(b_np), jnp.array(b_np)

            result_nb = nb_func(a_nb, b_nb)
            result_jax = jax_func(a_jax, b_jax)

            assert allclose_recursive(result_nb, result_jax, rtol, atol), (
                f"Failure for {op_name} with shapes {shape_a}, {shape_b} ({desc}), dtype {dtype}"
            )

    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, op_symbol",
        [op for op in BINARY_OPERATIONS if op[4] is not None],
    )
    def test_operator_overloads(
        self, op_name, nb_func, jax_func, constraints_fn, op_symbol, dtype, tolerances
    ):
        """Test operator overloads (+, -, *, /) work correctly."""
        rtol, atol = tolerances["value"]
        shape_a, shape_b = (3,), (3,)  # Simple shapes for overload test

        a_np = generate_test_data(shape_a, dtype, **constraints_fn(0, dtype))
        b_np = generate_test_data(shape_b, dtype, **constraints_fn(1, dtype))

        a_nb = nb.Array.from_numpy(a_np)
        b_nb = nb.Array.from_numpy(b_np)

        # Test operator overload
        if op_symbol == "+":
            result_nb_op = a_nb + b_nb
        elif op_symbol == "-":
            result_nb_op = a_nb - b_nb
        elif op_symbol == "*":
            result_nb_op = a_nb * b_nb
        elif op_symbol == "/":
            result_nb_op = a_nb / b_nb
        else:
            pytest.skip(f"Operator {op_symbol} not implemented in test")

        # Test function call
        result_nb_func = nb_func(a_nb, b_nb)
        result_jax = jax_func(jnp.array(a_np), jnp.array(b_np))

        # All should match
        assert allclose_recursive(result_nb_op, result_jax, rtol, atol), (
            f"Operator overload mismatch for {op_symbol}"
        )
        assert allclose_recursive(result_nb_func, result_jax, rtol, atol), (
            f"Function call mismatch for {op_name}"
        )
        assert allclose_recursive(result_nb_op, result_nb_func, rtol, atol), (
            f"Operator vs function mismatch for {op_name}"
        )

    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, op_symbol", BINARY_OPERATIONS
    )
    def test_binary_vjp(
        self, op_name, nb_func, jax_func, constraints_fn, op_symbol, dtype, tolerances
    ):
        """Test VJP for binary operations."""
        import jax

        rtol_grad, atol_grad = tolerances["gradient"]

        for shapes_tuple in SIMPLE_BINARY_SHAPES:
            primals_np = [
                generate_test_data(s, dtype, **constraints_fn(i, dtype))
                for i, s in enumerate(shapes_tuple)
            ]
            primals_nb = [nb.Array.from_numpy(p) for p in primals_np]
            primals_jax = [jnp.array(p) for p in primals_np]

            # Determine output shape for cotangent
            dummy_jax_out = jax_func(primals_jax[0], primals_jax[1])
            cotangent_np = generate_test_data(dummy_jax_out.shape, dummy_jax_out.dtype)
            cotangent_nb = nb.Array.from_numpy(cotangent_np)
            cotangent_jax = jnp.array(cotangent_np)

            # Nabla VJP
            def nabla_op(x, y):
                return nb_func(x, y)

            outputs_nb, vjp_fn_nb = nb.vjp(nabla_op, *primals_nb)
            grads_nb = vjp_fn_nb(cotangent_nb)

            # JAX VJP
            def jax_op(x, y):
                return jax_func(x, y)

            _, vjp_fn_jax = jax.vjp(jax_op, *primals_jax)
            grads_jax = vjp_fn_jax(cotangent_jax)

            assert len(grads_nb) == 2, f"Expected 2 gradients, got {len(grads_nb)}"
            for i in range(2):
                assert allclose_recursive(
                    grads_nb[i], grads_jax[i], rtol_grad, atol_grad
                ), f"VJP grad mismatch for {op_name}, input {i}, shapes {shapes_tuple}"

    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, op_symbol", BINARY_OPERATIONS
    )
    def test_binary_jvp(
        self, op_name, nb_func, jax_func, constraints_fn, op_symbol, dtype, tolerances
    ):
        """Test JVP for binary operations."""
        import jax

        rtol_val, atol_val = tolerances["value"]
        rtol_tan, atol_tan = tolerances["gradient"]

        for shapes_tuple in SIMPLE_BINARY_SHAPES:
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
                return [nb_func(inputs[0], inputs[1])]

            primal_out_nb, tangent_out_nb = nb.jvp(
                nabla_op, list(primals_nb), list(tangents_nb)
            )

            # JAX JVP
            def jax_op(x, y):
                return jax_func(x, y)

            primal_out_jax, tangent_out_jax = jax.jvp(jax_op, primals_jax, tangents_jax)

            assert allclose_recursive(
                primal_out_nb[0], primal_out_jax, rtol_val, atol_val
            ), f"JVP primal mismatch for {op_name}, shapes {shapes_tuple}"
            assert allclose_recursive(
                tangent_out_nb[0], tangent_out_jax, rtol_tan, atol_tan
            ), f"JVP tangent mismatch for {op_name}, shapes {shapes_tuple}"
