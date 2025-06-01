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

"""Unit tests for linear algebra operations."""

import pytest

from tests.unit.test_utils import (
    JAX_AVAILABLE,
    MATMUL_SHAPES,
    SIMPLE_MATMUL_SHAPES,
    allclose_recursive,
    generate_test_data,
    requires_jax,
)

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

import nabla as nb


@requires_jax
class TestLinearAlgebraOperations:
    def test_matmul_values(self, dtype, tolerances):
        """Test matrix multiplication values against JAX reference."""
        rtol, atol = tolerances["value"]

        for shape_a, shape_b, desc in MATMUL_SHAPES:
            a_np = generate_test_data(shape_a, dtype)
            b_np = generate_test_data(shape_b, dtype)

            a_nb, a_jax = nb.Array.from_numpy(a_np), jnp.array(a_np)
            b_nb, b_jax = nb.Array.from_numpy(b_np), jnp.array(b_np)

            result_nb = nb.matmul(a_nb, b_nb)
            result_jax = jnp.matmul(a_jax, b_jax)

            assert allclose_recursive(result_nb, result_jax, rtol, atol), (
                f"Matmul failure with shapes {shape_a}, {shape_b} ({desc}), dtype {dtype}"
            )

    def test_matmul_vjp(self, dtype, tolerances):
        """Test VJP for matrix multiplication."""
        rtol_grad, atol_grad = tolerances["gradient"]

        for shapes_tuple in SIMPLE_MATMUL_SHAPES:
            primals_np = [generate_test_data(s, dtype) for s in shapes_tuple]
            primals_nb = [nb.Array.from_numpy(p) for p in primals_np]
            primals_jax = [jnp.array(p) for p in primals_np]

            # Determine output shape for cotangent
            dummy_jax_out = jnp.matmul(primals_jax[0], primals_jax[1])
            cotangent_np = generate_test_data(dummy_jax_out.shape, dummy_jax_out.dtype)
            cotangent_nb = nb.Array.from_numpy(cotangent_np)
            cotangent_jax = jnp.array(cotangent_np)

            # Nabla VJP
            nabla_op = lambda inputs: [nb.matmul(inputs[0], inputs[1])]
            outputs_nb, vjp_fn_nb = nb.vjp(nabla_op, primals_nb)
            grads_nb = vjp_fn_nb([cotangent_nb])

            # JAX VJP
            jax_op = lambda x, y: jnp.matmul(x, y)
            _, vjp_fn_jax = jax.vjp(jax_op, *primals_jax)
            grads_jax = vjp_fn_jax(cotangent_jax)

            assert len(grads_nb) == 2, f"Expected 2 gradients, got {len(grads_nb)}"
            for i in range(2):
                assert allclose_recursive(
                    grads_nb[i], grads_jax[i], rtol_grad, atol_grad
                ), f"Matmul VJP grad mismatch for input {i}, shapes {shapes_tuple}"

    def test_matmul_jvp(self, dtype, tolerances):
        """Test JVP for matrix multiplication."""
        rtol_val, atol_val = tolerances["value"]
        rtol_tan, atol_tan = tolerances["gradient"]

        for shapes_tuple in SIMPLE_MATMUL_SHAPES:
            primals_np = [generate_test_data(s, dtype) for s in shapes_tuple]
            tangents_np = [generate_test_data(s, dtype) for s in shapes_tuple]

            primals_nb = tuple(nb.Array.from_numpy(p) for p in primals_np)
            tangents_nb = tuple(nb.Array.from_numpy(t) for t in tangents_np)
            primals_jax = tuple(jnp.array(p) for p in primals_np)
            tangents_jax = tuple(jnp.array(t) for t in tangents_np)

            # Nabla JVP
            nabla_op = lambda inputs: [nb.matmul(inputs[0], inputs[1])]
            primal_out_nb, tangent_out_nb = nb.jvp(
                nabla_op, list(primals_nb), list(tangents_nb)
            )

            # JAX JVP
            jax_op = lambda x, y: jnp.matmul(x, y)
            primal_out_jax, tangent_out_jax = jax.jvp(jax_op, primals_jax, tangents_jax)

            assert allclose_recursive(
                primal_out_nb[0], primal_out_jax, rtol_val, atol_val
            ), f"Matmul JVP primal mismatch, shapes {shapes_tuple}"
            assert allclose_recursive(
                tangent_out_nb[0], tangent_out_jax, rtol_tan, atol_tan
            ), f"Matmul JVP tangent mismatch, shapes {shapes_tuple}"

    @pytest.mark.skip(
        reason="Matmul vmap has known issues with batch dimension handling"
    )
    def test_matmul_vmap(self, dtype, tolerances):
        """Test vmap for matrix multiplication (currently skipped due to known issues)."""
        pass
