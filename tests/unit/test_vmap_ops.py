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

"""Unit tests for vmap transformations."""

import pytest

from tests.unit.test_utils import (
    JAX_AVAILABLE,
    allclose_recursive,
    generate_test_data,
    requires_jax,
)

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

import nabla as nb

# Operations that work well with vmap
VMAP_COMPATIBLE_OPERATIONS = [
    ("add", nb.add, jnp.add, lambda i, d: {}, 2),
    ("sub", nb.sub, jnp.subtract, lambda i, d: {}, 2),
    (
        "mul",
        nb.mul,
        jnp.multiply,
        lambda i, d: ({"for_binary_op_rhs": True} if i == 1 else {}),
        2,
    ),
    ("exp", nb.exp, jnp.exp, lambda i, d: {}, 1),
    ("sin", nb.sin, jnp.sin, lambda i, d: {}, 1),
    ("cos", nb.cos, jnp.cos, lambda i, d: {}, 1),
    ("relu", nb.relu, lambda x: jnp.maximum(x, 0), lambda i, d: {}, 1),
]


@requires_jax
class TestVmapTransformations:
    @pytest.mark.parametrize(
        "op_name, nb_func, jax_func, constraints_fn, arity", VMAP_COMPATIBLE_OPERATIONS
    )
    def test_vmap_operations(
        self, op_name, nb_func, jax_func, constraints_fn, arity, dtype, tolerances
    ):
        """Test vmap for compatible operations."""
        # Skip operations with known issues
        if op_name in ["reduce_sum", "matmul"]:
            pytest.skip(f"Skipping {op_name} vmap test due to known issues")

        rtol, atol = tolerances["value"]
        batch_size = 3

        if arity == 1:
            inner_shapes_config = [((2, 3),)]
        else:  # arity == 2
            inner_shapes_config = [((2, 3), (2, 3)), ((2, 3), ())]  # Test broadcasting

        for inner_shapes_tuple in inner_shapes_config:
            # Create batched inputs
            batched_primals_np = []
            in_axes_list = []

            # First argument (always batched)
            batched_primals_np.append(
                generate_test_data(
                    (batch_size,) + inner_shapes_tuple[0],
                    dtype,
                    **constraints_fn(0, dtype),
                )
            )
            in_axes_list.append(0)

            # Second argument (if binary operation)
            if arity == 2:
                if inner_shapes_tuple[1]:  # Non-empty shape (batched)
                    batched_primals_np.append(
                        generate_test_data(
                            (batch_size,) + inner_shapes_tuple[1],
                            dtype,
                            **constraints_fn(1, dtype),
                        )
                    )
                    in_axes_list.append(0)
                else:  # Scalar/broadcast (not batched)
                    batched_primals_np.append(
                        generate_test_data(
                            inner_shapes_tuple[1], dtype, **constraints_fn(1, dtype)
                        )
                    )
                    in_axes_list.append(None)

            in_axes = tuple(in_axes_list)

            primals_nb = [nb.Array.from_numpy(p) for p in batched_primals_np]
            primals_jax = [jnp.array(p) for p in batched_primals_np]

            # Define operations for vmap
            if arity == 1:
                nabla_op = lambda inputs: [nb_func(inputs[0])]
                jax_op = lambda x: jax_func(x)
            else:  # arity == 2
                nabla_op = lambda inputs: [nb_func(inputs[0], inputs[1])]
                jax_op = lambda x, y: jax_func(x, y)

            # Apply vmap
            vmapped_fn_nb = nb.vmap(nabla_op, in_axes=in_axes)
            result_nb = vmapped_fn_nb(primals_nb)

            vmapped_fn_jax = jax.vmap(jax_op, in_axes=in_axes)
            result_jax = vmapped_fn_jax(*primals_jax)

            assert allclose_recursive(result_nb[0], result_jax, rtol, atol), (
                f"vmap output mismatch for {op_name}, inner_shapes {inner_shapes_tuple}, in_axes {in_axes}"
            )

    def test_vmap_simple_function(self, dtype, tolerances):
        """Test vmap on a simple custom function."""
        rtol, atol = tolerances["value"]
        batch_size = 4

        def simple_func(inputs):
            x = inputs[0]
            return [x * x + nb.array([1.0], dtype=x.dtype)]

        def jax_simple_func(x):
            return x * x + 1.0

        # Create batched input
        x_batched_np = generate_test_data((batch_size, 3), dtype)
        x_batched_nb = nb.Array.from_numpy(x_batched_np)
        x_batched_jax = jnp.array(x_batched_np)

        # Apply vmap
        vmapped_nb = nb.vmap(simple_func)
        result_nb = vmapped_nb([x_batched_nb])

        vmapped_jax = jax.vmap(jax_simple_func)
        result_jax = vmapped_jax(x_batched_jax)

        assert allclose_recursive(result_nb[0], result_jax, rtol, atol), (
            "vmap simple function mismatch"
        )

    def test_vmap_nested_function(self, dtype, tolerances):
        """Test vmap on a function with multiple operations."""
        rtol, atol = tolerances["value"]
        batch_size = 2

        def nested_func(inputs):
            x, y = inputs[0], inputs[1]
            z = nb.add(x, y)
            return [nb.mul(z, z)]

        def jax_nested_func(x, y):
            z = jnp.add(x, y)
            return jnp.multiply(z, z)

        # Create batched inputs
        x_batched_np = generate_test_data((batch_size, 2, 3), dtype)
        y_batched_np = generate_test_data((batch_size, 2, 3), dtype)

        x_batched_nb = nb.Array.from_numpy(x_batched_np)
        y_batched_nb = nb.Array.from_numpy(y_batched_np)
        x_batched_jax = jnp.array(x_batched_np)
        y_batched_jax = jnp.array(y_batched_np)

        # Apply vmap
        vmapped_nb = nb.vmap(nested_func, in_axes=(0, 0))
        result_nb = vmapped_nb([x_batched_nb, y_batched_nb])

        vmapped_jax = jax.vmap(jax_nested_func, in_axes=(0, 0))
        result_jax = vmapped_jax(x_batched_jax, y_batched_jax)

        assert allclose_recursive(result_nb[0], result_jax, rtol, atol), (
            "vmap nested function mismatch"
        )
