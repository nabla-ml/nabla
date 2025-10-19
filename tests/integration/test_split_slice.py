import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb


def test_concatenate_basic():
    """Test basic concatenate operation and vmap."""

    def foo(a, b):
        return nb.concatenate([a, b], axis=0)

    # JAX version of the same function
    def foo_jax(a, b):
        return jnp.concatenate([a, b], axis=0)

    foo_vmapped = nb.vmap(foo, in_axes=(0, 0))
    foo_jax_vmapped = jax.vmap(foo_jax, in_axes=(0, 0))

    a = nb.ndarange((2, 3, 4))
    b = nb.ndarange((2, 3, 4)) + 1

    res = foo(a, b)

    res_jax = foo_jax(a.to_numpy(), b.to_numpy())

    assert res.shape == res_jax.shape, "Shapes do not match!"
    assert np.allclose(res.to_numpy(), res_jax), "Results do not match!"

    # Test vmapped version
    res_vmap = foo_vmapped(a, b)
    res_jax_vmap = foo_jax_vmapped(a.to_numpy(), b.to_numpy())

    assert res_vmap.shape == res_jax_vmap.shape, "Vmap shapes do not match!"
    assert np.allclose(res_vmap.to_numpy(), res_jax_vmap), "Vmap results do not match!"


def test_tensor_slice_comprehensive():
    """Test comprehensive tensor slice VJP/JVP operations."""

    def test_slice_vjp_jvp(slice_func, slice_func_jax, test_name):
        """Helper function to test slice operations."""
        # Create test input
        x = nb.ndarange((4, 6, 8))
        x_jax = x.to_numpy()

        # Test forward pass
        result_nabla = slice_func(x)
        result_jax = slice_func_jax(x_jax)

        assert result_nabla.shape == result_jax.shape, (
            f"{test_name}: Forward shapes don't match!"
        )
        assert np.allclose(result_nabla.to_numpy(), result_jax), (
            f"{test_name}: Forward results don't match!"
        )

        # Test VJP - create a cotangent with the same shape as the output
        cotangent = nb.ones(result_nabla.shape, dtype=result_nabla.dtype)
        cotangent_jax = jnp.ones(result_jax.shape, dtype=result_jax.dtype)

        # Use vjp to get the backward function and apply cotangent
        primals_out, vjp_fun = nb.vjp(slice_func, x)
        vjp_result = vjp_fun(cotangent)

        primals_out_jax, vjp_fun_jax = jax.vjp(slice_func_jax, x_jax)
        vjp_result_jax = vjp_fun_jax(cotangent_jax)

        # Handle nabla's natural VJP structure (single arg returns gradient directly)
        vjp_result_nabla = vjp_result  # nabla returns gradient directly
        vjp_result_jax_unwrapped = vjp_result_jax[0]  # JAX returns tuple

        assert vjp_result_nabla.shape == vjp_result_jax_unwrapped.shape, (
            f"{test_name}: VJP shapes don't match!"
        )
        assert np.allclose(
            vjp_result_nabla.to_numpy(), vjp_result_jax_unwrapped, atol=1e-6
        ), f"{test_name}: VJP results don't match!"

        # Test JVP - create a tangent with the same shape as the input
        tangent = nb.ones(x.shape)
        tangent_jax = jnp.ones(x_jax.shape, dtype=x_jax.dtype)

        # Use jvp to get the forward-mode derivative
        primals_out_jvp, tangents_out = nb.jvp(slice_func, (x,), (tangent,))
        primals_out_jvp_jax, tangents_out_jax = jax.jvp(
            slice_func_jax, (x_jax,), (tangent_jax,)
        )

        assert tangents_out.shape == tangents_out_jax.shape, (
            f"{test_name}: JVP shapes don't match!"
        )
        assert np.allclose(tangents_out.to_numpy(), tangents_out_jax, atol=1e-6), (
            f"{test_name}: JVP results don't match!"
        )

        # Test VMAP - create batched version of the function
        vmapped_slice_nabla = nb.vmap(slice_func, in_axes=0)
        vmapped_slice_jax = jax.vmap(slice_func_jax, in_axes=0)

        # Create batched input (add batch dimension)
        batched_x = nb.ndarange((3, 4, 6, 8))  # batch size 3
        batched_x_jax = batched_x.to_numpy()

        # Test vmapped forward pass
        vmapped_result_nabla = vmapped_slice_nabla(batched_x)
        vmapped_result_jax = vmapped_slice_jax(batched_x_jax)

        assert vmapped_result_nabla.shape == vmapped_result_jax.shape, (
            f"{test_name}: Vmapped forward shapes don't match!"
        )
        assert np.allclose(vmapped_result_nabla.to_numpy(), vmapped_result_jax), (
            f"{test_name}: Vmapped forward results don't match!"
        )

        # Test vmapped VJP
        batched_cotangent = nb.ones(
            vmapped_result_nabla.shape, dtype=vmapped_result_nabla.dtype
        )
        batched_cotangent_jax = jnp.ones(
            vmapped_result_jax.shape, dtype=vmapped_result_jax.dtype
        )

        primals_vmap, vjp_fun_vmap = nb.vjp(vmapped_slice_nabla, batched_x)
        vjp_result_vmap = vjp_fun_vmap(batched_cotangent)

        primals_vmap_jax, vjp_fun_vmap_jax = jax.vjp(vmapped_slice_jax, batched_x_jax)
        vjp_result_vmap_jax = vjp_fun_vmap_jax(batched_cotangent_jax)

        # Handle nabla's natural VJP structure (single arg returns gradient directly)
        vjp_result_vmap_nabla = vjp_result_vmap  # nabla returns gradient directly
        vjp_result_vmap_jax_unwrapped = vjp_result_vmap_jax[0]  # JAX returns tuple

        assert vjp_result_vmap_nabla.shape == vjp_result_vmap_jax_unwrapped.shape, (
            f"{test_name}: Vmapped VJP shapes don't match!"
        )
        assert np.allclose(
            vjp_result_vmap_nabla.to_numpy(), vjp_result_vmap_jax_unwrapped, atol=1e-6
        ), f"{test_name}: Vmapped VJP results don't match!"

        # Test vmapped JVP
        batched_tangent = nb.ones(batched_x.shape)
        batched_tangent_jax = jnp.ones(batched_x_jax.shape, dtype=batched_x_jax.dtype)

        primals_jvp_vmap, tangents_jvp_vmap = nb.jvp(
            vmapped_slice_nabla, (batched_x,), (batched_tangent,)
        )
        primals_jvp_vmap_jax, tangents_jvp_vmap_jax = jax.jvp(
            vmapped_slice_jax, (batched_x_jax,), (batched_tangent_jax,)
        )

        assert tangents_jvp_vmap.shape == tangents_jvp_vmap_jax.shape, (
            f"{test_name}: Vmapped JVP shapes don't match!"
        )
        assert np.allclose(
            tangents_jvp_vmap.to_numpy(), tangents_jvp_vmap_jax, atol=1e-6
        ), f"{test_name}: Vmapped JVP results don't match!"

    # Test various slice patterns (excluding stepped slicing which is not yet supported)
    slice_test_cases = [
        (lambda x: x[1:3, :, :], lambda x: x[1:3, :, :], "Slice along axis 0"),
        (lambda x: x[:, 2:5, :], lambda x: x[:, 2:5, :], "Slice along axis 1"),
        (lambda x: x[:, :, 1:6], lambda x: x[:, :, 1:6], "Slice along axis 2"),
        (lambda x: x[1:3, 2:4, 3:7], lambda x: x[1:3, 2:4, 3:7], "Multi-axis slice"),
        (lambda x: x[1:-1, -4:-1, :], lambda x: x[1:-1, -4:-1, :], "Negative indices"),
        # Skip stepped slicing cases - not yet supported in VJP
        # (lambda x: x[::2, :, :], lambda x: x[::2, :, :], "Step slice"),
        # (lambda x: x[:, ::2, ::3], lambda x: x[:, ::2, ::3], "Multi-step slice"),
        # (lambda x: x[::-1, :, :], lambda x: x[::-1, :, :], "Reverse slice"),
        (lambda x: x[1, :, :], lambda x: x[1, :, :], "Single index"),
        (lambda x: x[:, 1, :], lambda x: x[:, 1, :], "Single index axis 1"),
        (lambda x: x[:, :, 1], lambda x: x[:, :, 1], "Single index axis 2"),
        (lambda x: x[1:3, 1:5, 1:7], lambda x: x[1:3, 1:5, 1:7], "Interior slice"),
        (lambda x: x[-2:, -2:, -2:], lambda x: x[-2:, -2:, -2:], "Corner slice"),
        (lambda x: x[:1, :, :], lambda x: x[:1, :, :], "Edge slice"),
    ]

    for slice_func, slice_func_jax, test_name in slice_test_cases:
        test_slice_vjp_jvp(slice_func, slice_func_jax, test_name)


def test_1d_tensor_slicing():
    """Test 1D tensor slicing operations."""

    def test_1d_slice_vjp_jvp(slice_func, slice_func_jax, test_name):
        """Helper function for 1D slice testing."""
        # Create 1D test input
        x = nb.ndarange((10,))
        x_jax = x.to_numpy()

        # Test forward pass
        result_nabla = slice_func(x)
        result_jax = slice_func_jax(x_jax)

        # Now that nabla matches JAX behavior, no shape conversion needed
        assert result_nabla.shape == result_jax.shape, (
            f"1D {test_name}: Forward shapes don't match!"
        )
        assert np.allclose(result_nabla.to_numpy(), result_jax), (
            f"1D {test_name}: Forward results don't match!"
        )

        # Test VJP
        cotangent = nb.ones(result_nabla.shape, dtype=result_nabla.dtype)
        cotangent_jax = jnp.ones(result_jax.shape, dtype=result_jax.dtype)

        primals_out, vjp_fun = nb.vjp(slice_func, x)
        vjp_result = vjp_fun(cotangent)

        primals_out_jax, vjp_fun_jax = jax.vjp(slice_func_jax, x_jax)
        vjp_result_jax = vjp_fun_jax(cotangent_jax)

        # Handle nabla's natural VJP structure (single arg returns gradient directly)
        vjp_result_nabla = vjp_result  # nabla returns gradient directly
        vjp_result_jax_unwrapped = vjp_result_jax[0]  # JAX returns tuple

        assert vjp_result_nabla.shape == vjp_result_jax_unwrapped.shape, (
            f"1D {test_name}: VJP shapes don't match!"
        )
        assert np.allclose(
            vjp_result_nabla.to_numpy(), vjp_result_jax_unwrapped, atol=1e-6
        ), f"1D {test_name}: VJP results don't match!"

    # Test various 1D slice patterns (skip stepped slicing for now due to VJP limitation)
    slice_1d_cases = [
        (lambda x: x[2:7], lambda x: x[2:7], "Basic 1D slice"),
        # (lambda x: x[::2], lambda x: x[::2], "Step 1D slice"),  # Skip: stepped slicing VJP not supported
        # (lambda x: x[::-1], lambda x: x[::-1], "Reverse 1D slice"),  # Skip: stepped slicing VJP not supported
        (lambda x: x[-3:], lambda x: x[-3:], "Negative start 1D"),
        (lambda x: x[5], lambda x: x[5], "Single element 1D"),
    ]

    for slice_func, slice_func_jax, test_name in slice_1d_cases:
        test_1d_slice_vjp_jvp(slice_func, slice_func_jax, test_name)


def test_small_tensor_slicing():
    """Test slicing on small tensors."""

    def test_small_slice_vjp_jvp(slice_func, slice_func_jax, test_name, shape):
        """Helper function for small tensor slice testing."""
        # Create small test input
        x = nb.ndarange(shape)
        x_jax = x.to_numpy()

        # Test forward pass
        result_nabla = slice_func(x)
        result_jax = slice_func_jax(x_jax)

        assert result_nabla.shape == result_jax.shape, (
            f"Small {test_name}: Forward shapes don't match!"
        )
        assert np.allclose(result_nabla.to_numpy(), result_jax), (
            f"Small {test_name}: Forward results don't match!"
        )

        # Test VJP
        cotangent = nb.ones(result_nabla.shape, dtype=result_nabla.dtype)
        cotangent_jax = jnp.ones(result_jax.shape, dtype=result_jax.dtype)

        primals_out, vjp_fun = nb.vjp(slice_func, x)
        vjp_result = vjp_fun(cotangent)

        primals_out_jax, vjp_fun_jax = jax.vjp(slice_func_jax, x_jax)
        vjp_result_jax = vjp_fun_jax(cotangent_jax)

        # Handle nabla's natural VJP structure (single arg returns gradient directly)
        vjp_result_nabla = vjp_result  # nabla returns gradient directly
        vjp_result_jax_unwrapped = vjp_result_jax[0]  # JAX returns tuple

        assert vjp_result_nabla.shape == vjp_result_jax_unwrapped.shape, (
            f"Small {test_name}: VJP shapes don't match!"
        )
        assert np.allclose(
            vjp_result_nabla.to_numpy(), vjp_result_jax_unwrapped, atol=1e-6
        ), f"Small {test_name}: VJP results don't match!"

    # Test with small tensors
    small_test_cases = [
        ((lambda x: x[0:1, 0:1], lambda x: x[0:1, 0:1], "Single element 2D", (2, 2))),
        ((lambda x: x[:, 0], lambda x: x[:, 0], "Column slice", (3, 2))),
        ((lambda x: x[0, :], lambda x: x[0, :], "Row slice", (2, 3))),
    ]

    for slice_func, slice_func_jax, test_name, shape in small_test_cases:
        test_small_slice_vjp_jvp(slice_func, slice_func_jax, test_name, shape)


if __name__ == "__main__":
    """Run all slice tests when executed as script."""
    print("=== Tensor Slice Tests ===")
    test_concatenate_basic()
    test_tensor_slice_comprehensive()
    test_1d_tensor_slicing()
    test_small_tensor_slicing()
    print("\n🎉 All tensor slice VJP/JVP/VMAP tests passed!")
