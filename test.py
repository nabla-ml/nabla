import nabla as nb
import jax
import jax.numpy as jnp

if __name__ == "__main__":

    def foo(a, b):
        # return nb.sum(nb.broadcast_to(a, (4, 2, 3)))
        return nb.concatenate([a, b], axis=1)
    
    # JAX version of the same function
    def foo_jax(a, b):
        # return jnp.sum(jnp.broadcast_to(a, (4, 2, 3)))
        return jnp.concatenate([a, b], axis=1)
    
    foo_vmapped = nb.vmap(foo, in_axes=(0, 0))
    foo_jax_vmapped = jax.vmap(foo_jax, in_axes=(0, 0))
    
    a = nb.arange((2, 3, 4))
    b = nb.arange((2, 3, 4)) + 1

    res = foo(a, b)

    print("Nabla result:")
    print(res)
    print("numpy_shape:", res.to_numpy().shape)
    
    res_jax = foo_jax(a.to_numpy(), b.to_numpy())
    print("\nJAX result:")
    print(res_jax)
    print("shape:", res_jax.shape)
    

    assert res.shape == res_jax.shape, "Shapes do not match!"
    # use numpy all close?
    import numpy as np
    assert np.allclose(res.to_numpy(), res_jax), "Results do not match!"

    # now the vmapped stuff 
    res_vmap = foo_vmapped(a, b)
    print("\nNabla vmap result:")
    print(res_vmap)
    print("numpy_shape:", res_vmap.to_numpy().shape)

    res_jax_vmap = foo_jax_vmapped(a.to_numpy(), b.to_numpy())
    print("\nJAX vmap result:")
    print(res_jax_vmap)
    print("shape:", res_jax_vmap.shape)

    assert res_vmap.shape == res_jax_vmap.shape, "Vmap shapes do not match!"
    assert np.allclose(res_vmap.to_numpy(), res_jax_vmap), "Vmap results do not match!"

    print("\n" + "="*50)
    print("TESTING ARRAY SLICE VJP/JVP")
    print("="*50)

    # Test array slice VJP and JVP with different axes and slice patterns
    def test_slice_vjp_jvp(slice_func, slice_func_jax, test_name):
        print(f"\n--- Testing {test_name} ---")
        
        # Create test input
        x = nb.arange((4, 6, 8))
        x_jax = x.to_numpy()
        
        # Test forward pass
        result_nabla = slice_func(x)
        result_jax = slice_func_jax(x_jax)
        
        print(f"Input shape: {x.shape}")
        print(f"Nabla slice result shape: {result_nabla.shape}")
        print(f"JAX slice result shape: {result_jax.shape}")
        
        assert result_nabla.shape == result_jax.shape, f"{test_name}: Forward shapes don't match!"
        assert np.allclose(result_nabla.to_numpy(), result_jax), f"{test_name}: Forward results don't match!"
        
        # Test VJP - create a cotangent with the same shape as the output
        cotangent = nb.ones(result_nabla.shape)
        cotangent_jax = jnp.ones(result_jax.shape)
        
        # Use vjp to get the backward function and apply cotangent
        primals_out, vjp_fun = nb.vjp(slice_func, x)
        vjp_result = vjp_fun(cotangent)
        
        primals_out_jax, vjp_fun_jax = jax.vjp(slice_func_jax, x_jax)
        vjp_result_jax = vjp_fun_jax(cotangent_jax)
        
        print(f"Nabla VJP result shape: {vjp_result[0].shape}")
        print(f"JAX VJP result shape: {vjp_result_jax[0].shape}")
        
        assert vjp_result[0].shape == vjp_result_jax[0].shape, f"{test_name}: VJP shapes don't match!"
        assert np.allclose(vjp_result[0].to_numpy(), vjp_result_jax[0], atol=1e-6), f"{test_name}: VJP results don't match!"
        
        # Test JVP - create a tangent with the same shape as the input
        tangent = nb.ones(x.shape)
        tangent_jax = jnp.ones(x_jax.shape)
        
        # Use jvp to get the forward-mode derivative
        primals_out_jvp, tangents_out = nb.jvp(slice_func, (x,), (tangent,))
        primals_out_jvp_jax, tangents_out_jax = jax.jvp(slice_func_jax, (x_jax,), (tangent_jax,))
        
        print(f"Nabla JVP result shape: {tangents_out.shape}")
        print(f"JAX JVP result shape: {tangents_out_jax.shape}")
        
        assert tangents_out.shape == tangents_out_jax.shape, f"{test_name}: JVP shapes don't match!"
        assert np.allclose(tangents_out.to_numpy(), tangents_out_jax, atol=1e-6), f"{test_name}: JVP results don't match!"
        
        # Test VMAP - create batched version of the function
        vmapped_slice_nabla = nb.vmap(slice_func, in_axes=0)
        vmapped_slice_jax = jax.vmap(slice_func_jax, in_axes=0)
        
        # Create batched input (add batch dimension)
        batched_x = nb.arange((3, 4, 6, 8))  # batch size 3
        batched_x_jax = batched_x.to_numpy()
        
        # Test vmapped forward pass
        vmapped_result_nabla = vmapped_slice_nabla(batched_x)
        vmapped_result_jax = vmapped_slice_jax(batched_x_jax)
        
        print(f"Batched input shape: {batched_x.shape}")
        print(f"Nabla vmapped result shape: {vmapped_result_nabla.shape}")
        print(f"JAX vmapped result shape: {vmapped_result_jax.shape}")
        
        assert vmapped_result_nabla.shape == vmapped_result_jax.shape, f"{test_name}: Vmapped forward shapes don't match!"
        assert np.allclose(vmapped_result_nabla.to_numpy(), vmapped_result_jax), f"{test_name}: Vmapped forward results don't match!"
        
        # Test vmapped VJP
        batched_cotangent = nb.ones(vmapped_result_nabla.shape)
        batched_cotangent_jax = jnp.ones(vmapped_result_jax.shape)
        
        primals_vmap, vjp_fun_vmap = nb.vjp(vmapped_slice_nabla, batched_x)
        vjp_result_vmap = vjp_fun_vmap(batched_cotangent)
        
        primals_vmap_jax, vjp_fun_vmap_jax = jax.vjp(vmapped_slice_jax, batched_x_jax)
        vjp_result_vmap_jax = vjp_fun_vmap_jax(batched_cotangent_jax)
        
        print(f"Nabla vmapped VJP result shape: {vjp_result_vmap[0].shape}")
        print(f"JAX vmapped VJP result shape: {vjp_result_vmap_jax[0].shape}")
        
        assert vjp_result_vmap[0].shape == vjp_result_vmap_jax[0].shape, f"{test_name}: Vmapped VJP shapes don't match!"
        assert np.allclose(vjp_result_vmap[0].to_numpy(), vjp_result_vmap_jax[0], atol=1e-6), f"{test_name}: Vmapped VJP results don't match!"
        
        # Test vmapped JVP
        batched_tangent = nb.ones(batched_x.shape)
        batched_tangent_jax = jnp.ones(batched_x_jax.shape)
        
        primals_jvp_vmap, tangents_jvp_vmap = nb.jvp(vmapped_slice_nabla, (batched_x,), (batched_tangent,))
        primals_jvp_vmap_jax, tangents_jvp_vmap_jax = jax.jvp(vmapped_slice_jax, (batched_x_jax,), (batched_tangent_jax,))
        
        print(f"Nabla vmapped JVP result shape: {tangents_jvp_vmap.shape}")
        print(f"JAX vmapped JVP result shape: {tangents_jvp_vmap_jax.shape}")
        
        assert tangents_jvp_vmap.shape == tangents_jvp_vmap_jax.shape, f"{test_name}: Vmapped JVP shapes don't match!"
        assert np.allclose(tangents_jvp_vmap.to_numpy(), tangents_jvp_vmap_jax, atol=1e-6), f"{test_name}: Vmapped JVP results don't match!"
        
        print(f"✅ {test_name} (including vmap) passed!")

    # Test 1: Simple slice along axis 0
    def slice_axis0(x):
        return x[1:3, :, :]
    def slice_axis0_jax(x):
        return x[1:3, :, :]
    
    test_slice_vjp_jvp(slice_axis0, slice_axis0_jax, "Slice along axis 0")

    # Test 2: Simple slice along axis 1  
    def slice_axis1(x):
        return x[:, 2:5, :]
    def slice_axis1_jax(x):
        return x[:, 2:5, :]
    
    test_slice_vjp_jvp(slice_axis1, slice_axis1_jax, "Slice along axis 1")

    # Test 3: Simple slice along axis 2
    def slice_axis2(x):
        return x[:, :, 1:6]
    def slice_axis2_jax(x):
        return x[:, :, 1:6]
    
    test_slice_vjp_jvp(slice_axis2, slice_axis2_jax, "Slice along axis 2")

    # Test 4: Multi-axis slice
    def slice_multi(x):
        return x[1:3, 2:4, 3:7]
    def slice_multi_jax(x):
        return x[1:3, 2:4, 3:7]
    
    test_slice_vjp_jvp(slice_multi, slice_multi_jax, "Multi-axis slice")

    # Test 5: Slice with negative indices
    def slice_negative(x):
        return x[1:-1, -4:-1, :]
    def slice_negative_jax(x):
        return x[1:-1, -4:-1, :]
    
    test_slice_vjp_jvp(slice_negative, slice_negative_jax, "Slice with negative indices")

    # Test 6: Single element slice
    def slice_single(x):
        return x[2:3, 3:4, 4:5]
    def slice_single_jax(x):
        return x[2:3, 3:4, 4:5]
    
    test_slice_vjp_jvp(slice_single, slice_single_jax, "Single element slice")

    print("\n" + "="*50)
    print("TESTING EDGE CASES")
    print("="*50)

    # Edge Case 1: Empty slice (should create empty tensor)
    def slice_empty(x):
        return x[2:2, :, :]  # Empty slice
    def slice_empty_jax(x):
        return x[2:2, :, :]
    
    test_slice_vjp_jvp(slice_empty, slice_empty_jax, "Empty slice")

    # Edge Case 2: Full slice (equivalent to identity)
    def slice_full(x):
        return x[:, :, :]
    def slice_full_jax(x):
        return x[:, :, :]
    
    test_slice_vjp_jvp(slice_full, slice_full_jax, "Full slice (identity)")

    # Edge Case 3: Out-of-bounds handling (should clamp)
    def slice_oob(x):
        return x[1:100, 2:200, 3:300]  # Beyond array bounds
    def slice_oob_jax(x):
        return x[1:100, 2:200, 3:300]
    
    test_slice_vjp_jvp(slice_oob, slice_oob_jax, "Out-of-bounds slice")

    # Edge Case 4: Mixed positive/negative boundaries (no steps)
    def slice_mixed(x):
        return x[1:-1, -4:4, 2:-2]
    def slice_mixed_jax(x):
        return x[1:-1, -4:4, 2:-2]
    
    test_slice_vjp_jvp(slice_mixed, slice_mixed_jax, "Mixed positive/negative boundaries")

    # Edge Case 5: Boundary edge cases
    def slice_boundary(x):
        return x[0:1, -1:, :]  # First element and last element
    def slice_boundary_jax(x):
        return x[0:1, -1:, :]
    
    test_slice_vjp_jvp(slice_boundary, slice_boundary_jax, "Boundary slice")

    # Edge Case 6: Large ranges within bounds
    def slice_large_range(x):
        return x[0:4, 1:5, 0:8]  # Almost full ranges
    def slice_large_range_jax(x):
        return x[0:4, 1:5, 0:8]
    
    test_slice_vjp_jvp(slice_large_range, slice_large_range_jax, "Large range slice")

    # Edge Case 7: Zero-based indexing edge case
    def slice_zero_based(x):
        return x[0:2, 0:3, 0:4]  # Start from zero
    def slice_zero_based_jax(x):
        return x[0:2, 0:3, 0:4]
    
    test_slice_vjp_jvp(slice_zero_based, slice_zero_based_jax, "Zero-based slice")

    # Edge Case 8: Negative indexing from end
    def slice_negative_end(x):
        return x[-2:, -3:, -4:]  # From near end to end
    def slice_negative_end_jax(x):
        return x[-2:, -3:, -4:]
    
    test_slice_vjp_jvp(slice_negative_end, slice_negative_end_jax, "Negative end slice")

    # Edge Case 9: Asymmetric slicing
    def slice_asymmetric(x):
        return x[1:4, 0:6, 2:6]  # Different slice sizes
    def slice_asymmetric_jax(x):
        return x[1:4, 0:6, 2:6]
    
    test_slice_vjp_jvp(slice_asymmetric, slice_asymmetric_jax, "Asymmetric slice")

    # Edge Case 10: Adjacent slices
    def slice_adjacent(x):
        return x[1:2, 2:3, 3:4]  # Single elements adjacent to previous test
    def slice_adjacent_jax(x):
        return x[1:2, 2:3, 3:4]
    
    test_slice_vjp_jvp(slice_adjacent, slice_adjacent_jax, "Adjacent single element slice")

    # Edge Case 11: Test with 1D array
    print(f"\n--- Testing 1D Array Edge Cases ---")
    
    def test_1d_slice_vjp_jvp(slice_func, slice_func_jax, test_name):
        print(f"\n--- Testing 1D: {test_name} ---")
        
        # Create 1D test input
        x = nb.arange((10,))
        x_jax = x.to_numpy()
        
        # Test forward pass
        result_nabla = slice_func(x)
        result_jax = slice_func_jax(x_jax)
        
        print(f"1D Input shape: {x.shape}")
        print(f"Nabla 1D result shape: {result_nabla.shape}")
        print(f"JAX 1D result shape: {result_jax.shape}")
        
        assert result_nabla.shape == result_jax.shape, f"{test_name}: 1D Forward shapes don't match!"
        assert np.allclose(result_nabla.to_numpy(), result_jax), f"{test_name}: 1D Forward results don't match!"
        
        # Test VJP
        cotangent = nb.ones(result_nabla.shape)
        cotangent_jax = jnp.ones(result_jax.shape)
        
        primals_out, vjp_fun = nb.vjp(slice_func, x)
        vjp_result = vjp_fun(cotangent)
        
        primals_out_jax, vjp_fun_jax = jax.vjp(slice_func_jax, x_jax)
        vjp_result_jax = vjp_fun_jax(cotangent_jax)
        
        assert vjp_result[0].shape == vjp_result_jax[0].shape, f"{test_name}: 1D VJP shapes don't match!"
        assert np.allclose(vjp_result[0].to_numpy(), vjp_result_jax[0], atol=1e-6), f"{test_name}: 1D VJP results don't match!"
        
        print(f"✅ 1D {test_name} passed!")

    # 1D Edge cases
    def slice_1d_basic(x):
        return x[2:8]
    def slice_1d_basic_jax(x):
        return x[2:8]
    
    test_1d_slice_vjp_jvp(slice_1d_basic, slice_1d_basic_jax, "Basic 1D slice")

    def slice_1d_negative(x):
        return x[-5:-1]
    def slice_1d_negative_jax(x):
        return x[-5:-1]
    
    test_1d_slice_vjp_jvp(slice_1d_negative, slice_1d_negative_jax, "1D negative indices")

    def slice_1d_full(x):
        return x[:]
    def slice_1d_full_jax(x):
        return x[:]
    
    test_1d_slice_vjp_jvp(slice_1d_full, slice_1d_full_jax, "1D full slice")

    def slice_1d_partial(x):
        return x[3:7]
    def slice_1d_partial_jax(x):
        return x[3:7]
    
    test_1d_slice_vjp_jvp(slice_1d_partial, slice_1d_partial_jax, "1D partial slice")

    # Edge Case 12: Test with very small arrays
    print(f"\n--- Testing Small Array Edge Cases ---")
    
    def test_small_slice_vjp_jvp(slice_func, slice_func_jax, test_name, shape):
        print(f"\n--- Testing Small Array ({shape}): {test_name} ---")
        
        # Create small test input
        x = nb.arange(shape)
        x_jax = x.to_numpy()
        
        # Test forward pass
        result_nabla = slice_func(x)
        result_jax = slice_func_jax(x_jax)
        
        print(f"Small input shape: {x.shape}")
        print(f"Nabla small result shape: {result_nabla.shape}")
        print(f"JAX small result shape: {result_jax.shape}")
        
        assert result_nabla.shape == result_jax.shape, f"{test_name}: Small forward shapes don't match!"
        assert np.allclose(result_nabla.to_numpy(), result_jax), f"{test_name}: Small forward results don't match!"
        
        # Test VJP
        cotangent = nb.ones(result_nabla.shape)
        cotangent_jax = jnp.ones(result_jax.shape)
        
        primals_out, vjp_fun = nb.vjp(slice_func, x)
        vjp_result = vjp_fun(cotangent)
        
        primals_out_jax, vjp_fun_jax = jax.vjp(slice_func_jax, x_jax)
        vjp_result_jax = vjp_fun_jax(cotangent_jax)
        
        assert vjp_result[0].shape == vjp_result_jax[0].shape, f"{test_name}: Small VJP shapes don't match!"
        assert np.allclose(vjp_result[0].to_numpy(), vjp_result_jax[0], atol=1e-6), f"{test_name}: Small VJP results don't match!"
        
        print(f"✅ Small Array {test_name} passed!")

    # Small array tests
    def slice_2x2(x):
        return x[0:1, 1:2]
    def slice_2x2_jax(x):
        return x[0:1, 1:2]
    
    test_small_slice_vjp_jvp(slice_2x2, slice_2x2_jax, "2x2 slice", (2, 2))

    def slice_1x3(x):
        return x[:, 1:3]
    def slice_1x3_jax(x):
        return x[:, 1:3]
    
    test_small_slice_vjp_jvp(slice_1x3, slice_1x3_jax, "1x3 basic slice", (1, 3))

    # Edge Case 13: Minimal slices
    def slice_minimal(x):
        return x[0:1, 0:1, 0:1]  # Extract corner element
    def slice_minimal_jax(x):
        return x[0:1, 0:1, 0:1]
    
    test_slice_vjp_jvp(slice_minimal, slice_minimal_jax, "Minimal corner slice")

    # Edge Case 14: Maximum valid slices
    def slice_max_valid(x):
        return x[0:4, 0:6, 0:8]  # Extract everything
    def slice_max_valid_jax(x):
        return x[0:4, 0:6, 0:8]
    
    test_slice_vjp_jvp(slice_max_valid, slice_max_valid_jax, "Maximum valid slice")

    # Edge Case 15: Interior slicing
    def slice_interior(x):
        return x[1:3, 1:5, 1:7]  # Interior rectangle
    def slice_interior_jax(x):
        return x[1:3, 1:5, 1:7]
    
    test_slice_vjp_jvp(slice_interior, slice_interior_jax, "Interior slice")

    # Edge Case 16: Corner and edge slices
    def slice_corner(x):
        return x[-2:, -2:, -2:]  # Bottom-right corner
    def slice_corner_jax(x):
        return x[-2:, -2:, -2:]
    
    test_slice_vjp_jvp(slice_corner, slice_corner_jax, "Corner slice")

    def slice_edge(x):
        return x[:1, :, :]  # Top edge
    def slice_edge_jax(x):
        return x[:1, :, :]
    
    test_slice_vjp_jvp(slice_edge, slice_edge_jax, "Edge slice")

    print(f"\n🎉 All array slice VJP/JVP/VMAP tests (including edge cases) passed!")

    # jacobian = nb.jacrev(foo, argnums=0)
    # print(nb.xpr(jacobian, a))
    # print("\nNabla result:")
    # j = jacobian(a)
    # print(j)

    # # try the same with jax using the JAX-compatible function
    # jax_jacobian = jax.jacrev(foo_jax, argnums=0)
    # jax_result = jax_jacobian(a.to_numpy())
    # print(jax_result)
    # print("\nJAX result:")
    # print(jax_result)
    # print("shape:", jax_result.shape)

