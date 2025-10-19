# import numpy as np
# import pytest

# import nabla as nb


# def test_broadcast_to_vjp():
#     """Test VJP with broadcast_to operation."""

#     def foo(x):
#         x = nb.broadcast_to(x, (4, 2, 3))
#         return x

#     x = nb.ndarange((2, 3))

#     values, vjp_fn = nb.vjp(foo, x)
#     cotangent = nb.ones_like(values)
#     grad = vjp_fn(cotangent)

#     # Verify the gradient has the correct shape (should match input)
#     assert grad.shape == x.shape, f"Expected grad shape {x.shape}, got {grad.shape}"

#     # Verify the gradient values (should be summed across broadcasted dimensions)
#     expected_grad = nb.ones_like(x) * 4  # 4 copies were made
#     assert np.allclose(grad.to_numpy(), expected_grad.to_numpy()), (
#         "Gradient values don't match expected"
#     )


# def test_vmap_complex_operations():
#     """Test vmap with complex dimension manipulation operations."""

#     def foo(x):
#         x = nb.broadcast_to(x, (4, 2, 3))
#         x = nb.incr_batch_dim_ctr(x)
#         x = nb.decr_batch_dim_ctr(x)
#         x = nb.unsqueeze(x, [-2])
#         x = nb.squeeze(x, [-2])
#         x = nb.unsqueeze_batch_dims(x, [-2])
#         x = nb.squeeze_batch_dims(x, [-2])
#         return x

#     foo_vmapped = nb.vmap(nb.vmap(foo))

#     x = nb.ndarange((2, 3))

#     # Test that the operation completes without error
#     try:
#         result = foo_vmapped(x)
#         # Basic shape check - should maintain input shape through transformations
#         assert result.shape[0] == x.shape[0], "Batch dimension preserved incorrectly"
#         assert result.shape[1] == x.shape[1], "Batch dimension preserved incorrectly"
#     except Exception as e:
#         pytest.fail(f"Vmap operations failed: {e}")


# def test_std_basis_function():
#     """Test the _std_basis function for creating standard basis vectors."""
#     a = nb.ndarange((3, 2))
#     b = nb.ndarange((2, 4))
#     c = nb.ndarange((4, 1))

#     sizes, tangents = nb.core.trafos._std_basis([a, b, c])

#     # Verify we get correct number of sizes and tangents
#     assert len(sizes) == 3, "Should have 3 sizes for 3 input tensors"
#     assert len(tangents) == sum(sizes), "Total tangents should equal sum of sizes"

#     # Verify sizes match tensor shapes
#     expected_sizes = [a.size, b.size, c.size]
#     assert sizes == expected_sizes, f"Expected sizes {expected_sizes}, got {sizes}"

#     # Verify tangent shapes match input shapes
#     tangent_idx = 0
#     for i, arr in enumerate([a, b, c]):
#         for j in range(sizes[i]):
#             tangent = tangents[tangent_idx + j]
#             assert tangent.shape == arr.shape, (
#                 f"Tangent {tangent_idx + j} shape mismatch"
#             )
#         tangent_idx += sizes[i]


# if __name__ == "__main__":
#     """Run all tests when executed as script."""
#     print("=== Basic Nabla Function Tests ===")
#     test_broadcast_to_vjp()
#     test_vmap_complex_operations()
#     test_std_basis_function()
#     print("\n🎉 All basic function tests passed!")
