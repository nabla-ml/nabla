import numpy as np

import nabla as nb
from nabla import ops


def test_slice_update_correctness():
    print("\n--- Test Slice Update Correctness ---")

    shape = (4, 4)
    x_np = np.random.randn(*shape).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)

    update_shape = (2, 2)
    u_np = np.ones(update_shape).astype(np.float32) * 99.0
    u = nb.Tensor.from_dlpack(u_np)

    start = (1, 1)
    size = (2, 2)

    # Nabla
    y = ops.slice_update(x, u, start=start, size=size)
    y_val = y.numpy()

    # Reference
    y_ref = x_np.copy()
    y_ref[1:3, 1:3] = u_np

    print("Nabla Result:\n", y_val)
    print("Ref Result:\n", y_ref)

    np.testing.assert_allclose(y_val, y_ref, atol=1e-5)
    print("✅ Forward Correctness Passed")


def test_slice_tensor_grad_check():
    """Test that slice_tensor VJP (which uses slice_update) is correct."""
    print("\n--- Test Slice Tensor VJP (via slice_update) ---")

    shape = (5, 5)
    x_np = np.random.randn(*shape).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)
    x.trace()

    start = (1, 1)
    size = (2, 2)

    def func(t):
        return ops.slice_tensor(t, start=start, size=size)

    # Cotangent
    cot_np = np.ones(size).astype(np.float32)
    cot = nb.Tensor.from_dlpack(cot_np)

    # Expected VJP result
    expected_grad = np.zeros(shape).astype(np.float32)
    expected_grad[1:3, 1:3] = 1.0

    # backward is not easily available for loose tensors without context.
    # Use nabla.grad instead.

    def loss(t):
        out = func(t)
        # projected loss to match cotangent direction
        # Backprop ones: out * ones -> sum
        # But we want specific cotangent?
        # grad(f)(x) computes vjp(f, x, 1.0).
        # To compute vjp(f, x, u), we compute grad(lambda x: sum(f(x) * u)).

        # Element-wise multiply with cotangent and sum
        return nb.ops.reduce_sum(out * cot)

    grad_func = nb.grad(loss)
    grad = grad_func(x).numpy()

    print("Calculated Grad:\n", grad)
    print("Expected Grad:\n", expected_grad)

    np.testing.assert_allclose(grad, expected_grad, atol=1e-5)
    print("✅ VJP Correctness Passed")


def test_slice_update_batched():
    print("\n--- Test Batched Slice Update (vmap) ---")

    B = 2
    shape = (4, 4)
    x_np = np.random.randn(B, *shape).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)

    update_shape = (2, 2)
    u_np = np.ones((B, *update_shape)).astype(np.float32) * 5.0
    u = nb.Tensor.from_dlpack(u_np)

    start = (1, 1)
    size = (2, 2)

    # We want to use slice_update on each batch element
    # slice_update(x[b], u[b], start, size)

    def func(x_b, u_b):
        return ops.slice_update(x_b, u_b, start=start, size=size)

    # vmap over batch axis 0 for x and u
    vmapped_func = nb.vmap(func, in_axes=(0, 0))
    y = vmapped_func(x, u)
    y_val = y.numpy()

    # Reference
    y_ref = x_np.copy()
    for b in range(B):
        y_ref[b, 1:3, 1:3] = u_np[b]

    print("Nabla Batched Result shape:", y_val.shape)

    np.testing.assert_allclose(y_val, y_ref, atol=1e-5)
    print("✅ Batched Slice Update Passed")


if __name__ == "__main__":
    test_slice_update_correctness()
    test_slice_tensor_grad_check()
    test_slice_update_batched()
