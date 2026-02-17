import numpy as np

import nabla as nb
from nabla import ops


def test_slice_negative_indices():
    print("\n--- Test Negative Indices ---")
    shape = (4, 4)
    x_np = np.arange(16).reshape(shape).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)

    # Case 1: Slice last row
    # numpy: x[-1, :]
    print("Testing slice start=(-1, 0), size=(1, 4)")
    try:
        y = ops.slice_tensor(x, start=(-1, 0), size=(1, 4))
        y_val = y.numpy()
        print("Result shape:", y_val.shape)
        np.testing.assert_allclose(y_val, x_np[-1:, :], atol=1e-5)
        print("✅ Negative index -1 works")
    except Exception as e:
        print(f"❌ Negative index failed: {e}")

    # Case 2: Slice crossing? Not regular slice, but standard neg index
    # start = -2, size = 2 (should get last 2 rows)
    print("Testing slice start=(-2, 0), size=(2, 4)")
    try:
        y = ops.slice_tensor(x, start=(-2, 0), size=(2, 4))
        y_val = y.numpy()
        np.testing.assert_allclose(y_val, x_np[-2:, :], atol=1e-5)
        print("✅ Negative crossing works")
    except Exception as e:
        print(f"❌ Negative crossing failed: {e}")


def test_slice_update_vjp():
    print("\n--- Test Slice Update VJP ---")
    # y = slice_update(x, u, start, size)
    # L = sum(y)
    # dL/dx should be 1s everywhere EXCEPT slices (where it is 0)
    # dL/du should be 1s everywhere

    shape = (4, 4)
    x = nb.Tensor.from_dlpack(np.zeros(shape, dtype=np.float32))
    u = nb.Tensor.from_dlpack(np.ones((2, 2), dtype=np.float32))

    start = (1, 1)
    size = (2, 2)

    def loss_fn(x, u):
        y = ops.slice_update(x, u, start=start, size=size)
        return ops.reduce_sum(y)

    try:
        from nabla.core.autograd import grad

        g_fn = grad(loss_fn, argnums=(0, 1))
        dx, du = g_fn(x, u)

        dx_val = dx.numpy()
        du_val = du.numpy()

        print("dx shape:", dx_val.shape)
        print("du shape:", du_val.shape)

        # Expected dx: 1s everywhere, 0s at [1:3, 1:3]
        expected_dx = np.ones(shape, dtype=np.float32)
        expected_dx[1:3, 1:3] = 0.0

        # Expected du: 1s everywhere
        expected_du = np.ones((2, 2), dtype=np.float32)

        np.testing.assert_allclose(dx_val, expected_dx, atol=1e-5)
        print("✅ dx Correct")
        np.testing.assert_allclose(du_val, expected_du, atol=1e-5)
        print("✅ du Correct")

    except NotImplementedError:
        print("❌ SliceUpdate VJP not implemented")
    except Exception as e:
        print(f"❌ SliceUpdate VJP failed: {e}")


if __name__ == "__main__":
    test_slice_negative_indices()
    test_slice_update_vjp()
