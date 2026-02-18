"""Micro gate test: scalar Hessians must be correct before any broader tests.

Rung 1 of the incremental ladder from PHYSICAL_OPS_HESSIAN_HANDOVER.md.
"""
import numpy as np
import nabla as nb

def test_x_cubed():
    """f(x) = x^3 → H = 6x. At x=2, H=12."""
    x = nb.Tensor.from_dlpack(np.array([2.0], dtype=np.float32))
    expected = np.array([[12.0]], dtype=np.float32)

    modes = {
        "rev_rev": nb.jacrev(nb.jacrev(lambda x: nb.reduce_sum(x * x * x))),
        "fwd_fwd": nb.jacfwd(nb.jacfwd(lambda x: nb.reduce_sum(x * x * x))),
        "rev_fwd": nb.jacrev(nb.jacfwd(lambda x: nb.reduce_sum(x * x * x))),
        "fwd_rev": nb.jacfwd(nb.jacrev(lambda x: nb.reduce_sum(x * x * x))),
    }

    for mode_name, hess_fn in modes.items():
        try:
            H = hess_fn(x)
            val = H.to_numpy()
            ok = np.allclose(val, expected, atol=1e-3)
            print(f"  {mode_name}: val={val.flatten()} expected={expected.flatten()} {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            print(f"  {mode_name}: ERROR: {type(e).__name__}: {e}")


def test_exp_sum():
    """f(x) = sum(exp(x)) → H = diag(exp(x))."""
    x_np = np.array([0.5, 1.0, -0.5], dtype=np.float32)
    x = nb.Tensor.from_dlpack(x_np.copy())
    expected = np.diag(np.exp(x_np))

    modes = {
        "rev_rev": nb.jacrev(nb.jacrev(lambda x: nb.reduce_sum(nb.exp(x)))),
        "fwd_fwd": nb.jacfwd(nb.jacfwd(lambda x: nb.reduce_sum(nb.exp(x)))),
        "rev_fwd": nb.jacrev(nb.jacfwd(lambda x: nb.reduce_sum(nb.exp(x)))),
        "fwd_rev": nb.jacfwd(nb.jacrev(lambda x: nb.reduce_sum(nb.exp(x)))),
    }

    for mode_name, hess_fn in modes.items():
        try:
            H = hess_fn(x)
            val = H.to_numpy()
            ok = np.allclose(val, expected, atol=1e-3)
            print(f"  {mode_name}: max_err={np.max(np.abs(val - expected)):.6f} {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            print(f"  {mode_name}: ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=== Test x^3 Hessian (scalar) ===")
    test_x_cubed()
    print("\n=== Test sum(exp(x)) Hessian (vector) ===")
    test_exp_sum()
