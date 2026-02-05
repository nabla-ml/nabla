# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec


def test_sharded_mlp_grads():
    """Test that a simple MLP produces correctly sharded gradients."""
    mesh = DeviceMesh("dgx", (2,), ("tp",))

    # 1. Weights: Column Parallel (split axis 1)
    W_data = np.random.randn(8, 8).astype(np.float32)
    W = nb.ops.shard(
        nb.Tensor.from_dlpack(W_data), mesh, [DimSpec([]), DimSpec(["tp"])]
    )

    # 2. Inputs: Replicated
    X_data = np.random.randn(4, 8).astype(np.float32)
    X = nb.Tensor.from_dlpack(X_data)
    # Explicitly mark X as replicated on dgx to match grad sharding expectation exactly
    X = nb.ops.shard(X, mesh, [DimSpec([]), DimSpec([])])

    def mlp_step(x, w):
        # Linear layer: [4, 8] @ [8, 8] -> [4, 8]
        # x is (tp, *), w is (*, tp)
        # Propagation will likely choose shard(x, tp) if not already,
        # but here x is already sharded on tp (axis 0), and w on tp (axis 1).
        # This is essentially a "Mega-Batch" matmul or something similar depending on solver.
        h = nb.matmul(x, w)
        # Reduce to scalar for simplicity
        return nb.reduce_sum(h, axis=0)

    print("\n--- MLP Trace ---")
    t = trace(mlp_step, X, W)
    print(t)

    # Compute grads
    cot = nb.ops.full_like(t.outputs, 1.0)
    grads = backward_on_trace(t, cot)

    grad_W = grads[W]
    grad_X = grads[X]

    # VERIFICATION

    print(f"\nResults:")
    print(f"grad_W sharding: {grad_W.sharding}")
    print(f"grad_X sharding: {grad_X.sharding}")

    # 1. Check Sharding Alignment
    from nabla.core.sharding.spec import needs_reshard

    assert not needs_reshard(
        grad_W.sharding, W.sharding
    ), f"W grad sharding mismatch: {grad_W.sharding} vs {W.sharding}"
    assert not needs_reshard(
        grad_X.sharding, X.sharding
    ), f"X grad sharding mismatch: {grad_X.sharding} vs {X.sharding}"

    # 2. Check Values (compared to replicated execution)
    def expected_fn(x, w):
        h = nb.matmul(x, w)
        return nb.reduce_sum(h, axis=0)

    # Replicated version
    W_rep = nb.Tensor.from_dlpack(W_data)
    X_rep = nb.Tensor.from_dlpack(X_data)
    t_rep = trace(expected_fn, X_rep, W_rep)
    cot_rep = nb.ops.full_like(t_rep.outputs, 1.0)
    grads_rep = backward_on_trace(t_rep, cot_rep)

    expected_grad_W = grads_rep[W_rep].to_numpy()
    expected_grad_X = grads_rep[X_rep].to_numpy()

    actual_grad_W = grad_W.to_numpy()
    actual_grad_X = grad_X.to_numpy()

    np.testing.assert_allclose(actual_grad_W, expected_grad_W, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_grad_X, expected_grad_X, rtol=1e-5, atol=1e-5)

    print("\n✅ Success: MLP gradients are correct and correctly sharded!")


if __name__ == "__main__":
    test_sharded_mlp_grads()
