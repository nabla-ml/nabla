# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec


def test_dp_vs_mp_grads():
    mesh = DeviceMesh("dgx", (2,), ("tp",))

    # Shapes
    B, K, N = 4, 8, 4

    W_data = np.random.randn(K, N).astype(np.float32)
    X_data = np.random.randn(B, K).astype(np.float32)

    # -------------------------------------------------------------------------
    # 1. DATA PARALLEL (DP)
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Parallel (DP) ---")
    W_dp = nb.ops.shard(nb.Tensor.from_dlpack(W_data), mesh, [DimSpec([]), DimSpec([])])
    X_dp = nb.ops.shard(
        nb.Tensor.from_dlpack(X_data), mesh, [DimSpec(["tp"]), DimSpec([])]
    )

    def step_fn(x, w):
        h = nb.matmul(x, w)
        return nb.reduce_sum(h, axis=0)  # [4]

    t_dp = trace(step_fn, X_dp, W_dp)
    grad_dp = backward_on_trace(t_dp, nb.ops.full_like(t_dp.outputs, 1.0))

    print(f"DP grad_W sharding: {grad_dp[W_dp].sharding}")
    print(f"DP grad_X sharding: {grad_dp[X_dp].sharding}")

    # -------------------------------------------------------------------------
    # 2. MODEL PARALLEL (MP - Row Parallel Linear)
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Parallel (MP - Row Parallel) ---")
    # Row Parallel Linear: shard contraction axis (K)
    # X: (B, K_sharded), W: (K_sharded, N)
    X_mp = nb.ops.shard(
        nb.Tensor.from_dlpack(X_data), mesh, [DimSpec([]), DimSpec(["tp"])]
    )
    W_mp = nb.ops.shard(
        nb.Tensor.from_dlpack(W_data), mesh, [DimSpec(["tp"]), DimSpec([])]
    )

    t_mp = trace(step_fn, X_mp, W_mp)
    # Forward trace should show a Partial Sum + AllReduce
    print("\nMP Forward Trace:")
    print(t_mp)

    grad_mp = backward_on_trace(t_mp, nb.ops.full_like(t_mp.outputs, 1.0))

    print(f"MP grad_W sharding: {grad_mp[W_mp].sharding}")
    print(f"MP grad_X sharding: {grad_mp[X_mp].sharding}")

    # NUMERICAL VERIFICATION
    # Gradients should be identical to DP and identical to each other
    np.testing.assert_allclose(
        grad_dp[W_dp].to_numpy(), grad_mp[W_mp].to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        grad_dp[X_dp].to_numpy(), grad_mp[X_mp].to_numpy(), rtol=1e-5
    )

    print("\n✅ Success: DP and MP gradients match perfectly!")


if __name__ == "__main__":
    test_dp_vs_mp_grads()
