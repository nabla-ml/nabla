import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec


def debug_dp_mp_comparison():
    print("=== Debugging DP vs MP vs NumPy Ground Truth ===")
    mesh = DeviceMesh("dgx", (2,), ("tp",))

    # Shapes
    B, K, N = 4, 8, 4

    np.random.seed(42)
    W_data = np.random.randn(K, N).astype(np.float32)
    X_data = np.random.randn(B, K).astype(np.float32)

    # --- NumPy Ground Truth ---
    # Loss L_vec = sum(X@W, axis=0)
    # VJP with v=ones means L_scalar = sum(sum(X@W))

    # dL/dW = X.T @ ones(B, N)
    grad_W_gt = X_data.T @ np.ones((B, N), dtype=np.float32)

    # dL/dX = ones(B, N) @ W.T
    grad_X_gt = np.ones((B, N), dtype=np.float32) @ W_data.T

    print("\nGround Truth Stats:")
    print(f"grad_W mean: {grad_W_gt.mean():.4f}")
    print(f"grad_X mean: {grad_X_gt.mean():.4f}")

    # --- DP Execution ---
    print("\n--- DP Execution ---")
    W_dp = nb.ops.shard(nb.ops.constant(W_data), mesh, [DimSpec([]), DimSpec([])])
    # X sharded on axis 0 (Batch)
    X_dp = nb.ops.shard(nb.ops.constant(X_data), mesh, [DimSpec(["tp"]), DimSpec([])])

    # DEBUG: Inspect X_dp shards
    X_dp.hydrate()
    x_shards = X_dp.values
    print(f"DEBUG X_dp shards count: {len(x_shards)}")
    t0 = nb.Tensor(value=x_shards[0])
    t1 = nb.Tensor(value=x_shards[1]) if len(x_shards) > 1 else None

    nb.GRAPH.evaluate(t0, t1) if t1 is not None else nb.GRAPH.evaluate(t0)

    print(f"DEBUG X_dp shard0 sum: {t0.to_numpy().sum(axis=0)[0]}")
    if t1 is not None:
        print(f"DEBUG X_dp shard1 sum: {t1.to_numpy().sum(axis=0)[0]}")

    def step_fn(x, w):
        h = nb.matmul(x, w)
        return nb.reduce_sum(h, axis=0)

    t_dp = trace(step_fn, X_dp, W_dp)
    # Manually compute/rehydrate to ensure values are ready
    t_dp.compute()
    t_dp.rehydrate()

    cot_dp = nb.ops.full_like(
        nb.Tensor(impl=t_dp.nodes[-1].get_alive_outputs()[0]), 1.0
    )
    grad_dp_map = backward_on_trace(t_dp, cot_dp)

    grad_W_dp = grad_dp_map[W_dp].to_numpy()
    grad_X_dp = grad_dp_map[X_dp].to_numpy()

    print(f"DP grad_W shape: {grad_W_dp.shape}")
    print(f"DP grad_W sharding: {grad_dp_map[W_dp].sharding}")
    print(f"DP grad_W mismatch: {np.abs(grad_W_dp - grad_W_gt).max():.6f}")
    print(f"DP grad_X mismatch: {np.abs(grad_X_dp - grad_X_gt).max():.6f}")

    # --- MP Execution ---
    print("\n--- MP Execution ---")
    # Row Parallel: X sharded on K (axis 1), W sharded on K (axis 0)
    X_mp = nb.ops.shard(nb.ops.constant(X_data), mesh, [DimSpec([]), DimSpec(["tp"])])
    W_mp = nb.ops.shard(nb.ops.constant(W_data), mesh, [DimSpec(["tp"]), DimSpec([])])

    t_mp = trace(step_fn, X_mp, W_mp)
    t_mp.compute()
    t_mp.rehydrate()

    cot_mp = nb.ops.full_like(
        nb.Tensor(impl=t_mp.nodes[-1].get_alive_outputs()[0]), 1.0
    )
    grad_mp_map = backward_on_trace(t_mp, cot_mp)

    grad_W_mp = grad_mp_map[W_mp].to_numpy()
    print(f"MP grad_W shape: {grad_W_mp.shape}")
    print(f"MP grad_W sharding: {grad_mp_map[W_mp].sharding}")
    grad_X_mp = grad_mp_map[X_mp].to_numpy()

    print(f"MP grad_W mismatch: {np.abs(grad_W_mp - grad_W_gt).max():.6f}")
    print(f"MP grad_X mismatch: {np.abs(grad_X_mp - grad_X_gt).max():.6f}")

    # Inspect DP Logic
    if np.abs(grad_W_dp - grad_W_gt).max() > 1e-4:
        print("\n!!! DP FAILURE ANALYSIS !!!")
        print("DP grad_W (First col):", grad_W_dp[:, 0])
        print("GT grad_W (First col):", grad_W_gt[:, 0])
        # If DP is missing a reduction, it might look like local sums
        # Simulate local sums
        # Split X_data into 2 shards
        X_shards = np.split(X_data, 2, axis=0)
        local_sums = [s.sum(axis=0) for s in X_shards]
        print("Local shard sums (col 0):", [s[0] for s in local_sums])


if __name__ == "__main__":
    debug_dp_mp_comparison()
