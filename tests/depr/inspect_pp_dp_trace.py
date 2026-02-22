import numpy as np

from nabla import ops
from nabla.transforms import grad
from nabla.core.graph.tracing import trace
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.ops import communication
from nabla.transforms.vmap import vmap


def inspect_pp_dp_trace():
    print("=== Inspecting Pipeline Parallel + Data Parallel Trace ===")

    # Setup Device Mesh: 2 Stages, 2 DP Replicas (Total 4 devices)
    mesh = DeviceMesh("pp_dp_mesh", (2, 2), ("stage", "dp"))
    STAGES = 2
    DP = 2
    B_LOC = 2
    B_GLOBAL = B_LOC * DP
    D = 8

    print(f"Mesh: {mesh}")
    print(
        f"Global Batch: {B_GLOBAL}, Hidden Dim: {D}, Stages: {STAGES}, DP Replicas: {DP}"
    )

    w1 = np.random.randn(STAGES, D, D).astype(np.float32)
    w2 = np.random.randn(STAGES, D, D).astype(np.float32)

    # Inputs: (STAGES, B_GLOBAL, D)
    x = np.random.randn(STAGES, B_GLOBAL, D).astype(np.float32)
    target = np.random.randn(STAGES, B_GLOBAL, D).astype(np.float32)

    # Sharding
    # W: Sharded on 'stage', Replicated on 'dp' (implied by absence)
    w1_sharded = ops.shard(
        ops.constant(w1), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
    )
    w2_sharded = ops.shard(
        ops.constant(w2), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
    )

    # X: Sharded on 'stage' AND 'dp' (Batch dimension is split)
    # axis 0: stage (mapped by vmap)
    # axis 1: batch (sharded by dp)
    # axis 2: hidden (replicated)
    x_sharded = ops.shard(
        ops.constant(x), mesh, [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])]
    )
    target_sharded = ops.shard(
        ops.constant(target), mesh, [DimSpec(["stage"]), DimSpec(["dp"]), DimSpec([])]
    )

    params = {"w1": w1_sharded, "w2": w2_sharded}

    def stage_fn(x, p):
        # x: (B_GLOBAL, D) logical input to stage
        # Inside vmap(mesh="stage"): code runs on one stage.
        # But 'x' is still sharded on 'dp'.
        # x sharding: <dp, *>
        # w sharding: <*, *> (replicated on dp)

        # Matmul: <dp, *> @ <*, *> -> <dp, *>
        h = ops.relu(x @ p["w1"])
        out = h @ p["w2"]
        return out

    # vmap over 'stage' axis
    stage_mapped = vmap(
        stage_fn, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh
    )

    def pipeline_loss(params, x, target):
        out = stage_mapped(x, params)

        # Pipeline Communication: Shift stages
        # Mesh (2, 2): S0={0,1}, S1={2,3}
        # Permutation must respect DP pairing: (0->2, 1->3, etc.)
        perm = [(0, 2), (1, 3), (2, 0), (3, 1)]
        passed_out = communication.ppermute(out, perm)

        diff = passed_out - target
        return ops.mean(diff * diff)

    # Trace with create_graph=True to see the Backward Graph operations
    grad_fn = grad(pipeline_loss, create_graph=True)

    print("\n--- Tracing Gradient Function ---")
    traced = trace(grad_fn, params, x_sharded, target_sharded)

    print("\n--- Trace Graph ---")
    print(traced)


if __name__ == "__main__":
    inspect_pp_dp_trace()
