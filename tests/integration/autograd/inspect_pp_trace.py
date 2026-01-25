import numpy as np
from nabla import ops
from nabla.core import grad
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms.vmap import vmap
from nabla.ops import communication
from nabla.core.graph.tracing import trace


def inspect_pp_trace():
    print("=== Inspecting Pipeline Parallel + Data Parallel Trace ===")

    # Setup Device Mesh: 2 Stages
    mesh = DeviceMesh("pp_mesh", (2,), ("stage",))
    STAGES = 2
    B, D = 4, 8  # Batch Size 4, Hidden Dim 8

    print(f"Mesh: {mesh}")
    print(f"Batch Size: {B}, Hidden Dim: {D}, Stages: {STAGES}")

    # Random Weights and Inputs
    # Note: Weights have leading 'stage' dim to be vmapped over
    w1 = np.random.randn(STAGES, D, D).astype(np.float32)
    w2 = np.random.randn(STAGES, D, D).astype(np.float32)

    x = np.random.randn(STAGES, B, D).astype(np.float32)
    target = np.random.randn(STAGES, B, D).astype(np.float32)

    # Shard inputs across 'stage' dimension
    # w1, w2, x, target are all physically on their respective stages
    w1_sharded = ops.shard(
        ops.constant(w1), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
    )
    w2_sharded = ops.shard(
        ops.constant(w2), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
    )

    x_sharded = ops.shard(
        ops.constant(x), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
    )
    target_sharded = ops.shard(
        ops.constant(target), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])]
    )

    params = {"w1": w1_sharded, "w2": w2_sharded}

    # Define per-stage computation
    def stage_fn(x, p):
        # Local computation for one stage
        h = ops.relu(x @ p["w1"])
        out = h @ p["w2"]
        return out

    # Vmap over stages to simulate pipeline
    # spmd_axis_name="stage" means this vmap maps over the mesh axis "stage"
    stage_mapped = vmap(
        stage_fn, in_axes=(0, 0), out_axes=0, spmd_axis_name="stage", mesh=mesh
    )

    def pipeline_loss(params, x, target):
        # 1. Forward pass (parallel across stages)
        out = stage_mapped(x, params)

        # 2. Pipeline Communication
        # Pass output of stage i to stage i+1 (ring/linear topology)
        total = len(mesh.devices)
        perm = [(i, (i + 1) % total) for i in range(total)]

        # In a real pipeline, stage 0 passes to 1, 1 to 2, etc.
        # Here we simulate a cyclic pass for testing: 0->1, 1->0
        passed_out = communication.ppermute(out, perm)

        # 3. Compute Loss
        # passed_out is what the next stage received.
        # We compare it to target (which is also distributed).
        diff = passed_out - target

        # 4. Global Loss Reduction
        # mean() with axis=None reduces over ALL dimensions including the sharded one
        return ops.mean(diff * diff)

    # Create the gradient function
    # Enable create_graph=True because we want to trace the backward pass itself
    grad_fn = grad(pipeline_loss, create_graph=True)

    print("\n--- Tracing Gradient Function ---")
    # Trace the execution
    # We pass the sharded tensors as inputs
    traced = trace(grad_fn, params, x_sharded, target_sharded)

    print("\n--- Trace Graph ---")
    print(traced)


if __name__ == "__main__":
    inspect_pp_trace()
