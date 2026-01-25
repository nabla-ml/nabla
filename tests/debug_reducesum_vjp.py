import numpy as np
import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec


def debug_reducesum_vjp():
    print("debugging reducesum vjp...")
    mesh = DeviceMesh("dp", (2,), ("data",))

    # Input X: (4, 4) sharded on 0
    x_np = np.random.randn(4, 4).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)
    x = ops.shard(x, mesh, [DimSpec(["data"]), DimSpec([])])
    print(f"X: {x.shape} sharding={x.sharding}")

    # Cotangent: Scalar 1.0 Replicated
    cot_np = np.array(1.0, dtype=np.float32)
    cot = nb.Tensor.from_dlpack(cot_np)
    print(f"Cot: {cot.shape} sharding={cot.sharding}")

    # VJP Rule: broadcast_to(cot, x.shape)
    # Using ops.broadcast_to manually
    from nabla.ops.view import shape as shape_ops

    # We expect this to default to Replicated sharding since input is Replicated
    cot_bcast = shape_ops.broadcast_to(cot, x.shape)
    print(f"Bcast: {cot_bcast.shape} sharding={cot_bcast.sharding}")

    # Backward loop does: reshard(cot_bcast, x.sharding)
    from nabla.ops.communication import reshard

    try:
        grads = reshard(cot_bcast, mesh, x.sharding.dim_specs)
        print(f"Resharded: {grads.shape} sharding={grads.sharding}")

        # Verify values locally
        # Should be all 1s
        grads.hydrate()
        vals = grads.values
        print(f"Values on 0: {np.array(vals[0]).shape}")
        # if vals[0] is (2, 4) of ones, it's correct.
        v0 = np.array(vals[0])
        if np.allclose(v0, 1.0):
            print("Values are correct (ones)")
        else:
            print("Values are WRONG (not ones)")
            print(v0)

    except Exception as e:
        print(f"Reshard failed: {e}")


if __name__ == "__main__":
    debug_reducesum_vjp()
