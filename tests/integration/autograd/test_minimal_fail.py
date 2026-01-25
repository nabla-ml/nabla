import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec


def test_add_conflict_minimal():
    mesh_2d = DeviceMesh("mesh2d", (2, 2), ("dp", "tp"))

    x_np = np.zeros((8, 8), dtype=np.float32)
    y_np = np.zeros((8, 8), dtype=np.float32)

    x_nb = nb.Tensor.from_dlpack(x_np)
    y_nb = nb.Tensor.from_dlpack(y_np)

    # Shard x on row (dp), y on col (tp)
    x_nb = nb.ops.shard(x_nb, mesh_2d, [DimSpec(["dp"]), DimSpec([])])
    y_nb = nb.ops.shard(y_nb, mesh_2d, [DimSpec([]), DimSpec(["tp"])])

    def fn(x, y):
        z = x + y
        return nb.ops.reduce_sum(z, axis=[0, 1])

    print("\nTracing fn...")
    traced = trace(fn, x_nb, y_nb)
    print("Trace nodes:")
    print(traced)

    print("\nRunning backward...")
    cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
    grads = backward_on_trace(traced, cotangent)
    print("DONE")


if __name__ == "__main__":
    try:
        test_add_conflict_minimal()
    except Exception as e:
        import traceback

        traceback.print_exc()
        exit(1)
