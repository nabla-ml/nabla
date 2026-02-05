# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec


def test_pipeline_crossing_vjp():
    """Test that ppermute correctly routes gradients back across devices."""
    # 2 devices, axis 'pp' for pipeline parallelism
    mesh = DeviceMesh("pipe_mesh", (2,), ("pp",))

    # Replicated input
    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = nb.Tensor.from_dlpack(x_data)

    def pipeline_fn(a):
        # 1. Force data to be logically 'at' the mesh but actually replicated
        # so that ppermute has something to move between "shards" (devices)
        s = nb.ops.shard(a, mesh, [DimSpec([])], replicated_axes=set())

        # 2. Stage 0 work
        y = s * 2.0

        # 3. Pipeline Hand-off: Send from Device 0 to Device 1
        # InNablas ppermute, shard i moves to permutation[i]
        # Here we move shard 0 to shard 1.
        y_moved = nb.ops.ppermute(y, permutation=[(0, 1)])

        # 4. Stage 1 work
        # This work happens on shard 1's value
        z = y_moved + 5.0

        return nb.reduce_sum(z, axis=0)

    print("\n--- Pipeline Trace ---")
    t = trace(pipeline_fn, x)
    print(t)

    # 5. Compute grads
    from nabla.ops.creation import full_like

    cot = full_like(t.outputs, 1.0)
    grads = backward_on_trace(t, cot)

    grad_x = grads[x]
    actual_grad = grad_x.to_numpy()
    print(f"\nGradient computed: {actual_grad}")

    # Expected verification
    # Output is sum(s0 * 2 + 5) ... wait.
    # ppermute(shard0) -> shard1.
    # In forward:
    # Device 0: x -> x*2 -> (sent to 1)
    # Device 1: (received from 0) -> +5 -> sum
    # Backward:
    # Device 1: dSum/dz=1 -> dz/dy_moved=1 -> (sent 1->0 via inverse ppermute)
    # Device 0: (received from 1) -> dy/ds=2 -> s=x -> grad_x=2
    # The sum total should reflect this.

    expected = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(actual_grad, expected, rtol=1e-5)
    print("✅ Success: Pipeline gradients routed correctly across stages via ppermute!")


if __name__ == "__main__":
    test_pipeline_crossing_vjp()
