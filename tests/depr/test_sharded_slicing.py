import numpy as np
import pytest

import nabla as nb
from nabla import ops
from nabla.core.sharding import DeviceMesh, DimSpec


def test_gather_on_sharded_axis():
    print("\n=== Testing Gather on Sharded Axis ===")

    # Mesh: (2,)
    mesh = DeviceMesh("slice_mesh", (2,), ("tp",))

    # Data: (4, 4) sharded on axis 0
    # Global shape (4, 4), Local shape (2, 4)
    data = np.random.randn(4, 4).astype(np.float32)

    # Indices: [3] -> selects last row
    # Should work globally.
    idx = 3
    indices = np.array([idx], dtype=np.int32)

    # Nabla
    d_nb = nb.Tensor.from_dlpack(data)
    d_nb = ops.shard(d_nb, mesh, [DimSpec(["tp"]), DimSpec([])])

    i_nb = nb.Tensor.from_dlpack(indices)
    # Indices replicated
    i_nb = ops.shard(i_nb, mesh, [DimSpec([])])

    def slice_fn(x, i):
        # Gather axis 0
        return ops.gather(x, i, axis=0)

    # If logic is correct, it should fetch row 3 from device 1 (which has rows 2,3).
    # Device 0 (rows 0,1) should participate or send data?
    # If output is replicated on 'tp', then we need AllGather or Broadcast.

    traced = nb.core.graph.tracing.trace(slice_fn, d_nb, i_nb)
    print("Trace:", traced)

    try:
        out = slice_fn(d_nb, i_nb)
        print("Output shape:", out.shape)
        # Force realization
        out_np = out.to_numpy()
        print("Output val:", out_np)

        expected = data[indices]
        np.testing.assert_allclose(out_np, expected, atol=1e-5)
        print("PASS: Gather succeeded")

    except Exception as e:
        print(f"FAIL: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
