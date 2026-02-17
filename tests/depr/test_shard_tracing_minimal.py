# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import numpy as np

import nabla as nb
from nabla import DeviceMesh, DimSpec, ops
from nabla.core.graph.tracing import trace


def test_shard_tracing_minimal():
    print("\n--- Starting Shard Tracing Minimal Test ---")
    STAGES = 2
    mesh = DeviceMesh("test", (STAGES,), ("stage",))
    print(f"Created mesh: {mesh}")

    # 1. Create Tensor
    x_np = np.random.randn(2, 4).astype(np.float32)
    x = nb.Tensor.from_dlpack(x_np)
    print(f"Input tensor shape: {x.shape}")

    # 2. Shard
    # P("stage", None) corresponds to sharding on dim 0
    spec = [DimSpec(["stage"]), DimSpec([])]

    def fn(t):
        return ops.shard(t, mesh, spec)

    # 3. Trace
    print("Tracing shard operation...")
    tr = trace(fn, x)

    print("\nTrace Representation:")
    print(tr)

    # 4. Verify
    trace_str = str(tr).lower()
    assert "shard" in trace_str, "Trace should contain 'shard' operation"
    print("Verification passed: 'shard' op found in trace.")

    # 5. Rehydrate
    print("\nRehydrating trace...")
    tr.rehydrate()

    # Output should have 2 values
    out = tr.outputs
    # out is what fn returned during tracing.
    # rehydrate updates the values of the existing tensors in the trace.

    assert out.is_sharded, "Output should be sharded"
    assert len(out.values) == 2, (
        f"Output should have 2 values (shards), got {len(out.values)}"
    )

    # Check values match slices
    # evaluate() ensures all shards are Evaluated/Realized
    v0 = out.to_numpy()[0:1, :]
    v1 = out.to_numpy()[1:2, :]

    assert np.allclose(v0, x_np[0:1, :]), "Shard 0 values incorrect"
    assert np.allclose(v1, x_np[1:2, :]), "Shard 1 values incorrect"

    print("Rehydration successful: values match expected slices.")
    print("--- Shard Tracing Minimal Test Passed ---\\n")


if __name__ == "__main__":
    test_shard_tracing_minimal()
