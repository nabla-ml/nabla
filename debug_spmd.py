"""Debug script for nested vmap with trace output."""

import numpy as np
from nabla import DeviceMesh, P, vmap, relu
from nabla.utils.debug import capture_trace, GraphPrinter
from nabla.core.tensor import Tensor

# Setup - 3D mesh for nested vmap
mesh = DeviceMesh("mesh", (2, 4, 2), ("dp", "tp", "pp"))

outer_batch = 4
inner_batch = 8
features = 16

np_x = np.random.randn(outer_batch, inner_batch, features).astype(np.float32)

print("=" * 80)
print("TRACE: Nested vmap with BOTH spmd_axis_name")
print(f"Input shape: ({outer_batch}, {inner_batch}, {features})")
print("Expected output shape: (4, 8, 16)")
print("=" * 80)

def nested_vmapped_fn(x):
    @vmap(spmd_axis_name="dp", mesh=mesh)  # Outer on dp
    def outer(batch_x):
        @vmap(spmd_axis_name="pp", mesh=mesh)  # Inner on pp
        def inner(row):
            row_sharded = row.shard(mesh, P("tp"))  # Features on tp
            return relu(row_sharded)
        return inner(batch_x)
    
    return outer(x)

x = Tensor.from_dlpack(np_x)
print(f"Input x shape: {x.shape}")

# Capture and print trace
trace = capture_trace(nested_vmapped_fn, x)

printer = GraphPrinter(trace)
print("\nTRACE OUTPUT:")
print("-" * 80)
print(printer.to_string())
print("-" * 80)

# Inspect result
result = trace.outputs
print(f"\nResult shape: {result.shape}")
print(f"Result sharding: {result._impl.sharding}")
print(f"Result cached_shape: {result._impl.cached_shape}")

if result._impl.sharding:
    print("\nSharding dimensions:")
    for i, ds in enumerate(result._impl.sharding.dim_specs):
        print(f"  Dim {i}: axes={ds.axes}")
