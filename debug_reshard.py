#!/usr/bin/env python
"""Debug script to trace resharding logic for promise tensors."""

import numpy as np
import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.ops import add

# Create mesh
mesh = DeviceMesh("mesh", (2, 2), ("dp", "tp"))

# Create input arrays
np_a = np.random.randn(8, 8).astype(np.float32)
np_b = np.random.randn(8, 8).astype(np.float32)

# Create tensors
a = nb.Tensor.constant(np_a)
b = nb.Tensor.constant(np_b)

print("=== INITIAL TENSORS ===")
print(f"a.sharding: {a.sharding}")
print(f"b.sharding: {b.sharding}")
print(f"a._impl._graph_values: {a._impl._graph_values}")

# Shard with different specs
specs_a = [DimSpec(["dp"], is_open=False), DimSpec([], is_open=True)]
specs_b = [DimSpec([], is_open=True), DimSpec(["tp"], is_open=False)]

print("\n=== SHARDING A ===")
a_sharded = a.shard(mesh, specs_a)
print(f"a_sharded.sharding: {a_sharded.sharding}")
print(f"a_sharded._impl._graph_values: {a_sharded._impl._graph_values}")
print(f"a_sharded._impl.output_refs: {a_sharded._impl.output_refs}")
print(f"a_sharded.shape: {a_sharded.shape}")
print(f"a_sharded._impl._physical_shapes: {getattr(a_sharded._impl, '_physical_shapes', None)}")

print("\n=== SHARDING B ===")
b_sharded = b.shard(mesh, specs_b)
print(f"b_sharded.sharding: {b_sharded.sharding}")
print(f"b_sharded._impl._graph_values: {b_sharded._impl._graph_values}")
print(f"b_sharded._impl.output_refs: {b_sharded._impl.output_refs}")

print("\n=== BEFORE ADD ===")
print(f"a_sharded.sharding.dim_specs: {a_sharded.sharding.dim_specs}")
print(f"b_sharded.sharding.dim_specs: {b_sharded.sharding.dim_specs}")

print("\n=== CALLING ADD ===")
# Check if needs_reshard is called
from nabla.core.sharding.spec import needs_reshard

print(f"needs_reshard(a, b_spec): {needs_reshard(a_sharded.sharding, b_sharded.sharding)}")
print(f"needs_reshard(b, a_spec): {needs_reshard(b_sharded.sharding, a_sharded.sharding)}")

# Manually trace what reshard_inputs would do
from nabla.core.sharding import spmd
from nabla.ops.binary import AddOp

# Get input shardings from infer_output_sharding
print("\n=== CALLING infer_output_sharding ===")
output_spec, input_specs, needs_reduce = spmd.infer_output_sharding(AddOp(), (a_sharded, b_sharded), mesh, {})
print(f"output_spec: {output_spec}")
print(f"input_specs: {input_specs}")
print(f"needs_reduce: {needs_reduce}")

# Now call reshard_inputs
print("\n=== CALLING reshard_inputs ===")
resharded = spmd.reshard_inputs((a_sharded, b_sharded), input_specs, mesh)
print(f"resharded[0] is a_sharded: {resharded[0] is a_sharded}")
print(f"resharded[1] is b_sharded: {resharded[1] is b_sharded}")
print(f"resharded[0].sharding: {resharded[0].sharding}")
print(f"resharded[1].sharding: {resharded[1].sharding}")
print(f"resharded[0]._impl.output_refs: {resharded[0]._impl.output_refs}")
print(f"resharded[1]._impl.output_refs: {resharded[1]._impl.output_refs}")
print(f"resharded[0]._impl.output_refs is a_sharded._impl.output_refs: {resharded[0]._impl.output_refs is a_sharded._impl.output_refs}")
print(f"resharded[1]._impl.output_refs is b_sharded._impl.output_refs: {resharded[1]._impl.output_refs is b_sharded._impl.output_refs}")

# Check the OpNode DAG
print("\n=== OpNode DAG Analysis ===")
print(f"a_sharded opnode: {id(a_sharded._impl.output_refs)}")
print(f"resharded[0] opnode: {id(resharded[0]._impl.output_refs)}")
print(f"b_sharded opnode: {id(b_sharded._impl.output_refs)}")
print(f"resharded[1] opnode: {id(resharded[1]._impl.output_refs)}")

# Check the op_args of the resharded OpNodes
if resharded[0]._impl.output_refs:
    opnode = resharded[0]._impl.output_refs
    print(f"\nresharded[0] OpNode.op_args: {opnode.op_args}")
    print(f"resharded[0] OpNode.op_kwargs: {opnode.op_kwargs}")
if resharded[1]._impl.output_refs:
    opnode = resharded[1]._impl.output_refs
    print(f"resharded[1] OpNode.op_args: {opnode.op_args}")
    print(f"resharded[1] OpNode.op_kwargs: {opnode.op_kwargs}")

result = add(a_sharded, b_sharded)

print("\n=== AFTER ADD ===")
print(f"result.sharding: {result.sharding}")
print(f"result._impl._graph_values: {result._impl._graph_values}")
print(f"result._impl.output_refs: {result._impl.output_refs}")
print(f"result.shape: {result.shape}")
print(f"result._impl._physical_shapes: {getattr(result._impl, '_physical_shapes', None)}")

# Check what the AddOp's OpNode has as inputs
add_opnode = result._impl.output_refs
print(f"\n=== AddOp OpNode Analysis ===")
print(f"add_opnode.op_args: {add_opnode.op_args}")
for i, arg in enumerate(add_opnode.op_args):
    print(f"  arg[{i}] id: {id(arg)}")
    print(f"  arg[{i}].sharding: {arg.sharding}")
    print(f"  arg[{i}].output_refs: {arg.output_refs}")

# Compare with resharded tensors
print(f"\n=== Comparison ===")
print(f"resharded[0]._impl id: {id(resharded[0]._impl)}")
print(f"resharded[1]._impl id: {id(resharded[1]._impl)}")
print(f"a_sharded._impl id: {id(a_sharded._impl)}")
print(f"b_sharded._impl id: {id(b_sharded._impl)}")
print(f"result.sharding: {result.sharding}")
print(f"result._impl._graph_values: {result._impl._graph_values}")
print(f"result._impl.output_refs: {result._impl.output_refs}")
print(f"result.shape: {result.shape}")
print(f"result._impl._physical_shapes: {getattr(result._impl, '_physical_shapes', None)}")

print("\n=== EVALUATING (with debug) ===")
# Enable debug
import nabla.core.graph.engine as engine_module
engine_module.DEBUG_LAZY_EVAL = True

result_np = result.numpy()
expected = np_a + np_b

print(f"result shape: {result_np.shape}")
print(f"expected shape: {expected.shape}")
print(f"match: {np.allclose(result_np, expected, rtol=1e-5, atol=1e-6)}")

if not np.allclose(result_np, expected, rtol=1e-5, atol=1e-6):
    print("\n=== MISMATCH DETAILS ===")
    print(f"result:\n{result_np[:3, :5]}")
    print(f"expected:\n{expected[:3, :5]}")
    diff = np.abs(result_np - expected)
    print(f"max diff: {diff.max()}")
