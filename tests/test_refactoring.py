"""Test the tracing infrastructure refactoring (OutputRefs op metadata)."""

import sys
sys.path.insert(0, '/Users/tillife/Documents/CodingProjects/nabla')

from nabla.core.tensor import Tensor
from nabla.ops import multi_output as multi_output_ops
from nabla.core.tracing import OutputRefs

print("=" * 60)
print("Test: OutputRefs Op Metadata Refactoring")
print("=" * 60)

# Create a traced operation
x = Tensor.arange(0, 10)
x._impl.traced = True
y = x + 5

print(f"y._impl.output_refs type: {type(y._impl.output_refs)}")

# Verify metadata is now in OutputRefs
refs = y._impl.output_refs
print(f"refs.op: {refs.op}")
print(f"refs.op_args: {len(refs.op_args)} args")
print(f"refs.op_kwargs: {refs.op_kwargs}")

assert refs.op is not None, "OutputRefs should have op"
assert len(refs.op_args) == 2, "Should have 2 args (x, 5)"
# For BinaryOperation, op_args stores the BROADCASTED TensorImpl refs,
# not the original x._impl. The first arg will be a broadcast_to output.
from nabla.core.tensor_impl import TensorImpl
assert isinstance(refs.op_args[0], TensorImpl), f"First arg should be TensorImpl, got {type(refs.op_args[0])}"

# Verify TensorImpl properties delegate correctly
print(f"\nVerifying TensorImpl delegation:")
print(f"y._impl.op: {y._impl.op}")
print(f"y._impl.op_name: {y._impl.op_name}")
print(f"y._impl.parents: {len(y._impl.parents)} parents")

assert y._impl.op is refs.op, "TensorImpl.op should delegate to output_refs.op"
# The parent is the broadcast_to output, which itself has x._impl as a parent
assert len(y._impl.parents) > 0, "TensorImpl.parents should have at least one parent"

# Test Multi-Output Operation (Split)
print("\n" + "=" * 60)
print("Test: Multi-Output Metadata Sharing")
print("=" * 60)

a, b = multi_output_ops.split(x, num_splits=2, axis=0)

refs_a = a._impl.output_refs
refs_b = b._impl.output_refs

print(f"refs_a is refs_b? {refs_a is refs_b}")
assert refs_a is refs_b, "Siblings should share OutputRefs instance"

# Verify we only have ONE copy of op_args in memory (via the shared object)
args_a = refs_a.op_args
args_b = refs_b.op_args

print(f"args_a is args_b? {args_a is args_b}")
assert args_a is args_b, "Should access exactly the same tuple object"

print("\n✓ SUCCESS: Refactoring verified!")
print("  - Metadata moved to OutputRefs")
print("  - TensorImpl properties delegate correctly")
print("  - Single copy of op_args shared among siblings")
print("=" * 60)
