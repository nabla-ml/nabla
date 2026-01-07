"""Test graph traversal utilities with multi-output operations."""



from nabla.core.tensor import Tensor
from nabla.ops import multi_output as multi_output_ops
from nabla.core.graph_utils import (
    get_operations_topological,
    get_all_impls_topological,
    print_trace_graph
)

print("=" * 70)
print("Testing Graph Traversal with Multi-Output Operations")
print("=" * 70)

# Create a computation with multi-output ops
print("\n1. Building computation graph...")
x = Tensor.arange(0, 12)  # Input
x._impl.traced = True  # Enable tracing
print(f"   x: shape {x.shape}")

# Split into 3 parts
a, b, c = multi_output_ops.split(x, num_splits=3, axis=0)
print(f"   split -> a, b, c: shapes {a.shape}, {b.shape}, {c.shape}")

# Use outputs independently
y1 = a + 10
y2 = b * 2
y3 = c - 5
print(f"   y1 = a + 10, y2 = b * 2, y3 = c - 5")

# Combine
result = y1 + y2 + y3
print(f"   result = y1 + y2 + y3: shape {result.shape}")

# Test 1: Get operations in topological order
print("\n" + "=" * 70)
print("Test 1: Operation-Level Traversal (Deduplication)")
print("=" * 70)

ops = get_operations_topological([result._impl])
print(f"\nTotal operations: {len(ops)}")

for idx, (op_id, outputs) in enumerate(reversed(ops)):  # Forward order
    op_name = outputs[0].op_name or "unknown"
    num_outputs = len(outputs)
    print(f"  {idx}. {op_name:15s} -> {num_outputs} output(s)")

print("\n✓ Split operation appears only ONCE (deduplication works!)")

# Test 2: Get all TensorImpls
print("\n" + "=" * 70)
print("Test 2: TensorImpl-Level Traversal")
print("=" * 70)

all_impls = get_all_impls_topological([result._impl])
print(f"\nTotal TensorImpls: {len(all_impls)}")

for idx, impl in enumerate(all_impls):
    op_name = impl.op_name or "leaf"
    traced = "traced" if impl.traced else "untraced"
    print(f"  {idx:2d}. {op_name:15s} ({traced})")

# Test 3: Pretty print
print("\n" + "=" * 70)
print("Test 3: Pretty Print (with sibling info)")
print("=" * 70)
print()
print_trace_graph([result._impl], show_siblings=True)

# Test 4: Verify deduplication
print("\n" + "=" * 70)
print("Test 4: Deduplication Verification")
print("=" * 70)

# Count how many times "split" appears
split_count = sum(1 for _, outs in ops if outs[0].op_name == "split")
print(f"\nSplit operation count: {split_count}")
assert split_count == 1, "Split should appear exactly once!"

# Verify the split op has 3 outputs
split_op = next((outs for _, outs in ops if outs[0].op_name == "split"), None)
assert split_op is not None
assert len(split_op) == 3, f"Split should have 3 outputs, got {len(split_op)}"
print(f"Split outputs: {len(split_op)}")

print("\n✓ Deduplication verified!")

print("\n" + "=" * 70)
print("All graph traversal tests passed!")
print("=" * 70)
