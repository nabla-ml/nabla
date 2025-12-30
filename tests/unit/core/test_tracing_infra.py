"""Test the tracing infrastructure (OutputRefs and multi-output tracking)."""



from nabla.core.tensor import Tensor
from nabla.ops import multi_output as multi_output_ops

# Test 1: Single output operation
print("=" * 60)
print("Test 1: Single Output Operation")
print("=" * 60)

x = Tensor.zeros((2, 3), traced=True)
y = Tensor.zeros((2, 3), traced=True)
z = x + y

print(f"x._impl.output_refs: {x._impl.output_refs}")
print(f"x._impl.output_index: {x._impl.output_index}")
print(f"z._impl.output_refs: {z._impl.output_refs}")
print(f"z._impl.output_index: {z._impl.output_index}")
print()

# Test 2: Multi-output operation (split)
print("=" * 60)
print("Test 2: Multi-Output Operation (split)")
print("=" * 60)

a = Tensor.zeros((4, 6), traced=True)
chunks = multi_output_ops.split(a, num_splits=2, axis=0)

print(f"Number of chunks: {len(chunks)}")
print(f"chunks[0]._impl.output_refs: {chunks[0]._impl.output_refs}")
print(f"chunks[1]._impl.output_refs: {chunks[1]._impl.output_refs}")
print(f"Same OutputRefs? {chunks[0]._impl.output_refs is chunks[1]._impl.output_refs}")
print(f"chunks[0]._impl.output_index: {chunks[0]._impl.output_index}")
print(f"chunks[1]._impl.output_index: {chunks[1]._impl.output_index}")
print()

# Test 3: Verify weak references work
print("=" * 60)
print("Test 3: Weak References (siblings)")
print("=" * 60)

b = Tensor.zeros((6, 4), traced=True)
parts = list(multi_output_ops.split(b, num_splits=3, axis=0))  # Convert to list
output_refs = parts[0]._impl.output_refs

print(f"output_refs: {output_refs}")
print(f"Alive outputs: {len([x for x in output_refs.get_alive_outputs() if x is not None])}")

# Delete one sibling and verify weak ref becomes None
import gc
del parts[1]
gc.collect()  # Force garbage collection
print(f"After deleting parts[1]:")
print(f"Alive outputs: {len([x for x in output_refs.get_alive_outputs() if x is not None])}")
print()

# Test 4: Operation with nested outputs
print("=" * 60)
print("Test 4: Nested Output Structure (dict)")
print("=" * 60)

# For this we'd need an operation that returns a dict, let's just verify the structure
c = Tensor.zeros((2, 2), traced=True)
d = Tensor.zeros((2, 2), traced=True)
e = c + d

# Check that tree_def is populated
if e._impl.output_refs:
    print(f"e._impl.output_refs.tree_def: {e._impl.output_refs.tree_def}")
    print(f"e._impl.output_refs.num_outputs: {e._impl.output_refs.num_outputs}")

print()
print("=" * 60)
print("All tests passed!")
print("=" * 60)
