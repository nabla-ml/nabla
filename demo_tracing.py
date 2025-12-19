"""Demo script showing tracing with multi-output ops and graph printing."""

from eager.core.tensor import Tensor
from eager.core.graph_utils import print_trace_graph, get_all_impls_topological
from eager import multi_output_ops


def main():
    print("=" * 60)
    print(" TRACING DEMO: Multi-Output Ops + Graph Visualization")
    print("=" * 60)
    
    # Create a traced input tensor
    x = Tensor.arange(0, 12).trace()
    print(f"\n1. Created traced input: x = arange(0, 12).trace()")
    print(f"   x.shape = {x.shape}, traced = {x._impl.traced}")
    
    # Apply some basic ops
    y = x * 2
    print(f"\n2. y = x * 2")
    
    z = y + 10
    print(f"   z = y + 10")
    
    # Multi-output operation: split
    a, b, c = multi_output_ops.split(z, num_splits=3, axis=0)
    print(f"\n3. a, b, c = split(z, num_splits=3)")
    print(f"   a.shape = {a.shape}, b.shape = {b.shape}, c.shape = {c.shape}")
    
    # Continue computation on split outputs
    d = a + b  # Combine two split outputs
    print(f"\n4. d = a + b")
    
    e = c * d  # Another combination
    print(f"   e = c * d")
    
    # Check that siblings share the same OutputRefs
    print(f"\n5. Sibling check:")
    print(f"   a and b share OutputRefs? {a._impl.output_refs is b._impl.output_refs}")
    print(f"   a.output_index = {a._impl.output_index}, b.output_index = {b._impl.output_index}")
    
    # Print the computation graph
    print("\n" + "=" * 60)
    print(" TRACED COMPUTATION GRAPH")
    print("=" * 60 + "\n")
    
    # Print from the final output 'e'
    print_trace_graph([e._impl], show_siblings=True)
    
    # Also show topological order of all TensorImpls
    print("\n" + "=" * 60)
    print(" ALL TENSORIMPLS (Topological Order)")
    print("=" * 60)
    
    all_impls = get_all_impls_topological([e._impl])
    for i, impl in enumerate(all_impls):
        op_name = impl.op_name or "leaf/input"
        shape = impl.logical_shape
        parents_count = len(impl.parents)
        out_idx = impl.output_index
        print(f"  {i:2d}. {op_name:12s} | shape={str(shape):12s} | parents={parents_count} | out_idx={out_idx}")
    
    print("\n✓ Tracing demo complete!")


if __name__ == "__main__":
    main()
