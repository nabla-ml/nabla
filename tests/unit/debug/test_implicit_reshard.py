
import numpy as np
import nabla
from nabla import Tensor, DeviceMesh, DimSpec, ops
from nabla.utils.debug import capture_trace

def test_implicit_reshard_trace():
    """Verify that implicit reshards appear as nodes in the trace."""
    print("=" * 70)
    print("TRACE: Implicit Resharding")
    print("=" * 70)
    
    # Mesh: (dp=2, tp=2)
    mesh = DeviceMesh("mesh", (2, 2), ("dp", "tp"))
    
    # Shape: (4, 4)
    # A: Sharded on DP [("dp"), ("tp")]
    a = Tensor.ones((4, 4))
    a = a.shard(mesh, [DimSpec(["dp"]), DimSpec(["tp"])])
    
    # B: Sharded on TP [("tp"), ("dp")] (Axes swapped needs all-to-all or gather+shard)
    b = Tensor.ones((4, 4))
    b = b.shard(mesh, [DimSpec(["tp"]), DimSpec(["dp"])])
    
    print(f"A spec: {a._impl.sharding}")
    print(f"B spec: {b._impl.sharding}")
    
    # Operation: Add A + B
    # Should force alignemnt. 
    # Usually naive propagation keeps input specs.
    # But binary op might force one to match the other or both to a common ground?
    # Actually `BinaryOperation` generic behavior does NOT force common spec unless required.
    # But `spmd.infer_output_sharding` usually falls back to elementwise which requires inputs to match?
    # No, `elementwise_template` propagates requirements.
    # If we have conflict, `resolve` will pick one.
    # Then `update` sets global requirements.
    # Then `reshard_inputs` enforces them.
    
    print("-" * 70)
    print("TRACE OUTPUT:")
    print("-" * 70)
    
    # We trace a lambda
    trace = capture_trace(lambda x, y: x + y, a, b)
    print(trace)

if __name__ == "__main__":
    test_implicit_reshard_trace()
