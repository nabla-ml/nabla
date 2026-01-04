import nabla
from nabla.utils.debug import xpr, capture_trace
from nabla.sharding import DeviceMesh, ShardingSpec, DimSpec

def manual_test():
    print("=== Test 1: Simple Capture Trace ===")
    
    def simple_fn(x, y):
        a = x + y
        b = a * 2
        return b
        
    x = nabla.ones((2, 2))
    y = nabla.ones((2, 2))
    
    trace = capture_trace(simple_fn, x, y)
    print(trace)
    
    print("\n=== Test 2: xpr with traced inputs ===")
    
    # Manually trace and run ops
    a = nabla.ones((4, 4))
    a.trace()  # Enable tracing
    b = a + a
    c = b * 2
    
    print(xpr(c))

    print("\n=== Test 3: Reduction with Kwargs ===")
    
    def reduce_fn(x):
        y = nabla.reduce_sum(x, axis=0, keepdims=True)
        return y
        
    x = nabla.ones((4, 4))
    trace = capture_trace(reduce_fn, x)
    print(trace)
    
    print("\n=== Test 4: Sharding Visualization ===")
    from nabla.sharding import DeviceMesh, DimSpec
    from nabla.ops.communication import shard, all_gather
    
    mesh = DeviceMesh("mesh", (2,), ("x",))
    
    def sharded_fn(x):
        # Shard the input
        sharded = shard(x, mesh, [DimSpec(["x"]), DimSpec([])])
        # Do some local ops
        y = sharded * 2
        z = y + 1
        # Gather back
        gathered = all_gather(z, axis=0)
        return gathered
    
    x = nabla.ones((4, 4))
    trace = capture_trace(sharded_fn, x)
    print(trace)
    
    print("\n=== Test 5: Batch Dims (vmap-like) ===")
    from nabla.ops._physical import incr_batch_dims
    
    # Simulate what vmap does internally
    x = nabla.ones((3, 4, 4))
    x.trace()
    batched = incr_batch_dims(x)  # Now has batch_dims=1
    y = batched * 2 + 1
    
    print(xpr(y))
    print("\n(Batch dims should appear in light purple/blue color)")

if __name__ == "__main__":
    manual_test()
