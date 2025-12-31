
from max.graph import Graph, TensorValue, ops
from max.graph.ops.allreduce import sum as allreduce_sum
from max.dtype import DType

def test_same_device_allreduce():
    # Test: Can we run allreduce on inputs that live on the SAME device (CPU)?
    # If yes, we don't need a "SimulatedBackend" - we just use the real op.
    
    graph = Graph("test_graph")
    with graph:
        # Create two CPU tensors
        from max.graph import DeviceRef
        t1 = ops.constant(1.0, DType.float32, device=DeviceRef.CPU())
        t2 = ops.constant(2.0, DType.float32, device=DeviceRef.CPU())
        
        # We need signal buffers for allreduce? 
        # API says: signal_buffers: Iterable[BufferValueLike]
        # Let's see if we can pass empty or if it requires real buffers.
        # But wait, signal buffers are usually per-device.
        
        try:
            # Attempt 1: Just pass the tensors
            # We assume signal buffers are optional or we can mock them?
            # The signature showed signal_buffers is required argument.
            
            # Let's create dummy buffers if needed.
            # But wait, max.graph.ops.allreduce.sum doc says:
            # signal_buffers: Device buffer values used for synchronization.
            
            # This implies the op IS low-level and expects specific buffer management.
            # Usually users don't manage signal buffers manually in high-level frameworks.
            # But this is the low-level graph API.
            
            # Let's try to pass None or empty list?
            print("Attempting to call allreduce_sum with CPU tensors...")
            
            # We might need to mock signal buffers.
            # Creating a dummy buffer instruction?
            # ops.alloc?
            
            # For this test, let's just see if it compiles/constructs without error
            # when passed logical inputs.
            
            # Inputs must be iterable
            inputs = [t1, t2]
            
            # Signal buffers... tricky.
            # Let's try to leave it out purely to see the error message -> hint
                # Try with empty list
                try:
                    results = allreduce_sum(inputs, signal_buffers=[])
                    print("Call with [] signals SUCCESS!")
                except Exception as e:
                    print(f"Call with [] signals failed: {e}")
                
        except Exception as e:
            print(f"FATAL: {e}")

if __name__ == "__main__":
    test_same_device_allreduce()
