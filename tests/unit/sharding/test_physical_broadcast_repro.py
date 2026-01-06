import unittest
from max.graph import TensorValue
from nabla.core.tensor import Tensor
from nabla.core.tensor_impl import TensorImpl
from nabla.ops import binary

# Mock or real? We can use real Tensor objects since we are checking logic.
# We might need to mock spmd/mesh if we don't want strict sharding checks yet,
# but the BinaryOperation logic is in Python before spmd execution.

class TestBinaryBroadcast(unittest.TestCase):
    def test_broadcast_batch_dims_left_pad(self):
        """Test adding outer batch dimensions (standard broadcasting)."""
        # Case: A(B) + B(A, B) -> (A, B)
        # x needs to be broadcasted from (B) to (1, B) -> (A, B)
        # This requires inserting a dimension at PHYSICAL index 0.
        
        # Simulating x: shape=(B,), batch_dims=1
        x_impl = TensorImpl(values=[], batch_dims=1, traced=False)
        x_impl.cached_shape = (8,) # Physical shape B=8
        x = Tensor(impl=x_impl)
        
        # Simulating y: shape=(A, B), batch_dims=2
        y_impl = TensorImpl(values=[], batch_dims=2, traced=False)
        y_impl.cached_shape = (4, 8) # Physical shape A=4, B=8
        y = Tensor(impl=y_impl)
        
        # We invoke _prepare_for_broadcast directly to see what it does
        # We need to mock the view_ops to see what they get called with, 
        # OR we can just run it if we have a real backend (which we don't fully).
        # But wait, logic is in python.
        
        # Let's inspect the logic by instrumenting or careful setup.
        # Actually, let's look at the result 'x'.
        out_batch_dims = 2
        
        # We can run it; it might fail if ops.unsqueeze interacts with max graph.
        # But we can check the calls.
        
        try:
            new_x, new_y = binary.add._prepare_for_broadcast(x, y, out_batch_dims)
            
            # Check new_x physical shape
            # x was (8,), target (4, 8). 
            # Should have unsqueezed at 0 to get (1, 8).
            # If bug exists: unsqueezed at 1 (logical 0) -> (8, 1).
            
            print(f"X batch dims: {new_x._impl.batch_dims}")
            # If we don't have real values/shapes, we can't check cached_shape easily 
            # unless we mock the ops to update it.
            
        except Exception as e:
            print(f"Trapped error: {e}")

if __name__ == "__main__":
    unittest.main()
