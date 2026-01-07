
import unittest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec
from nabla.utils import debug
import nabla.ops as ops
from nabla.ops import view, creation

def get_mesh(shape, axes):
    return DeviceMesh("cluster", shape, axes, devices=list(range(np.prod(shape))))

def reduce_mean(x, axis, keepdims=False):
    s = ops.reduce_sum(x, axis=axis, keepdims=keepdims)
    # We need to divide by the dimension size. 
    # Since we don't have shape access inside trace easily without running it, 
    # we simulate mean by just sum for sharding verification purposes.
    # Or assuming dimension size is scalar constant which broadcasts?
    # For sharding structure, Sum is equivalent to Mean.
    return s 

def layernorm_manual(x, gamma, beta, eps=1e-5):
    # x: [B, H]
    # mean: [B, 1]. reduction over H.
    # If H is sharded, this requires AllReduce.
    mean = ops.reduce_sum(x, axis=-1, keepdims=True) # approximating mean with sum
    
    # Broadcast: [B, H] - [B, 1]
    # If [B, 1] is Replicated (from AllReduce) and [B, H] is Sharded, 
    # this broadcast should be valid (Replicated broadcasts to Sharded).
    d = ops.sub(x, mean)
    
    # d**2
    sq = ops.mul(d, d)
    
    # var = reduce_sum(sq) / N
    var = ops.reduce_sum(sq, axis=-1, keepdims=True)
    
    # rsqrt
    # assuming div(1, sqrt(var))
    # mimicking with simple ops for trace
    # mimicking with simple ops for trace
    denom = ops.add(var, creation.full(var.shape, eps))
    # inv_denom = ops.div(1.0, denom) # not implemented? 
    # Using simple division
    norm = ops.div(d, denom)
    
    # Scale and Shift
    # norm: [B, H] <dp, tp>
    # Gamma: [H] <tp>
    # Beta: [H] <tp>
    # Elementwise should preserve sharding.
    out = ops.add(ops.mul(norm, gamma), beta)
    
    return out

class TestShardedLayerNorm(unittest.TestCase):
    
    def test_sharded_layernorm_trace(self):
        """
        Test sharding propagation through LayerNorm where the normalized axis is Sharded.
        Inputs: [Batch, Hidden] <dp, tp>
        Resolution:
        1. Reduce(Hidden): Requires AllReduce over 'tp'. Result <dp, *>.
        2. Broadcast(Result -> Input): <dp, *> -> <dp, tp>. Valid.
        3. Elementwise: Preserves <dp, tp>.
        """
        print("\n\n================================================================================")
        print("TRACE: Sharded LayerNorm (Reduction over Sharded Axis)")
        print("================================================================================")
        
        # 1. Setup Mesh (2, 2)
        dp, tp = 2, 2
        mesh = get_mesh((dp, tp), ("dp", "tp"))
        
        # 2. Dimensions
        B, H = 4, 16 
        
        # 3. Tensors
        # Input X: [B, H] <dp, tp>
        # Normalized axis (H) is Sharded!
        # Use small values for numerical stability
        X = (Tensor.ones((B, H)) * 0.5).shard(mesh, [DimSpec(["dp"]), DimSpec(["tp"])])
        
        # Gamma, Beta: [H] <tp>
        Gamma = Tensor.ones((H,)).shard(mesh, [DimSpec(["tp"])])
        Beta = Tensor.zeros((H,)).shard(mesh, [DimSpec(["tp"])])
        
        def ln_op(x, g, b):
            return layernorm_manual(x, g, b)
            
        trace = debug.capture_trace(ln_op, X, Gamma, Beta)
        print(trace)
        
        trace_str = str(trace)
        
        # Verification
        # 1. Must see all_reduce (for the sum/mean reductions)
        # 2. Must NOT see all_gather (broadcasting should handle it)
        
        self.assertIn("all_reduce", trace_str, "LayerNorm reduction over sharded axis MUST trigger all_reduce")
        
        # Verify final output sharding
        out_tensor = trace.outputs[0] if isinstance(trace.outputs, (list, tuple)) else trace.outputs
        out_spec = out_tensor.sharding
        # Output should be [dp, tp]
        # Elementwise operations with [tp] weights should preserve [tp].
        self.assertEqual(out_spec.dim_specs[0].axes, ["dp"])
        self.assertEqual(out_spec.dim_specs[1].axes, ["tp"])
        
        # Explicit check that we don't accidentally AllGather inputs
        # (Though implicit 'reshard' might appear if broadcast logic is naive?
        # Ideally, <dp, *> broadcasts to <dp, tp> locally without comms).
        # We allow 'reshard' if it helps, but 'all_gather' on Input X would be bad.
        # But 'all_reduce' is expected.
        
        # Numerical Verification
        import asyncio
        async def verify():
            out_tensor_v = trace.outputs[0] if isinstance(trace.outputs, (list, tuple)) else trace.outputs
            await out_tensor_v.realize
            actual = out_tensor_v.to_numpy()
            
            # Get input values
            await X.realize
            await Gamma.realize
            await Beta.realize
            
            np_x = X.to_numpy()
            np_gamma = Gamma.to_numpy()
            np_beta = Beta.to_numpy()
            
            # NumPy Reference LayerNorm (matching the simplified ops in layernorm_manual)
            # mean = sum(x, axis=-1)
            mean = np.sum(np_x, axis=-1, keepdims=True)
            
            # d = x - mean
            d = np_x - mean
            
            # sq = d * d
            sq = d * d
            
            # var = sum(sq, axis=-1)
            var = np.sum(sq, axis=-1, keepdims=True)
            
            # denom = var + eps
            eps = 1e-5
            denom = var + eps
            
            # norm = d / denom
            norm = d / denom
            
            # out = norm * gamma + beta
            expected = norm * np_gamma + np_beta
            
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            print(f"\n✓ Numerical verification PASSED: Shape {actual.shape}")
            
        asyncio.run(verify())

if __name__ == "__main__":
    unittest.main()
