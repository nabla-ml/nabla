
import unittest
import asyncio
import numpy as np
from nabla.core.tensor import Tensor
from nabla import ops
from nabla.sharding import DeviceMesh, DimSpec
from nabla.core.trace import trace
from nabla.ops import communication

class TestTruePipelineParallelism(unittest.TestCase):
    """
    Tests for True Pipeline Parallelism where stages execute INDEPENDENTLY.
    
    The key difference from previous 'PP' tests is using the 'Stage' dimension
    as a BATCH dimension for the operation, rather than flattening it into
    rows/cols. This prevents the contracting dimension from being sharded,
    thus avoiding AllReduce.
    """

    def test_pp_stage_batch_1d(self):
        """
        Simulate a Pipeline Step:
        Each stage 's' holds an activation x[s] and a weight layer W[s].
        It computes y[s] = x[s] @ W[s].
        Then passes y[s] to passing neighbor.
        
        This corresponds to data-parallel execution of different layers!
        x: [STAGES, D] <stage, *>
        W: [STAGES, D, D] <stage, *, *> 
        """
        print("\n" + "="*80)
        print("TRUE PP TEST: Stage as Batch Dimension")
        print("="*80)
        
        STAGES = 4
        D_MODEL = 8
        
        mesh = DeviceMesh("pp", (STAGES,), ("stage",))
        
        # Inputs:
        # x: Each stage has a vector of size D_MODEL
        # Full shape: [STAGES, D_MODEL]
        x_np = np.random.randn(STAGES, D_MODEL).astype(np.float32)
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        # Weights: Each stage has a layer W of size [D_MODEL, D_MODEL]
        # Full shape: [STAGES, D_MODEL, D_MODEL]
        w_np = np.random.randn(STAGES, D_MODEL, D_MODEL).astype(np.float32) * 0.1
        W = ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])])
        
        def pipeline_step(x, W):
            # x: [S, D]
            # W: [S, D, D]
            # We want batch matmul: x[s] @ W[s]
            # unsqueeze x to [S, 1, D] for matmul broadcasting against [S, D, D]
            x_b = ops.unsqueeze(x, axis=1)
            
            # [S, 1, D] @ [S, D, D] -> [S, 1, D]
            # The contract dim is D. It is NOT sharded (sharding is on S).
            y_b = ops.matmul(x_b, W)
            
            y = ops.squeeze(y_b, axis=1)
            y = ops.relu(y)
            
            # Permute to next stage
            perm = [(i, (i + 1) % STAGES) for i in range(STAGES)]
            return communication.ppermute(y, perm)
        
        print(f"Inputs:" )
        print(f"  x global: {x.shape} local: {x._impl.physical_local_shape(0)}")
        print(f"  W global: {W.shape} local: {W._impl.physical_local_shape(0)}")
        
        # TRACE
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(pipeline_step, x, W)
        print(t)
        print("-" * 60)
        
        # CRITICAL ASSERTION: No AllReduce!
        self.assertNotIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        
        # Verify result
        result = pipeline_step(x, W)
        
        async def verify():
            # NumPy Reference
            # "Batch" matmul manually
            x_local = [x_np[i] for i in range(STAGES)] # list of (D,)
            w_local = [w_np[i] for i in range(STAGES)] # list of (D, D)
            
            y_local = []
            for i in range(STAGES):
                # (1, D) @ (D, D) -> (1, D)
                res = np.maximum(x_local[i] @ w_local[i], 0)
                y_local.append(res)
            
            # PPermute logic
            y_permuted = [None] * STAGES
            for i in range(STAGES):
                # i sends to (i+1)%S
                # receiver r = (i+1)%S gets from i
                # So output[r] = y_local[i]
                y_permuted[(i + 1) % STAGES] = y_local[i]
                
            expected = np.stack(y_permuted, axis=0)
            
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print("\n✅ PASS: Numerical Verification")

        asyncio.run(verify())

    def test_pp_stage_batch_2d(self):
        """
        True Batched PP Test.
        x: [STAGES, BATCH, D] <stage, *, *>
        W: [STAGES, D, D] <stage, *, *>
        """
        print("\n" + "="*80)
        print("TRUE PP TEST: Batched (2D Input)")
        print("="*80)
        
        STAGES = 4
        BATCH = 2
        D_MODEL = 8
        
        mesh = DeviceMesh("pp", (STAGES,), ("stage",))
        
        # Inputs:
        # x: [STAGES, BATCH, D]
        x_np = np.random.randn(STAGES, BATCH, D_MODEL).astype(np.float32)
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])])
        
        # Weights: [STAGES, D, D]
        w_np = np.random.randn(STAGES, D_MODEL, D_MODEL).astype(np.float32) * 0.1
        W = ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec(["stage"]), DimSpec([]), DimSpec([])])
        
        def pipeline_step(x, W):
            # x: [S, B, D]
            # W: [S, D, D]
            # matmul([S, B, D], [S, D, D]) -> [S, B, D]
            # Matmul handles batch broadcasting, so S aligns.
            y = ops.matmul(x, W)
            y = ops.relu(y)
            
            # Permute
            perm = [(i, (i + 1) % STAGES) for i in range(STAGES)]
            return communication.ppermute(y, perm)
            
        print(f"Inputs:" )
        print(f"  x global: {x.shape} local: {x._impl.physical_local_shape(0)}")
        print(f"  W global: {W.shape} local: {W._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(pipeline_step, x, W)
        print(t)
        print("-" * 60)
        
        self.assertNotIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        
        result = pipeline_step(x, W)
        
        async def verify():
            # NumPy Reference
            x_local = [x_np[i] for i in range(STAGES)] # list of (B, D)
            w_local = [w_np[i] for i in range(STAGES)] # list of (D, D)
            
            y_local = []
            for i in range(STAGES):
                # (B, D) @ (D, D) -> (B, D)
                res = np.maximum(x_local[i] @ w_local[i], 0)
                y_local.append(res)
            
            
            y_permuted = [None] * STAGES
            for i in range(STAGES):
                y_permuted[(i + 1) % STAGES] = y_local[i]
                
            expected = np.stack(y_permuted, axis=0)
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print("\n✅ PASS: Numerical Verification")
            
        asyncio.run(verify())

    def test_pp_dp_2d_mesh(self):
        """
        Combined Pipeline Parallelism (PP) and Data Parallelism (DP).
        Mesh: (PP_SIZE, DP_SIZE)
        
        x: [STAGES, BATCH_GLOBAL, D]
           - STAGES sharded on 'pp' (Each stage processes one layer group)
           - BATCH_GLOBAL sharded on 'dp' (Data Parallelism within each stage)
           - Spec: <pp, dp, *>
           
        W: [STAGES, D, D]
           - STAGES sharded on 'pp'
           - Spec: <pp, *, *> (Replicated across DP dims)
        """
        print("\n" + "="*80)
        print("TRUE PP + DP TEST (2D MESH)")
        print("="*80)
        
        PP_SIZE = 2
        DP_SIZE = 2
        STAGES = PP_SIZE
        
        BATCH_PER_DP = 2
        BATCH_GLOBAL = BATCH_PER_DP * DP_SIZE
        D_MODEL = 8
        
        # Mesh: 2x2 grid
        # (0,0) (0,1)
        # (1,0) (1,1)
        mesh = DeviceMesh("cluster", (PP_SIZE, DP_SIZE), ("pp", "dp"))
        
        # Input: [STAGES, BATCH, D]
        x_np = np.random.randn(STAGES, BATCH_GLOBAL, D_MODEL).astype(np.float32)
        # Shard: dim0 -> pp, dim1 -> dp
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec(["pp"]), DimSpec(["dp"]), DimSpec([])])
        
        # Weights: [STAGES, D, D]
        w_np = np.random.randn(STAGES, D_MODEL, D_MODEL).astype(np.float32) * 0.1
        # Shard: dim0 -> pp, others replicated
        W = ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec(["pp"]), DimSpec([]), DimSpec([])])
        
        def hybrid_step(x, W):
            # x: <pp, dp, *>
            # W: <pp, *, *> (broadcasts over dp)
            # Output: <pp, dp, *>
            y = ops.matmul(x, W)
            y = ops.relu(y)
            
            # PP Permutation: Move data from Stage i to (i+1)%S
            # This movement happens for ALL DP ranks in parallel.
            # (p, d) sends to ((p+1)%P, d)
            perm = []
            for p in range(PP_SIZE):
                for d in range(DP_SIZE):
                    src_id = p * DP_SIZE + d
                    next_p = (p + 1) % PP_SIZE
                    dst_id = next_p * DP_SIZE + d
                    perm.append((src_id, dst_id))
            
            return communication.ppermute(y, perm)
            
        print(f"Inputs:" )
        print(f"  x global: {x.shape} local: {x._impl.physical_local_shape(0)}")
        print(f"  W global: {W.shape} local: {W._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(hybrid_step, x, W)
        print(t)
        print("-" * 60)
        
        self.assertNotIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        
        result = hybrid_step(x, W)
        
        async def verify():
            # Calculate Expected Result locally
            # We effectively simulate the parallel grid
            
            # 1. Compute locally per (pp, dp) shard
            y_shards = {} # (p,d) -> result slice
            
            for p in range(PP_SIZE):
                for d in range(DP_SIZE):
                    # Local Inputs
                    # x slice: STAGE=p, BATCH=d*B_local : (d+1)*B_local
                    x_slice = x_np[p, d*BATCH_PER_DP : (d+1)*BATCH_PER_DP, :]
                    w_slice = w_np[p, :, :]
                    
                    # Local Compute
                    y_slice = np.maximum(x_slice @ w_slice, 0)
                    y_shards[(p, d)] = y_slice
            
            # 2. Simulate PPermute
            # (p, d) -> y_final[(p+1)%P, d_slice]
            # We want to reconstruct the GLOBAL [STAGES, BATCH, D]
            
            y_final_check = np.zeros_like(x_np)
            
            for p in range(PP_SIZE):
                for d in range(DP_SIZE):
                    # Data at (p,d) *moves to* ((p+1)%P, d)
                    # So the data currently held at y_shards[(p,d)] IS the data for stage P, batch slice D
                    # WAIT. PPermute effectively rotates the TENSOR CONTENT relative to the DEVICE GRID.
                    # Or does it rotate the DEVICE OWNERSHIP?
                    # x[s] starts at stage s. 
                    # After ppermute, the data at stage s moves to s+1.
                    # So y_shards[(p,d)] contains the calculation result of stage p.
                    # It creates a tensor 'result' where slicing result along PP axis at (p+1) gives this data.
                    
                    target_p = (p + 1) % PP_SIZE
                    
                    # Place into global array
                    batch_start = d * BATCH_PER_DP
                    batch_end = (d + 1) * BATCH_PER_DP
                    y_final_check[target_p, batch_start:batch_end, :] = y_shards[(p, d)]
            
            actual = result.to_numpy()
            np.testing.assert_allclose(actual, y_final_check, rtol=1e-5, atol=1e-5)
            print("\n✅ PASS: Numerical Verification")
            
        asyncio.run(verify())


if __name__ == "__main__":
    unittest.main()
