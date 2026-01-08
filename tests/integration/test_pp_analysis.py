"""Pipeline Parallelism Tests with Tracing.

Contains:
1. Unbatched 2D MLP PP test (like the original)
2. Unbatched 2D Transformer block PP test  
3. PP with replicated batch dimension test
"""

import unittest
import asyncio
import numpy as np
from nabla.core.tensor import Tensor
from nabla import ops
from nabla.sharding import DeviceMesh, DimSpec
from nabla.core.trace import trace
from nabla.ops import communication


class TestPPAnalysis(unittest.TestCase):
    """Pipeline Parallelism analysis tests."""
    
    def test_unbatched_2d_mlp_pp(self):
        """Unbatched 2D MLP: same pattern as original test_pp_sharding.
        
        h: (SEQ_LEN * STAGES, D_MODEL) sharded <stage, *> -> (SEQ_LEN, D_MODEL)
        W: (D_MODEL * STAGES, D_MODEL) sharded <stage, *> -> (D_MODEL, D_MODEL)
        
        Each stage has different seq positions and different layer weights.
        No AllReduce - complete local computation.
        """
        print("\n" + "="*80)
        print("UNBATCHED 2D MLP PP")
        print("="*80)
        
        SEQ_LEN = 4
        D_MODEL = 8
        NUM_STAGES = 4
        
        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        # Concatenate along first axis, shard on stage
        h_np = np.random.randn(SEQ_LEN * NUM_STAGES, D_MODEL).astype(np.float32) * 0.01
        h = ops.shard(Tensor.from_dlpack(h_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        W_np = np.random.randn(D_MODEL * NUM_STAGES, D_MODEL).astype(np.float32) * 0.1
        W = ops.shard(Tensor.from_dlpack(W_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        def mlp(h, W):
            return ops.relu(h @ W)
        
        def single_pass(h, W):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            h = mlp(h, W)
            h = communication.ppermute(h, perm)
            return h
        
        print(f"\nShapes:")
        print(f"  h: {h.shape} <stage, *> local {h._impl.physical_local_shape(0)}")
        print(f"  W: {W.shape} <stage, *> local {W._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(single_pass, h, W)
        print(t)
        print("-" * 60)
        
        self.assertNotIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        
        result = single_pass(h, W)
        self.assertEqual(tuple(int(d) for d in result.shape), (SEQ_LEN * NUM_STAGES, D_MODEL))
        print(f"\n✅ PASS: Result shape {result.shape}")

        # --- Numerical Verification ---
        async def verify():
            # NumPy Reference
            h_shards = np.split(h_np, NUM_STAGES, axis=0)
            w_shards = np.split(W_np, NUM_STAGES, axis=0)
            
            shard_results = []
            for i in range(NUM_STAGES):
                # Local computation per stage
                res = np.maximum(h_shards[i] @ w_shards[i], 0) # ReLU
                shard_results.append(res)
            
            # Simulate PP Permute: i -> (i+1)%N
            # Stage k receives from k-1
            permuted_results = [None] * NUM_STAGES
            for i in range(NUM_STAGES):
                target = (i + 1) % NUM_STAGES
                permuted_results[target] = shard_results[i]
            
            expected = np.concatenate(permuted_results, axis=0)
            
            # Get Nabla result
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print("✅ PASS: Numerical verification")

        asyncio.run(verify())
    def test_unbatched_2d_transformer_pp(self):
        """Unbatched 2D Transformer block: attention + FFN.
        
        Same sharding pattern as MLP - concatenate on first axis.
        """
        print("\n" + "="*80)
        print("UNBATCHED 2D TRANSFORMER PP")
        print("="*80)
        
        SEQ_LEN = 4
        D_MODEL = 8
        D_FF = 16
        NUM_STAGES = 4
        
        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        def attention_2d(Q, K, V):
            """2D attention: (seq, d) @ (d, seq) -> (seq, seq)"""
            scores = Q @ ops.view.swap_axes(K, 0, 1)
            scores = scores / np.sqrt(D_MODEL)
            attn_weights = ops.softmax(scores, axis=-1)
            return attn_weights @ V
        
        def transformer_2d(x, Wq, Wk, Wv, Wo, W1, W2):
            """2D transformer block."""
            Q = x @ Wq
            K = x @ Wk  
            V = x @ Wv
            attn_out = attention_2d(Q, K, V) @ Wo
            x = x + attn_out
            
            ffn_out = x @ W1
            ffn_out = ops.relu(ffn_out)
            ffn_out = ffn_out @ W2
            x = x + ffn_out
            
            return x
        
        def single_pass(x, Wq, Wk, Wv, Wo, W1, W2):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            x = transformer_2d(x, Wq, Wk, Wv, Wo, W1, W2)
            x = communication.ppermute(x, perm)
            return x
        
        # Input and weights - all sharded on first axis
        x_np = np.random.randn(SEQ_LEN * NUM_STAGES, D_MODEL).astype(np.float32) * 0.01
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        def make_weight(d_in, d_out):
            w_np = np.random.randn(d_in * NUM_STAGES, d_out).astype(np.float32) * 0.1
            return ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        Wq = make_weight(D_MODEL, D_MODEL)
        Wk = make_weight(D_MODEL, D_MODEL)
        Wv = make_weight(D_MODEL, D_MODEL)
        Wo = make_weight(D_MODEL, D_MODEL)
        W1 = make_weight(D_MODEL, D_FF)
        W2 = make_weight(D_FF, D_MODEL)
        
        print(f"\nShapes:")
        print(f"  x: {x.shape} local {x._impl.physical_local_shape(0)}")
        print(f"  Wq: {Wq.shape} local {Wq._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(single_pass, x, Wq, Wk, Wv, Wo, W1, W2)
        print(t)
        print("-" * 60)
        
        self.assertNotIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        self.assertIn("softmax", str(t))
        
        result = single_pass(x, Wq, Wk, Wv, Wo, W1, W2)
        self.assertEqual(tuple(int(d) for d in result.shape), (SEQ_LEN * NUM_STAGES, D_MODEL))
        print(f"\n✅ PASS: Result shape {result.shape}")

        # --- Numerical Verification ---
        async def verify():
            # Split inputs and weights per stage
            x_shards = np.split(x_np, NUM_STAGES, axis=0)
            
            # Wq was created sharded, so convert back then split
            wq_shards = np.split(Wq.to_numpy(), NUM_STAGES, axis=0)
            
            # Helper to split based on "make_weight" logic (sharding on dim 0)
            def split_w(t_val):
                return np.split(t_val.to_numpy(), NUM_STAGES, axis=0)
            
            sq = split_w(Wq); sk = split_w(Wk); sv = split_w(Wv); so = split_w(Wo)
            s1 = split_w(W1); s2 = split_w(W2)
            
            shard_results = []
            for i in range(NUM_STAGES):
                xi = x_shards[i]
                
                # Attention
                Q = xi @ sq[i]
                K = xi @ sk[i]
                V = xi @ sv[i]
                scores = Q @ K.swapaxes(0, 1) / np.sqrt(D_MODEL)
                # Softmax on last axis
                e_x = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                attn = e_x / np.sum(e_x, axis=-1, keepdims=True)
                attn_out = (attn @ V) @ so[i]
                
                h = xi + attn_out
                
                # FFN
                ff = np.maximum(h @ s1[i], 0)
                ff_out = ff @ s2[i]
                
                res = h + ff_out
                shard_results.append(res)
            
            # Permute
            permuted = [None] * NUM_STAGES
            for i in range(NUM_STAGES):
                permuted[(i + 1) % NUM_STAGES] = shard_results[i]
            
            expected = np.concatenate(permuted, axis=0)
            actual = result.to_numpy() # This triggers execution
            
            # Tolerances for float32 accumulation
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            print("✅ PASS: Numerical verification")
        
        asyncio.run(verify())
    
    def test_pp_with_replicated_batch(self):
        """PP with replicated batch dimension.
        
        x: (BATCH, D * STAGES) sharded <*, stage> -> (BATCH, D) per stage
        W: (D * STAGES, OUT) sharded <stage, *> -> (D, OUT) per stage
        
        Batch is replicated, data axis is sharded.
        """
        print("\n" + "="*80)
        print("PP WITH REPLICATED BATCH")
        print("="*80)
        
        BATCH = 2
        D_MODEL = 8
        D_OUT = 4
        NUM_STAGES = 4
        
        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        # Shard LAST axis (data), replicate batch
        x_np = np.random.randn(BATCH, D_MODEL * NUM_STAGES).astype(np.float32)
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec([]), DimSpec(["stage"])])
        
        w_np = np.random.randn(D_MODEL * NUM_STAGES, D_OUT).astype(np.float32)
        W = ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        def single_pass(x, W):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            y = x @ W
            y = communication.ppermute(y, perm)
            return y
        
        print(f"\nShapes:")
        print(f"  x: {x.shape} <*, stage> local {x._impl.physical_local_shape(0)}")
        print(f"  W: {W.shape} <stage, *> local {W._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(single_pass, x, W)
        print(t)
        print("-" * 60)
        
        self.assertIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        
        result = single_pass(x, W)
        self.assertEqual(tuple(int(d) for d in result.shape), (BATCH, D_OUT))
        print(f"\n✅ PASS: Result shape {result.shape}")

        async def verify():
            # x split on 1 (data), W split on 0 (input)
            x_shards = np.split(x_np, NUM_STAGES, axis=1)
            w_shards = np.split(w_np, NUM_STAGES, axis=0)
            
            # Partial sums
            # With all_reduce, inputs are summed
            total_y = sum(x_s @ w_s for x_s, w_s in zip(x_shards, w_shards))
            
            # ppermute rotates replicated result -> identity
            expected = total_y
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print("✅ PASS: Numerical verification")
        
        asyncio.run(verify())
    
    def test_mlp_batched_shard_data_dim(self):
        """Full MLP with batched input, data concatenated/sharded on dim 1.
        
        This is the key pattern you described:
        - Batch dimension (dim 0): REPLICATED - same batch on all stages
        - Data dimension (dim 1): CONCATENATED and SHARDED - different data per stage
        
        x: (BATCH, D_IN * STAGES) sharded <*, stage> -> local (BATCH, D_IN)
        W1: (D_IN * STAGES, D_HIDDEN) sharded <stage, *> -> local (D_IN, D_HIDDEN)
        W2: (D_HIDDEN * STAGES, D_OUT) sharded <stage, *> -> local (D_HIDDEN, D_OUT)
        """
        print("\n" + "="*80)
        print("FULL MLP PP - Batched input, data sharded on dim 1")
        print("="*80)
        
        BATCH = 2    # Replicated across stages
        D_IN = 8     # Input features per stage
        D_HIDDEN = 16
        D_OUT = 4
        NUM_STAGES = 4
        
        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        # x: batch replicated, data sharded on dim 1
        # (BATCH, D_IN * STAGES) = (2, 32) -> local (2, 8) per stage
        x_np = np.random.randn(BATCH, D_IN * NUM_STAGES).astype(np.float32)
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec([]), DimSpec(["stage"])])
        
        # W1: input dim sharded to match x dim 1
        # (D_IN * STAGES, D_HIDDEN) = (32, 16) -> local (8, 16) per stage
        w1_np = np.random.randn(D_IN * NUM_STAGES, D_HIDDEN).astype(np.float32)
        W1 = ops.shard(Tensor.from_dlpack(w1_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        # W2: NOT sharded on input dim - hidden activations are local per stage
        # Each stage has its own W2, so we concatenate on input dim but that's for stacking
        # Actually, h is (batch, hidden) replicated after first matmul, so W2 should be replicated too
        # (D_HIDDEN, D_OUT) = (16, 4) replicated per stage
        w2_np = np.random.randn(D_HIDDEN, D_OUT).astype(np.float32)
        W2 = ops.shard(Tensor.from_dlpack(w2_np), mesh, [DimSpec([]), DimSpec([])])
        
        def mlp(x, W1, W2):
            h = x @ W1      # (batch, d_in) @ (d_in, hidden) = (batch, hidden)
            h = ops.relu(h)
            y = h @ W2      # (batch, hidden) @ (hidden, out) = (batch, out)
            return y
        
        def single_pass(x, W1, W2):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            y = mlp(x, W1, W2)
            y = communication.ppermute(y, perm)
            return y
        
        print(f"\nShapes:")
        print(f"  x: {x.shape} <*, stage> local {x._impl.physical_local_shape(0)}")
        print(f"  W1: {W1.shape} <stage, *> local {W1._impl.physical_local_shape(0)}")
        print(f"  W2: {W2.shape} <stage, *> local {W2._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(single_pass, x, W1, W2)
        print(t)
        print("-" * 60)
        
        self.assertIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        self.assertIn("relu", str(t))
        
        result = single_pass(x, W1, W2)
        self.assertEqual(tuple(int(d) for d in result.shape), (BATCH, D_OUT))
        print(f"\n✅ PASS: Result shape {result.shape}")
        
        async def verify():
            # x split on 1, W1 split on 0
            x_shards = np.split(x_np, NUM_STAGES, axis=1)
            w1_shards = np.split(w1_np, NUM_STAGES, axis=0)
            w2_whole = w2_np # W2 is replicated
            
            # Calculate global sum of partials (AllReduce)
            h_partials = [x_shards[i] @ w1_shards[i] for i in range(NUM_STAGES)]
            h_total = sum(h_partials)
            
            # Apply ReLU on total
            h_total = np.maximum(h_total, 0)
            
            # Apply W2
            final_y = h_total @ w2_whole
            
            # ppermute -> identity
            expected = final_y
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print("✅ PASS: Numerical verification")

        asyncio.run(verify())
    
    def test_mlp_unbatched_pp(self):
        """Full MLP with NO batch - single 1D input per stage.
        
        x: (D_IN * STAGES,) sharded <stage> -> local (D_IN,)
        W1: (D_IN * STAGES, D_HIDDEN) sharded <stage, *> -> local (D_IN, D_HIDDEN)
        W2: (D_HIDDEN, D_OUT) replicated <*, *> -> local (D_HIDDEN, D_OUT)
        
        This is the simplest PP case: 1D vectors through an MLP.
        """
        print("\n" + "="*80)
        print("FULL MLP PP - Unbatched (1D input)")
        print("="*80)
        
        D_IN = 8     # Input features per stage
        D_HIDDEN = 16
        D_OUT = 4
        NUM_STAGES = 4
        
        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        # x: 1D vector, sharded on its only axis
        # (D_IN * STAGES,) = (32,) -> local (8,) per stage
        x_np = np.random.randn(D_IN * NUM_STAGES).astype(np.float32)
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec(["stage"])])
        
        # W1: input dim sharded to match x
        # (D_IN * STAGES, D_HIDDEN) = (32, 16) -> local (8, 16) per stage
        w1_np = np.random.randn(D_IN * NUM_STAGES, D_HIDDEN).astype(np.float32)
        W1 = ops.shard(Tensor.from_dlpack(w1_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        # W2: fully replicated
        # (D_HIDDEN, D_OUT) = (16, 4) replicated per stage
        w2_np = np.random.randn(D_HIDDEN, D_OUT).astype(np.float32)
        W2 = ops.shard(Tensor.from_dlpack(w2_np), mesh, [DimSpec([]), DimSpec([])])
        
        def mlp(x, W1, W2):
            h = x @ W1      # (d_in,) @ (d_in, hidden) = (hidden,)
            h = ops.relu(h)
            y = h @ W2      # (hidden,) @ (hidden, out) = (out,)
            return y
        
        def single_pass(x, W1, W2):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            y = mlp(x, W1, W2)
            y = communication.ppermute(y, perm)
            return y
        
        print(f"\nShapes:")
        print(f"  x: {x.shape} <stage> local {x._impl.physical_local_shape(0)}")
        print(f"  W1: {W1.shape} <stage, *> local {W1._impl.physical_local_shape(0)}")
        print(f"  W2: {W2.shape} <*, *> local {W2._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(single_pass, x, W1, W2)
        print(t)
        print("-" * 60)
        
        self.assertIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        self.assertIn("relu", str(t))
        
        result = single_pass(x, W1, W2)
        self.assertEqual(tuple(int(d) for d in result.shape), (D_OUT,))
        print(f"\n✅ PASS: Result shape {result.shape}")

        async def verify():
            # x split on 0 (only axis), W1 split on 0
            x_shards = np.split(x_np, NUM_STAGES, axis=0)
            w1_shards = np.split(w1_np, NUM_STAGES, axis=0)
            w2_whole = w2_np
            
            # Calculate global sum of partials (AllReduce)
            h_partials = [x_shards[i] @ w1_shards[i] for i in range(NUM_STAGES)]
            h_total = sum(h_partials)
            
            # Apply ReLU and W2
            h_relu = np.maximum(h_total, 0)
            final_y = h_relu @ w2_whole
            
            # ppermute -> identity
            expected = final_y
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print("✅ PASS: Numerical verification")

        asyncio.run(verify())

    def test_transformer_attention_block_pp(self):
        """Full Transformer Attention Block with PP.
        
        Batched input with sequence dimension sharded across stages:
        - x: (BATCH, SEQ * STAGES, D_MODEL) sharded <*, stage, *> -> local (BATCH, SEQ, D_MODEL)
        - Wq/Wk/Wv/Wo: (D_MODEL * STAGES, D_MODEL) sharded <stage, *> -> local (D_MODEL, D_MODEL)
        - W1/W2: FFN weights similarly sharded
        
        This models PP where each stage processes a different sequence chunk.
        """
        print("\n" + "="*80)
        print("TRANSFORMER ATTENTION BLOCK PP")
        print("="*80)
        
        BATCH = 2
        SEQ = 4         # Sequence length per stage
        D_MODEL = 8
        D_FF = 16
        NUM_STAGES = 4
        
        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        # Input: batch replicated, sequence sharded
        # (BATCH, SEQ * STAGES, D_MODEL) = (2, 16, 8) -> local (2, 4, 8) per stage
        x_np = np.random.randn(BATCH, SEQ * NUM_STAGES, D_MODEL).astype(np.float32) * 0.01
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec([]), DimSpec(["stage"]), DimSpec([])])
        
        # Projection weights: input dim sharded to match x's sequence/embedding
        # For attention, we treat the flattened (seq, d_model) as one dimension conceptually
        # But practically: (D_MODEL * STAGES, D_MODEL) -> local (D_MODEL, D_MODEL)
        def make_proj_weight():
            w_np = np.random.randn(D_MODEL * NUM_STAGES, D_MODEL).astype(np.float32) * 0.1
            return ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        # Simpler approach: Keep projection weights replicated,
        # only the input sequence is sharded. Each stage processes its local sequence chunk.
        def make_weight(d_in, d_out):
            w_np = np.random.randn(d_in, d_out).astype(np.float32) * 0.1
            return ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec([]), DimSpec([])])
        
        Wq = make_weight(D_MODEL, D_MODEL)
        Wk = make_weight(D_MODEL, D_MODEL)
        Wv = make_weight(D_MODEL, D_MODEL)
        Wo = make_weight(D_MODEL, D_MODEL)
        W1 = make_weight(D_MODEL, D_FF)
        W2 = make_weight(D_FF, D_MODEL)
        
        def attention(x, Wq, Wk, Wv, Wo):
            """Self-attention: (batch, seq, d) -> (batch, seq, d)"""
            Q = x @ Wq  # (batch, seq, d) @ (d, d) -> (batch, seq, d)
            K = x @ Wk
            V = x @ Wv
            
            # Attention scores: Q @ K^T
            K_T = ops.view.swap_axes(K, -1, -2)  # (batch, d, seq)
            scores = Q @ K_T  # (batch, seq, seq)
            scores = scores / np.sqrt(D_MODEL)
            attn_weights = ops.softmax(scores, axis=-1)
            
            # Apply attention
            attn_out = attn_weights @ V  # (batch, seq, d)
            return attn_out @ Wo
        
        def ffn(x, W1, W2):
            """FFN: (batch, seq, d) -> (batch, seq, d)"""
            h = x @ W1  # (batch, seq, d_ff)
            h = ops.relu(h)
            return h @ W2  # (batch, seq, d)
        
        def transformer_block(x, Wq, Wk, Wv, Wo, W1, W2):
            """Full transformer block with residual connections."""
            # Self-attention with residual
            attn_out = attention(x, Wq, Wk, Wv, Wo)
            x = x + attn_out
            
            # FFN with residual
            ffn_out = ffn(x, W1, W2)
            x = x + ffn_out
            
            return x
        
        def single_pass(x, Wq, Wk, Wv, Wo, W1, W2):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            y = transformer_block(x, Wq, Wk, Wv, Wo, W1, W2)
            y = communication.ppermute(y, perm)
            return y
        
        print(f"\nShapes:")
        print(f"  x: {x.shape} <*, stage, *> local {x._impl.physical_local_shape(0)}")
        print(f"  Wq: {Wq.shape} <*, *> local {Wq._impl.physical_local_shape(0)}")
        print(f"  W1: {W1.shape} <*, *> local {W1._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(single_pass, x, Wq, Wk, Wv, Wo, W1, W2)
        print(t)
        print("-" * 60)
        
        # Should have attention ops, no all_reduce needed
        self.assertIn("ppermute", str(t))
        self.assertIn("softmax", str(t))
        self.assertIn("relu", str(t))
        
        result = single_pass(x, Wq, Wk, Wv, Wo, W1, W2)
        self.assertEqual(tuple(int(d) for d in result.shape), (BATCH, SEQ * NUM_STAGES, D_MODEL))
        print(f"\n✅ PASS: Result shape {result.shape}")

        async def verify():
            # x split on 1 (Seq), weights are replicated (use full)
            x_shards = np.split(x_np, NUM_STAGES, axis=1)
            wq = Wq.to_numpy()
            wk = Wk.to_numpy()
            wv = Wv.to_numpy()
            wo = Wo.to_numpy()
            w1 = W1.to_numpy()
            w2 = W2.to_numpy()
            
            shard_results = []
            for i in range(NUM_STAGES):
                xi = x_shards[i]
                
                # Attention (Local)
                Q = xi @ wq
                K = xi @ wk
                V = xi @ wv
                scores = Q @ K.swapaxes(-1, -2) / np.sqrt(D_MODEL)
                e_x = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                attn = e_x / np.sum(e_x, axis=-1, keepdims=True)
                attn_out = (attn @ V) @ wo
                
                h = xi + attn_out
                
                # FFN
                ff = np.maximum(h @ w1, 0)
                ff_out = ff @ w2
                
                res = h + ff_out
                shard_results.append(res)
            
            # Permute
            permuted = [None] * NUM_STAGES
            for i in range(NUM_STAGES):
                permuted[(i + 1) % NUM_STAGES] = shard_results[i]
            
            # Reconstruct global sequence
            expected = np.concatenate(permuted, axis=1)
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            print("✅ PASS: Numerical verification")
        
        asyncio.run(verify())

    def test_1d_input_pp(self):
        """1D input concatenated on its single axis.
        
        This is the true pipeline case with no batch dimension:
        x: (D * STAGES,) sharded <stage> -> (D,) per stage
        W: (D * STAGES, OUT) sharded <stage, *> -> (D, OUT) per stage
        
        The 1D input gets promoted to 2D for matmul, then squeezed back.
        """
        print("\n" + "="*80)
        print("1D INPUT PP - Concatenated on single axis")
        print("="*80)
        
        D_MODEL = 8
        D_OUT = 4
        NUM_STAGES = 4
        
        mesh = DeviceMesh("pp", (NUM_STAGES,), ("stage",))
        
        # 1D input: concatenate on its only axis
        # (D_MODEL * STAGES,) = (32,) -> (8,) per stage
        x_np = np.random.randn(D_MODEL * NUM_STAGES).astype(np.float32)
        x = ops.shard(Tensor.from_dlpack(x_np), mesh, [DimSpec(["stage"])])
        
        # W: shard on first axis to match x
        # (D_MODEL * STAGES, D_OUT) = (32, 4) -> (8, 4) per stage
        w_np = np.random.randn(D_MODEL * NUM_STAGES, D_OUT).astype(np.float32)
        W = ops.shard(Tensor.from_dlpack(w_np), mesh, [DimSpec(["stage"]), DimSpec([])])
        
        def single_pass(x, W):
            perm = [(i, (i + 1) % NUM_STAGES) for i in range(NUM_STAGES)]
            y = x @ W  # 1D @ 2D -> 1D: (D,) @ (D, OUT) -> (OUT,)
            y = communication.ppermute(y, perm)
            return y
        
        print(f"\nShapes:")
        print(f"  x: {x.shape} <stage> local {x._impl.physical_local_shape(0)}")
        print(f"  W: {W.shape} <stage, *> local {W._impl.physical_local_shape(0)}")
        
        print("\n📊 TRACE:")
        print("-" * 60)
        t = trace(single_pass, x, W)
        print(t)
        print("-" * 60)
        
        self.assertIn("all_reduce", str(t))
        self.assertIn("ppermute", str(t))
        
        result = single_pass(x, W)
        # Output is 1D: (D_OUT,) = (4,) but replicated
        self.assertEqual(tuple(int(d) for d in result.shape), (D_OUT,))
        print(f"\n✅ PASS: Result shape {result.shape}")
        
        async def verify():
            # x split on 0 (only axis), W split on 0
            x_shards = np.split(x_np, NUM_STAGES, axis=0)
            w_shards = np.split(w_np, NUM_STAGES, axis=0)
            
            # 1D @ 2D -> 1D partial sum
            # With all_reduce, we sum these up
            total_y = sum(x_shards[i] @ w_shards[i] for i in range(NUM_STAGES))
            
            # ppermute rotates replicated tensor -> effectively identity
            expected = total_y
            actual = result.to_numpy()
            
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print("✅ PASS: Numerical verification")

        asyncio.run(verify())


if __name__ == "__main__":
    unittest.main(verbosity=2)
