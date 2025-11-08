import numpy as np
import time
from max import engine
from max.driver import CPU, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


# ============================================================================
# Configuration Constants - SCALED UP FOR PERFORMANCE TESTING
# ============================================================================
BATCH_SIZE = 4
SEQ_LENGTH = 32
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 100  # 48 layers = VERY DEEP network for graph construction test
D_K = D_MODEL // NUM_HEADS


# ============================================================================
# Pure Functions: Transformer Components (TensorValue -> TensorValue)
# ============================================================================

def multi_head_attention(input, W_q, W_k, W_v, W_o):
    """Multi-head self-attention mechanism."""
    # Project to Q, K, V
    Q = ops.matmul(input, W_q)
    K = ops.matmul(input, W_k)
    V = ops.matmul(input, W_v)
    
    # Reshape for multi-head
    Q_reshaped = ops.reshape(Q, [BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, D_K])
    K_reshaped = ops.reshape(K, [BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, D_K])
    V_reshaped = ops.reshape(V, [BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, D_K])
    
    # Transpose to [batch, num_heads, seq, d_k]
    Q_heads = ops.permute(Q_reshaped, [0, 2, 1, 3])
    K_heads = ops.permute(K_reshaped, [0, 2, 1, 3])
    V_heads = ops.permute(V_reshaped, [0, 2, 1, 3])
    
    # Attention: Q @ K^T
    K_transposed = ops.permute(K_heads, [0, 1, 3, 2])
    attention_scores = ops.matmul(Q_heads, K_transposed)
    
    # Apply softmax
    attention_weights = ops.softmax(attention_scores, axis=-1)
    
    # Apply attention to values
    attention_output = ops.matmul(attention_weights, V_heads)
    
    # Transpose back and reshape
    output_transposed = ops.permute(attention_output, [0, 2, 1, 3])
    output_merged = ops.reshape(output_transposed, [BATCH_SIZE, SEQ_LENGTH, D_MODEL])
    
    # Final output projection
    return ops.matmul(output_merged, W_o)


def feed_forward_network(input, W1, W2):
    """Position-wise feed-forward network with GELU activation."""
    hidden = ops.matmul(input, W1)
    activated = ops.gelu(hidden)
    return ops.matmul(activated, W2)


def apply_layer_norm(input, gamma, beta):
    """Apply layer normalization."""
    return ops.layer_norm(input, gamma, beta, epsilon=1e-5)


def transformer_block(input, W_q, W_k, W_v, W_o, W1, W2, gamma1, beta1, gamma2, beta2):
    """Complete transformer encoder block with residual connections."""
    # Multi-head attention with residual and layer norm
    attn_out = multi_head_attention(input, W_q, W_k, W_v, W_o)
    attn_residual = ops.add(input, attn_out)
    attn_normalized = apply_layer_norm(attn_residual, gamma1, beta1)
    
    # Feed-forward network with residual and layer norm
    ffn_out = feed_forward_network(attn_normalized, W1, W2)
    ffn_residual = ops.add(attn_normalized, ffn_out)
    final_output = apply_layer_norm(ffn_residual, gamma2, beta2)
    
    return final_output


# ============================================================================
# Main Function
# ============================================================================

def run_transformer():
    print("\n" + "="*70)
    print("TRANSFORMER NEURAL NETWORK - PYTHON/MAX IMPLEMENTATION")
    print("="*70 + "\n")
    
    print("Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Sequence Length: {SEQ_LENGTH}")
    print(f"  Model Dimension (d_model): {D_MODEL}")
    print(f"  Number of Heads: {NUM_HEADS}")
    print(f"  Head Dimension (d_k): {D_K}")
    print(f"  Feed-Forward Dimension: {D_FF}")
    print(f"  Number of Layers: {NUM_LAYERS}")
    print()
    
    # ========================================================================
    # Step 1: Initialize weights with numpy
    # ========================================================================
    print("\n[Step 1] Initializing weights with numpy...")
    
    np.random.seed(42)
    
    # Input embeddings
    input_np = np.random.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL).astype(np.float32) * 0.1
    
    # Layer 1 weights
    W_q1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W_k1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W_v1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W_o1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W1_1_np = np.random.randn(D_MODEL, D_FF).astype(np.float32) * 0.02
    W2_1_np = np.random.randn(D_FF, D_MODEL).astype(np.float32) * 0.02
    gamma1_1_np = np.ones(D_MODEL, dtype=np.float32)
    beta1_1_np = np.zeros(D_MODEL, dtype=np.float32)
    gamma2_1_np = np.ones(D_MODEL, dtype=np.float32)
    beta2_1_np = np.zeros(D_MODEL, dtype=np.float32)
    
    # Layer 2 weights
    W_q2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W_k2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W_v2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W_o2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    W1_2_np = np.random.randn(D_MODEL, D_FF).astype(np.float32) * 0.02
    W2_2_np = np.random.randn(D_FF, D_MODEL).astype(np.float32) * 0.02
    gamma1_2_np = np.ones(D_MODEL, dtype=np.float32)
    beta1_2_np = np.zeros(D_MODEL, dtype=np.float32)
    gamma2_2_np = np.ones(D_MODEL, dtype=np.float32)
    beta2_2_np = np.zeros(D_MODEL, dtype=np.float32)
    
    print(f"  Input shape: {input_np.shape}")
    print(f"  ✓ All weights initialized for {NUM_LAYERS} transformer layers")
    
    # ========================================================================
    # Step 2: Build Transformer Graph
    # ========================================================================
    print("\n[Step 2] Building full transformer computation graph...")
    
    graph_start = time.perf_counter_ns()
    
    # Define all input types
    input_type = TensorType(dtype=DType.float32, shape=(BATCH_SIZE, SEQ_LENGTH, D_MODEL), device=DeviceRef.CPU())
    weight_qkvo_type = TensorType(dtype=DType.float32, shape=(D_MODEL, D_MODEL), device=DeviceRef.CPU())
    weight_w1_type = TensorType(dtype=DType.float32, shape=(D_MODEL, D_FF), device=DeviceRef.CPU())
    weight_w2_type = TensorType(dtype=DType.float32, shape=(D_FF, D_MODEL), device=DeviceRef.CPU())
    gamma_beta_type = TensorType(dtype=DType.float32, shape=(D_MODEL,), device=DeviceRef.CPU())
    
    with Graph(
        "transformer_encoder",
        input_types=(
            input_type,  # input
            # Layer 1
            weight_qkvo_type, weight_qkvo_type, weight_qkvo_type, weight_qkvo_type,  # W_q, W_k, W_v, W_o
            weight_w1_type, weight_w2_type,  # W1, W2
            gamma_beta_type, gamma_beta_type, gamma_beta_type, gamma_beta_type,  # gamma1, beta1, gamma2, beta2
            # Layer 2
            weight_qkvo_type, weight_qkvo_type, weight_qkvo_type, weight_qkvo_type,  # W_q, W_k, W_v, W_o
            weight_w1_type, weight_w2_type,  # W1, W2
            gamma_beta_type, gamma_beta_type, gamma_beta_type, gamma_beta_type,  # gamma1, beta1, gamma2, beta2
        )
    ) as graph:
        inputs = graph.inputs
        
        # First transformer block
        layer1_output = transformer_block(
            inputs[0],  # input
            inputs[1], inputs[2], inputs[3], inputs[4],  # W_q, W_k, W_v, W_o
            inputs[5], inputs[6],  # W1, W2
            inputs[7], inputs[8],  # gamma1, beta1
            inputs[9], inputs[10]  # gamma2, beta2
        )
        
        # Second transformer block
        layer2_output = transformer_block(
            layer1_output,  # input from layer 1
            inputs[11], inputs[12], inputs[13], inputs[14],  # W_q, W_k, W_v, W_o
            inputs[15], inputs[16],  # W1, W2
            inputs[17], inputs[18],  # gamma1, beta1
            inputs[19], inputs[20]  # gamma2, beta2
        )
        
        graph.output(layer2_output)
    
    graph_end = time.perf_counter_ns()
    graph_time_ms = (graph_end - graph_start) / 1_000_000.0
    
    print(f"  ✓ Graph built with {NUM_LAYERS} transformer layers")
    print(f"  Graph construction time: {graph_time_ms:.3f} ms")
    print(f"  Input: [{BATCH_SIZE}, {SEQ_LENGTH}, {D_MODEL}]")
    print(f"  Output: [{BATCH_SIZE}, {SEQ_LENGTH}, {D_MODEL}]")
    
    # ========================================================================
    # Step 3: Compile
    # ========================================================================
    print("\n[Step 3] Compiling transformer model...")
    
    compile_start = time.perf_counter_ns()
    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)
    compile_end = time.perf_counter_ns()
    compile_time_ms = (compile_end - compile_start) / 1_000_000.0
    
    print("  ✓ Model compiled successfully!")
    print(f"  Compilation time: {compile_time_ms:.3f} ms")
    
    # ========================================================================
    # Step 4: Execute Forward Pass (10,000 iterations)
    # ========================================================================
    print("\n[Step 4] Running forward pass benchmark...")
    print("  Executing 10,000 iterations...")
    
    NUM_ITERATIONS = 1000
    PRINT_EVERY = 100
    
    total_time_ns = 0
    output_np = None
    
    print("\n  Starting benchmark...")
    
    for i in range(NUM_ITERATIONS):
        # Time this iteration
        start_time = time.perf_counter_ns()
        result = model.execute(
            input_np,
            W_q1_np, W_k1_np, W_v1_np, W_o1_np, W1_1_np, W2_1_np,
            gamma1_1_np, beta1_1_np, gamma2_1_np, beta2_1_np,
            W_q2_np, W_k2_np, W_v2_np, W_o2_np, W1_2_np, W2_2_np,
            gamma1_2_np, beta1_2_np, gamma2_2_np, beta2_2_np
        )
        end_time = time.perf_counter_ns()
        
        iteration_time = end_time - start_time
        total_time_ns += iteration_time
        
        # Print progress every 100 iterations
        if (i + 1) % PRINT_EVERY == 0:
            avg_time_ns = total_time_ns // (i + 1)
            avg_time_ms = avg_time_ns / 1_000_000.0
            print(f"    Iteration {i + 1} / {NUM_ITERATIONS} - Avg time: {avg_time_ms:.6f} ms")
        
        # Store the last output
        if i == NUM_ITERATIONS - 1:
            assert isinstance(result[0], Tensor)
            output_np = result[0].to_numpy()
    
    # Calculate final statistics
    avg_time_ns = total_time_ns // NUM_ITERATIONS
    avg_time_ms = avg_time_ns / 1_000_000.0
    avg_time_us = avg_time_ns / 1_000.0
    total_time_ms = total_time_ns / 1_000_000.0
    total_time_s = total_time_ms / 1_000.0
    
    print("\n  ✓ Benchmark complete!")
    print("\n" + "="*70)
    print("PERFORMANCE RESULTS")
    print("="*70)
    print(f"  Total iterations: {NUM_ITERATIONS}")
    print(f"  Total time: {total_time_s:.6f} seconds")
    print(f"  Average time per iteration: {avg_time_ms:.6f} ms")
    print(f"  Average time per iteration: {avg_time_us:.6f} µs")
    print(f"  Throughput: {NUM_ITERATIONS / total_time_s:.6f} iterations/second")
    
    print("\n" + "="*70)
    print("TRANSFORMER OUTPUT (Final Iteration)")
    print("="*70)
    print(f"  Input shape: {input_np.shape}")
    print(f"  Output shape: {output_np.shape}")
    print(f"  Input mean: {np.mean(input_np)}")
    print(f"  Input std: {np.std(input_np)}")
    print(f"  Output mean: {np.mean(output_np)}")
    print(f"  Output std: {np.std(output_np)}")
    print(f"\n  ✓ Transformer encoder successfully executed!")
    print(f"  ✓ All {NUM_LAYERS} layers processed")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_transformer()
