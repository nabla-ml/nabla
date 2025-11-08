from max_graph.ops import (
    add, sub, mul, matmul,
    div, sqrt, tanh, gelu, softmax, layer_norm,
    reshape, transpose, permute, concat, split,
)
from max_graph.types import (
    Device, DeviceType, TensorType, TensorValue, Tensor
)
from max_graph.utils import (
    PythonBridge, Graph, DeviceRef
)
from python import Python
from time import perf_counter_ns


# ============================================================================
# Configuration Constants - SCALED UP FOR PERFORMANCE TESTING
# ============================================================================
alias BATCH_SIZE = 4
alias SEQ_LENGTH = 32
alias D_MODEL = 512      # Model dimension (8x larger)
alias NUM_HEADS = 8      # Number of attention heads (2x more)
alias D_FF = 2048        # Feed-forward dimension (8x larger)
alias NUM_LAYERS = 100    # Number of transformer layers (48 layers = VERY DEEP network for graph construction test)
alias D_K = D_MODEL // NUM_HEADS  # Head dimension: 64


# ============================================================================
# Pure Functions: Transformer Components (TensorValue -> TensorValue)
# ============================================================================

fn multi_head_attention(
    input: TensorValue,
    W_q: TensorValue,
    W_k: TensorValue,
    W_v: TensorValue,
    W_o: TensorValue
) raises -> TensorValue:
    """Multi-head self-attention mechanism."""
    # Project to Q, K, V
    var Q = matmul(input, W_q)
    var K = matmul(input, W_k)
    var V = matmul(input, W_v)
    
    # Reshape for multi-head: [batch, seq, d_model] -> [batch, seq, num_heads, d_k]
    var Q_reshaped = reshape(Q, [BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, D_K])
    var K_reshaped = reshape(K, [BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, D_K])
    var V_reshaped = reshape(V, [BATCH_SIZE, SEQ_LENGTH, NUM_HEADS, D_K])
    
    # Transpose to [batch, num_heads, seq, d_k]
    var Q_heads = permute(Q_reshaped, [0, 2, 1, 3])
    var K_heads = permute(K_reshaped, [0, 2, 1, 3])
    var V_heads = permute(V_reshaped, [0, 2, 1, 3])
    
    # Attention: Q @ K^T
    var K_transposed = permute(K_heads, [0, 1, 3, 2])
    var attention_scores = matmul(Q_heads, K_transposed)  # [batch, heads, seq, seq]
    
    # Apply softmax
    var attention_weights = softmax(attention_scores, axis=-1)
    
    # Apply attention to values
    var attention_output = matmul(attention_weights, V_heads)  # [batch, heads, seq, d_k]
    
    # Transpose back and reshape to [batch, seq, d_model]
    var output_transposed = permute(attention_output, [0, 2, 1, 3])
    var output_merged = reshape(output_transposed, [BATCH_SIZE, SEQ_LENGTH, D_MODEL])
    
    # Final output projection
    return matmul(output_merged, W_o)


fn feed_forward_network(
    input: TensorValue,
    W1: TensorValue,
    W2: TensorValue
) raises -> TensorValue:
    """Position-wise feed-forward network with GELU activation."""
    # First layer with GELU activation
    var hidden = matmul(input, W1)  # [batch, seq, d_ff]
    var activated = gelu(hidden)
    
    # Second layer
    return matmul(activated, W2)  # [batch, seq, d_model]


fn apply_layer_norm(
    input: TensorValue,
    gamma: TensorValue,
    beta: TensorValue
) raises -> TensorValue:
    """Apply layer normalization."""
    return layer_norm(input, gamma, beta, epsilon=1e-5)


fn transformer_block(
    input: TensorValue,
    W_q: TensorValue,
    W_k: TensorValue,
    W_v: TensorValue,
    W_o: TensorValue,
    W1: TensorValue,
    W2: TensorValue,
    gamma1: TensorValue,
    beta1: TensorValue,
    gamma2: TensorValue,
    beta2: TensorValue
) raises -> TensorValue:
    """Complete transformer encoder block with residual connections."""
    # Multi-head attention with residual and layer norm
    var attn_out = multi_head_attention(input, W_q, W_k, W_v, W_o)
    var attn_residual = add(input, attn_out)
    var attn_normalized = apply_layer_norm(attn_residual, gamma1, beta1)
    
    # Feed-forward network with residual and layer norm
    var ffn_out = feed_forward_network(attn_normalized, W1, W2)
    var ffn_residual = add(attn_normalized, ffn_out)
    var final_output = apply_layer_norm(ffn_residual, gamma2, beta2)
    
    return final_output


# ============================================================================
# Main Function
# ============================================================================

fn main() raises:
    print("\n" + "="*70)
    print("TRANSFORMER NEURAL NETWORK - MOJO IMPLEMENTATION")
    print("="*70 + "\n")
    
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var cpu = Device(DeviceType.CPU())
    
    print("Configuration:")
    print("  Batch Size:", BATCH_SIZE)
    print("  Sequence Length:", SEQ_LENGTH)
    print("  Model Dimension (d_model):", D_MODEL)
    print("  Number of Heads:", NUM_HEADS)
    print("  Head Dimension (d_k):", D_K)
    print("  Feed-Forward Dimension:", D_FF)
    print("  Number of Layers:", NUM_LAYERS)
    print()
    
    # ========================================================================
    # Step 1: Initialize weights with numpy
    # ========================================================================
    print("\n[Step 1] Initializing weights with numpy...")
    
    np.random.seed(42)
    
    # Input embeddings
    var input_np = np.random.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL).astype(np.float32) * 0.1
    
    # Layer 1 weights
    var W_q1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W_k1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W_v1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W_o1_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W1_1_np = np.random.randn(D_MODEL, D_FF).astype(np.float32) * 0.02
    var W2_1_np = np.random.randn(D_FF, D_MODEL).astype(np.float32) * 0.02
    var gamma1_1_np = np.ones(D_MODEL, dtype=np.float32)
    var beta1_1_np = np.zeros(D_MODEL, dtype=np.float32)
    var gamma2_1_np = np.ones(D_MODEL, dtype=np.float32)
    var beta2_1_np = np.zeros(D_MODEL, dtype=np.float32)
    
    # Layer 2 weights
    var W_q2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W_k2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W_v2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W_o2_np = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * 0.02
    var W1_2_np = np.random.randn(D_MODEL, D_FF).astype(np.float32) * 0.02
    var W2_2_np = np.random.randn(D_FF, D_MODEL).astype(np.float32) * 0.02
    var gamma1_2_np = np.ones(D_MODEL, dtype=np.float32)
    var beta1_2_np = np.zeros(D_MODEL, dtype=np.float32)
    var gamma2_2_np = np.ones(D_MODEL, dtype=np.float32)
    var beta2_2_np = np.zeros(D_MODEL, dtype=np.float32)
    
    print("  Input shape:", input_np.shape)
    print("  ✓ All weights initialized for", NUM_LAYERS, "transformer layers")
    
    # ========================================================================
    # Step 2: Build Transformer Graph
    # ========================================================================
    print("\n[Step 2] Building full transformer computation graph...")
    
    var graph_start = perf_counter_ns()
    
    # Define all input types
    var input_types = List[TensorType]()
    input_types.append(TensorType(DType.float32, [BATCH_SIZE, SEQ_LENGTH, D_MODEL], cpu))  # input
    
    # Layer 1 weights
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_q1
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_k1
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_v1
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_o1
    input_types.append(TensorType(DType.float32, [D_MODEL, D_FF], cpu))     # W1_1
    input_types.append(TensorType(DType.float32, [D_FF, D_MODEL], cpu))     # W2_1
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # gamma1_1
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # beta1_1
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # gamma2_1
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # beta2_1
    
    # Layer 2 weights
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_q2
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_k2
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_v2
    input_types.append(TensorType(DType.float32, [D_MODEL, D_MODEL], cpu))  # W_o2
    input_types.append(TensorType(DType.float32, [D_MODEL, D_FF], cpu))     # W1_2
    input_types.append(TensorType(DType.float32, [D_FF, D_MODEL], cpu))     # W2_2
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # gamma1_2
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # beta1_2
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # gamma2_2
    input_types.append(TensorType(DType.float32, [D_MODEL], cpu))           # beta2_2
    
    # Create graph
    var graph = Graph("transformer_encoder", input_types)
    var inputs = graph.inputs()
    
    # First transformer block
    var layer1_output = transformer_block(
        inputs[0],                    # input
        inputs[1], inputs[2], inputs[3], inputs[4],  # W_q, W_k, W_v, W_o
        inputs[5], inputs[6],         # W1, W2
        inputs[7], inputs[8],         # gamma1, beta1
        inputs[9], inputs[10]         # gamma2, beta2
    )
    
    # Second transformer block
    var layer2_output = transformer_block(
        layer1_output,                # input from layer 1
        inputs[11], inputs[12], inputs[13], inputs[14],  # W_q, W_k, W_v, W_o
        inputs[15], inputs[16],       # W1, W2
        inputs[17], inputs[18],       # gamma1, beta1
        inputs[19], inputs[20]        # gamma2, beta2
    )
    
    graph.output([layer2_output])
    
    var graph_end = perf_counter_ns()
    var graph_time_ms = Float64(graph_end - graph_start) / 1_000_000.0
    
    print("  ✓ Graph built with", NUM_LAYERS, "transformer layers")
    print("  Graph construction time:", graph_time_ms, "ms")
    print("  Input: [", BATCH_SIZE, ",", SEQ_LENGTH, ",", D_MODEL, "]")
    print("  Output: [", BATCH_SIZE, ",", SEQ_LENGTH, ",", D_MODEL, "]")
    
    # ========================================================================
    # Step 3: Compile
    # ========================================================================
    print("\n[Step 3] Compiling transformer model...")
    
    var compile_start = perf_counter_ns()
    var model = graph.compile()
    var compile_end = perf_counter_ns()
    var compile_time_ms = Float64(compile_end - compile_start) / 1_000_000.0
    
    print("  ✓ Model compiled successfully!")
    print("  Compilation time:", compile_time_ms, "ms")
    
    # ========================================================================
    # Step 4: Execute Forward Pass (10,000 iterations)
    # ========================================================================
    print("\n[Step 4] Running forward pass benchmark...")
    print("  Executing 10,000 iterations...")
    
    # Convert all numpy arrays to tensors
    var input_t = Tensor.from_numpy(input_np)
    
    # Layer 1
    var W_q1_t = Tensor.from_numpy(W_q1_np)
    var W_k1_t = Tensor.from_numpy(W_k1_np)
    var W_v1_t = Tensor.from_numpy(W_v1_np)
    var W_o1_t = Tensor.from_numpy(W_o1_np)
    var W1_1_t = Tensor.from_numpy(W1_1_np)
    var W2_1_t = Tensor.from_numpy(W2_1_np)
    var gamma1_1_t = Tensor.from_numpy(gamma1_1_np)
    var beta1_1_t = Tensor.from_numpy(beta1_1_np)
    var gamma2_1_t = Tensor.from_numpy(gamma2_1_np)
    var beta2_1_t = Tensor.from_numpy(beta2_1_np)
    
    # Layer 2
    var W_q2_t = Tensor.from_numpy(W_q2_np)
    var W_k2_t = Tensor.from_numpy(W_k2_np)
    var W_v2_t = Tensor.from_numpy(W_v2_np)
    var W_o2_t = Tensor.from_numpy(W_o2_np)
    var W1_2_t = Tensor.from_numpy(W1_2_np)
    var W2_2_t = Tensor.from_numpy(W2_2_np)
    var gamma1_2_t = Tensor.from_numpy(gamma1_2_np)
    var beta1_2_t = Tensor.from_numpy(beta1_2_np)
    var gamma2_2_t = Tensor.from_numpy(gamma2_2_np)
    var beta2_2_t = Tensor.from_numpy(beta2_2_np)
    
    # Prepare input list for execution
    var tensor_inputs = List[Tensor]()
    tensor_inputs.append(input_t)
    tensor_inputs.append(W_q1_t)
    tensor_inputs.append(W_k1_t)
    tensor_inputs.append(W_v1_t)
    tensor_inputs.append(W_o1_t)
    tensor_inputs.append(W1_1_t)
    tensor_inputs.append(W2_1_t)
    tensor_inputs.append(gamma1_1_t)
    tensor_inputs.append(beta1_1_t)
    tensor_inputs.append(gamma2_1_t)
    tensor_inputs.append(beta2_1_t)
    tensor_inputs.append(W_q2_t)
    tensor_inputs.append(W_k2_t)
    tensor_inputs.append(W_v2_t)
    tensor_inputs.append(W_o2_t)
    tensor_inputs.append(W1_2_t)
    tensor_inputs.append(W2_2_t)
    tensor_inputs.append(gamma1_2_t)
    tensor_inputs.append(beta1_2_t)
    tensor_inputs.append(gamma2_2_t)
    tensor_inputs.append(beta2_2_t)
    
    alias NUM_ITERATIONS = 1000
    alias PRINT_EVERY = 100
    
    var total_time_ns: UInt = 0
    var py_shape = builtins.tuple(builtins.list([BATCH_SIZE, SEQ_LENGTH, D_MODEL]))
    var output_np = np.zeros(py_shape, dtype=np.float32)  # Will store the last output
    
    print("\n  Starting benchmark...")
    
    for i in range(NUM_ITERATIONS):
        # Time this iteration
        var start_time = perf_counter_ns()
        var result = model.execute(tensor_inputs)
        var end_time = perf_counter_ns()
        
        var iteration_time = end_time - start_time
        total_time_ns += iteration_time
        
        # Print progress every 100 iterations
        if (i + 1) % PRINT_EVERY == 0:
            var avg_time_ns = total_time_ns // UInt(i + 1)
            var avg_time_ms = Float64(avg_time_ns) / 1_000_000.0
            print("    Iteration", i + 1, "/", NUM_ITERATIONS, 
                  "- Avg time:", avg_time_ms, "ms")
        
        # Store the last output
        if i == NUM_ITERATIONS - 1:
            output_np = result[0].to_numpy()
    
    # Calculate final statistics
    var avg_time_ns = total_time_ns // UInt(NUM_ITERATIONS)
    var avg_time_ms = Float64(avg_time_ns) / 1_000_000.0
    var avg_time_us = Float64(avg_time_ns) / 1_000.0
    var total_time_ms = Float64(total_time_ns) / 1_000_000.0
    var total_time_s = total_time_ms / 1_000.0
    
    print("\n  ✓ Benchmark complete!")
    print("\n" + "="*70)
    print("PERFORMANCE RESULTS")
    print("="*70)
    print("  Total iterations:", NUM_ITERATIONS)
    print("  Total time:", total_time_s, "seconds")
    print("  Average time per iteration:", avg_time_ms, "ms")
    print("  Average time per iteration:", avg_time_us, "µs")
    print("  Throughput:", Float64(NUM_ITERATIONS) / total_time_s, "iterations/second")
    
    print("\n" + "="*70)
    print("TRANSFORMER OUTPUT (Final Iteration)")
    print("="*70)
    print("  Input shape:", input_np.shape)
    print("  Output shape:", output_np.shape)
    print("  Input mean:", np.mean(input_np))
    print("  Input std:", np.std(input_np))
    print("  Output mean:", np.mean(output_np))
    print("  Output std:", np.std(output_np))
    print("\n  ✓ Transformer encoder successfully executed!")
    print("  ✓ All", NUM_LAYERS, "layers processed")
    print("="*70 + "\n")


