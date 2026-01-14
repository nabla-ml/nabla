"""
Stress Test: Large Scale MLP with vmap + compile

Tests the robustness of the system with deep networks, large tensors, 
and varying large batch sizes.
"""

import time
from nabla import Tensor
from nabla.transforms.vmap import vmap
from nabla.transforms.compile import compile
from nabla.core.compute_graph import GRAPH
from nabla import relu

def reset_graph():
    """Reset global graph state."""
    GRAPH._reset(GRAPH.context, 0)

def stress_test_large_mlp():
    print("\n" + "=" * 60)
    print(" 🏋️‍♂️ NABLA: Large Scale Stress Test")
    print(" Deep MLP + Large Tensors + Dynamic Batching")
    print("=" * 60 + "\n")

    reset_graph()

    # Configuration
    INPUT_DIM = 1024
    HIDDEN_DIM = 1024
    OUTPUT_DIM = 10
    NUM_LAYERS = 20  # Deep network
    
    print(f"🔧 Configuration:")
    print(f"   Input Dim: {INPUT_DIM}")
    print(f"   Hidden Dim: {HIDDEN_DIM}")
    print(f"   Output Dim: {OUTPUT_DIM}")
    print(f"   Layers: {NUM_LAYERS}")
    print(f"   Total Parameters: ~{(INPUT_DIM * HIDDEN_DIM) + (NUM_LAYERS - 1) * (HIDDEN_DIM * HIDDEN_DIM) + (HIDDEN_DIM * OUTPUT_DIM):,}")
    print()

    # Initialize weights
    print("🏗️ Initializing weights...")
    weights = []
    
    # Input layer
    W_in = Tensor.uniform((INPUT_DIM, HIDDEN_DIM))
    weights.append(W_in)
    
    # Hidden layers
    for _ in range(NUM_LAYERS - 1):
        W_h = Tensor.uniform((HIDDEN_DIM, HIDDEN_DIM))
        weights.append(W_h)
        
    # Output layer
    W_out = Tensor.uniform((HIDDEN_DIM, OUTPUT_DIM))
    weights.append(W_out)
    
    print("✅ Weights initialized.")

    # Define model with vmap + compile
    @compile(dynamic_dims={0: {0: "batch"}})
    @vmap(in_axes=(0, None)) # x is batched (0), weights are list (None - treated as pytree leaf if passed as container? No, we pass weights as list, so we need to handle that.
    # vmap treats list arguments as pytrees. If we pass 'weights' as a single argument (list of tensors), and we want it to be unbatched, we should specify in_axes for it.
    # If weights is the second argument: in_axes=(0, None) means the whole list is unbatched.
    def large_mlp(x, weights_list):
        h = x
        
        # Input -> Hidden 1
        h = h @ weights_list[0]
        h = relu(h)
        
        # Hidden -> Hidden
        for i in range(1, len(weights_list) - 1):
            h = h @ weights_list[i]
            h = relu(h)
            
        # Hidden -> Output
        out = h @ weights_list[-1]
        return out

    # Batch sizes to test
    batch_sizes = [1, 32, 128, 512, 1024, 2048]
    
    print("\n🧪 Starting execution stress test...")
    print("-" * 80)
    print(f"{'Batch':>8} | {'Status':>12} | {'Time (s)':>10} | {'Output Shape':>20} | {'Stats'}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        # Generate input
        x = Tensor.uniform((batch_size, INPUT_DIM))
        
        try:
            # Run model
            output = large_mlp(x, weights)
            
            elapsed = time.time() - start_time
            stats = large_mlp.stats
            status = "COMPILED" if stats.misses > 0 and batch_size == batch_sizes[0] else ("CACHE HIT" if stats.hits > 0 else "UNKNOWN")
            
            # Check compilation behavior
            if batch_size == batch_sizes[0]:
                if stats.misses != 1:
                     print(f"WARNING: Expected 1 miss for first run, got {stats.misses}")
            else:
                 if stats.misses != 1:
                     # We expect exactly 1 miss total (from the first run), so subsequent runs should keep it at 1.
                     # If misses increases, we are recompiling.
                     pass
            
            print(f"{batch_size:>8} | {status:>12} | {elapsed:>10.4f} | {str(tuple(output.shape)):>20} | misses={stats.misses}, hits={stats.hits}")
            
        except Exception as e:
            print(f"{batch_size:>8} | {'FAILED':>12} | {time.time()-start_time:>10.4f} | {'ERROR':>20} | {e}")
            raise e

    print("-" * 80)
    final_stats = large_mlp.stats
    print(f"\n📈 Final Stats: Hit Rate: {final_stats.hit_rate:.1f}% ({final_stats.hits} hits, {final_stats.misses} miss)")
    
    if final_stats.misses > 1:
         print("❌ FAILED: Multiple compilations detected for dynamic batch size!")
         exit(1)
         
    print("\n✅ Stress Test Completed Successfully!")

if __name__ == "__main__":
    stress_test_large_mlp()
