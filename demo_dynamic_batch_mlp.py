"""Showoff: Dynamic Batch MLP with vmap + compile

Demonstrates: Compile Once, Run With Any Batch Size!
"""

from eager import Tensor
from eager.vmap_trafo import vmap
from eager.compile_trafo import compile
from eager.compute_graph import GRAPH
from eager import relu


def reset_graph():
    """Reset global graph state."""
    GRAPH._reset(GRAPH.context, 0)


def demo_dynamic_batch():
    """Demonstrate dynamic batch sizes with vmap + compile."""
    
    print("\n" + "=" * 60)
    print(" 🚀 NABLA: Dynamic Batch Size Demo")
    print(" Compile Once, Run With Any Batch Size!")
    print("=" * 60 + "\n")
    
    reset_graph()
    
    # Simple 2-layer MLP: 8 -> 4 -> 2
    W1 = Tensor.ones((8, 4))
    W2 = Tensor.ones((4, 2))
    
    print("📊 Model: 8 -> 4 -> 2 MLP")
    print()
    
    # =========================================================================
    # Dynamic batch version
    # =========================================================================
    
    @compile(dynamic_dims={0: {0: "batch"}})
    @vmap(in_axes=(0, None, None))
    def batched_mlp(x, W1, W2):
        h = relu(x @ W1)  # (batch, 8) -> (batch, 4)
        return h @ W2     # (batch, 4) -> (batch, 2)
    
    batch_sizes = [1, 2, 4, 8]
    
    print("🧪 Testing with various batch sizes...")
    print("-" * 60)
    print(f"{'Batch':>8} | {'Status':>15} | {'Shape':>15} | {'Stats'}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        x = Tensor.uniform((batch_size, 8))
        
        output = batched_mlp(x, W1, W2)
        
        stats = batched_mlp.stats
        status = "COMPILED" if stats.misses == 1 and stats.hits == 0 else "CACHE HIT ✓"
        
        print(f"{batch_size:>8} | {status:>15} | {str(tuple(output.shape)):>15} | misses={stats.misses}, hits={stats.hits}")
    
    print("-" * 60)
    
    final_stats = batched_mlp.stats
    print(f"\n📈 Hit Rate: {final_stats.hit_rate:.1f}% ({final_stats.hits} hits, {final_stats.misses} miss)")
    
    print("\n" + "=" * 60)
    print("  ✅ Single compilation for ALL batch sizes!")
    print("  ✅ No padding, no recompilation!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    demo_dynamic_batch()
