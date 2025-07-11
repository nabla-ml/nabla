import json
import os
import time
import tracemalloc

import numpy as np
import psutil

# Framework imports
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp

import nabla as nb


class MemoryProfiler:
    """Track memory usage during benchmark execution"""

    def __init__(self):
        self.peak_memory = 0
        self.allocations = []

    def start_tracking(self):
        tracemalloc.start()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def stop_tracking(self):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory = peak / 1024 / 1024  # MB
        return self.peak_memory


class CompilationProfiler:
    """Track JIT compilation times"""

    @staticmethod
    def time_compilation(func, args):
        """Time the compilation phase separately from execution"""
        start_time = time.perf_counter()
        compiled_func = jax.jit(func) if "jax" in str(func) else nb.jit(func)
        compiled_func(*args)  # First call triggers compilation
        compilation_time = time.perf_counter() - start_time

        # Time pure execution
        start_exec = time.perf_counter()
        result = compiled_func(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        exec_time = time.perf_counter() - start_exec

        return compilation_time, exec_time


def scaling_benchmark_attention(
    framework: str, seq_lengths: list[int], embed_dim: int = 512
):
    """Benchmark attention scaling with sequence length"""
    results = {}

    def attention_nb(x, w_q, w_k, w_v):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        scores = q @ nb.permute(k, (0, 2, 1)) / np.sqrt(embed_dim)
        # Use simple scores instead of softmax to avoid numerical issues in scaling
        return nb.sum(scores @ v, axes=None)

    def attention_jax(x, w_q, w_k, w_v):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        scores = q @ jnp.transpose(k, (0, 2, 1)) / jnp.sqrt(embed_dim)
        return jnp.sum(scores @ v)

    func = attention_nb if framework == "nabla" else attention_jax
    array_lib = nb if framework == "nabla" else jnp

    for seq_len in seq_lengths:
        print(f"Testing {framework} attention scaling: seq_len={seq_len}")

        # Generate inputs
        x = array_lib.array(np.random.rand(4, seq_len, embed_dim).astype(np.float32))
        w_q = array_lib.array(np.random.rand(embed_dim, embed_dim).astype(np.float32))
        w_k = array_lib.array(np.random.rand(embed_dim, embed_dim).astype(np.float32))
        w_v = array_lib.array(np.random.rand(embed_dim, embed_dim).astype(np.float32))

        # Memory profiling
        profiler = MemoryProfiler()
        profiler.start_tracking()

        # Compilation + execution timing
        comp_time, exec_time = CompilationProfiler.time_compilation(
            func, (x, w_q, w_k, w_v)
        )

        peak_mem = profiler.stop_tracking()

        results[seq_len] = {
            "compilation_time": comp_time * 1000,  # ms
            "execution_time": exec_time * 1000,  # ms
            "peak_memory": peak_mem,  # MB
            "total_time": (comp_time + exec_time) * 1000,
        }

    return results


def depth_scaling_benchmark(framework: str, depths: list[int], layer_dim: int = 256):
    """Benchmark neural network depth scaling"""
    results = {}

    def deep_network_nb(x, weights):
        for w in weights:
            x = nb.tanh(x @ w)
        return nb.sum(x, axes=None)

    def deep_network_jax(x, weights):
        for w in weights:
            x = jnp.tanh(x @ w)
        return jnp.sum(x)

    func = deep_network_nb if framework == "nabla" else deep_network_jax
    array_lib = nb if framework == "nabla" else jnp

    for depth in depths:
        print(f"Testing {framework} depth scaling: depth={depth}")

        # Generate inputs
        x = array_lib.array(np.random.rand(32, layer_dim).astype(np.float32))
        weights = [
            array_lib.array(np.random.rand(layer_dim, layer_dim).astype(np.float32))
            for _ in range(depth)
        ]

        # Gradient computation
        grad_func = (
            nb.jacrev(func, argnums=0)
            if framework == "nabla"
            else jax.jacrev(func, argnums=0)
        )

        # Memory profiling
        profiler = MemoryProfiler()
        profiler.start_tracking()

        # Compilation + execution timing
        start_time = time.perf_counter()
        compiled_grad = (
            nb.jit(grad_func) if framework == "nabla" else jax.jit(grad_func)
        )
        result = compiled_grad(x, weights)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        total_time = time.perf_counter() - start_time

        peak_mem = profiler.stop_tracking()

        results[depth] = {
            "total_time": total_time * 1000,  # ms
            "peak_memory": peak_mem,  # MB
        }

    return results


def memory_efficiency_comparison():
    """Compare memory efficiency patterns between frameworks"""
    print("=== MEMORY EFFICIENCY ANALYSIS ===")

    sizes = [64, 128, 256, 512, 1024]
    nabla_memory = {}
    jax_memory = {}

    # Test matrix multiplication memory scaling
    for size in sizes:
        print(f"Testing memory usage for {size}x{size} matrices...")

        # Nabla memory test
        profiler = MemoryProfiler()
        profiler.start_tracking()

        a_nb = nb.array(np.random.rand(size, size).astype(np.float32))
        b_nb = nb.array(np.random.rand(size, size).astype(np.float32))

        @nb.jit
        def matmul_grad_nb(a, b):
            return nb.sum(nb.jacrev(lambda x: nb.sum(x @ b), argnums=0)(a), axes=None)

        result_nb = matmul_grad_nb(a_nb, b_nb)
        nabla_memory[size] = profiler.stop_tracking()

        # JAX memory test
        profiler.start_tracking()

        a_jax = jnp.array(np.random.rand(size, size).astype(np.float32))
        b_jax = jnp.array(np.random.rand(size, size).astype(np.float32))

        @jax.jit
        def matmul_grad_jax(a, b):
            return jnp.sum(jax.jacrev(lambda x: jnp.sum(x @ b), argnums=0)(a))

        result_jax = matmul_grad_jax(a_jax, b_jax)
        result_jax.block_until_ready()
        jax_memory[size] = profiler.stop_tracking()

    return nabla_memory, jax_memory


def compilation_time_analysis():
    """Analyze JIT compilation times for different function complexities"""
    print("=== COMPILATION TIME ANALYSIS ===")

    complexities = ["simple", "medium", "complex", "very_complex"]
    results = {"nabla": {}, "jax": {}}

    def simple_func_nb(x):
        return nb.sum(x**2)

    def simple_func_jax(x):
        return jnp.sum(x**2)

    def medium_func_nb(x, w):
        return nb.sum(nb.tanh(x @ w))

    def medium_func_jax(x, w):
        return jnp.sum(jnp.tanh(x @ w))

    def complex_func_nb(x, w1, w2):
        h = nb.relu(x @ w1)
        return nb.sum(nb.sigmoid(h @ w2))

    def complex_func_jax(x, w1, w2):
        h = jax.nn.relu(x @ w1)
        return jnp.sum(jax.nn.sigmoid(h @ w2))

    def very_complex_func_nb(x, w1, w2, w3):
        h1 = nb.relu(x @ w1)
        h2 = nb.tanh(h1 @ w2)
        attn_scores = h2 @ nb.permute(h2, (1, 0))
        attn_out = attn_scores @ h2
        return nb.sum(attn_out @ w3)

    def very_complex_func_jax(x, w1, w2, w3):
        h1 = jax.nn.relu(x @ w1)
        h2 = jnp.tanh(h1 @ w2)
        attn_scores = h2 @ jnp.transpose(h2, (1, 0))
        attn_out = attn_scores @ h2
        return jnp.sum(attn_out @ w3)

    # Test data
    x = np.random.rand(128, 256).astype(np.float32)
    w1 = np.random.rand(256, 128).astype(np.float32)
    w2 = np.random.rand(128, 64).astype(np.float32)
    w3 = np.random.rand(64, 32).astype(np.float32)

    # Test cases
    test_cases = {
        "simple": (simple_func_nb, simple_func_jax, (x,)),
        "medium": (medium_func_nb, medium_func_jax, (x, w1)),
        "complex": (complex_func_nb, complex_func_jax, (x, w1, w2)),
        "very_complex": (very_complex_func_nb, very_complex_func_jax, (x, w1, w2, w3)),
    }

    for complexity, (nb_func, jax_func, args) in test_cases.items():
        print(f"Testing compilation time for {complexity} function...")

        # Convert args to appropriate framework arrays
        nb_args = [nb.array(arg) for arg in args]
        jax_args = [jnp.array(arg) for arg in args]

        # Nabla compilation time
        comp_time_nb, exec_time_nb = CompilationProfiler.time_compilation(
            nb_func, nb_args
        )

        # JAX compilation time
        comp_time_jax, exec_time_jax = CompilationProfiler.time_compilation(
            jax_func, jax_args
        )

        results["nabla"][complexity] = {
            "compilation_time": comp_time_nb * 1000,
            "execution_time": exec_time_nb * 1000,
        }
        results["jax"][complexity] = {
            "compilation_time": comp_time_jax * 1000,
            "execution_time": exec_time_jax * 1000,
        }

    return results


def run_benchmarks():
    """Run comprehensive benchmark tests"""
    print("=" * 60)
    print("THESIS-QUALITY BENCHMARK SUITE")
    print("Nabla vs JAX: Advanced Analysis")
    print("=" * 60)

    # 1. Attention Scaling Analysis
    print("\n1. ATTENTION SCALING ANALYSIS")
    seq_lengths = [64, 128, 256, 512, 1024]
    nabla_attention = scaling_benchmark_attention("nabla", seq_lengths)
    jax_attention = scaling_benchmark_attention("jax", seq_lengths)

    # 2. Depth Scaling Analysis
    print("\n2. NEURAL NETWORK DEPTH SCALING")
    depths = [2, 4, 8, 16, 32]
    nabla_depth = depth_scaling_benchmark("nabla", depths)
    jax_depth = depth_scaling_benchmark("jax", depths)

    # 3. Memory Efficiency Analysis
    print("\n3. MEMORY EFFICIENCY COMPARISON")
    nabla_mem, jax_mem = memory_efficiency_comparison()

    # 4. Compilation Time Analysis
    print("\n4. COMPILATION TIME ANALYSIS (MAX vs XLA)")
    compilation_results = compilation_time_analysis()

    # Save results for analysis
    results = {
        "attention_scaling": {"nabla": nabla_attention, "jax": jax_attention},
        "depth_scaling": {"nabla": nabla_depth, "jax": jax_depth},
        "memory_efficiency": {"nabla": nabla_mem, "jax": jax_mem},
        "compilation_times": compilation_results,
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary analysis
    print("\n" + "=" * 60)
    print("THESIS BENCHMARK SUMMARY")
    print("=" * 60)

    print("\n📈 ATTENTION SCALING INSIGHTS:")
    print("At seq_len=1024:")
    print(
        f"  Nabla: {nabla_attention[1024]['total_time']:.2f}ms, {nabla_attention[1024]['peak_memory']:.1f}MB"
    )
    print(
        f"  JAX:   {jax_attention[1024]['total_time']:.2f}ms, {jax_attention[1024]['peak_memory']:.1f}MB"
    )

    print("\n🧠 MEMORY EFFICIENCY:")
    print("At 1024x1024 matrices:")
    print(f"  Nabla: {nabla_mem[1024]:.1f}MB peak memory")
    print(f"  JAX:   {jax_mem[1024]:.1f}MB peak memory")

    print("\n⚡ COMPILATION ANALYSIS (MAX vs XLA):")
    for complexity in ["simple", "complex"]:
        nb_comp = compilation_results["nabla"][complexity]["compilation_time"]
        jax_comp = compilation_results["jax"][complexity]["compilation_time"]
        print(f"  {complexity.title()} functions:")
        print(f"    Nabla (MAX): {nb_comp:.2f}ms compilation")
        print(f"    JAX (XLA):   {jax_comp:.2f}ms compilation")

    print("\n📊 Results saved to: benchmark_results.json")
    print("Ready for thesis analysis and publication!")


if __name__ == "__main__":
    run_benchmarks()
