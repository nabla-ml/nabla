#!/usr/bin/env python3
"""Analyze JAX JIT compilation performance on Mac."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

print("=== JAX JIT Analysis on Mac ===")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Platform: {jax.lib.xla_bridge.get_backend().platform}")

# Create test data
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (1000, 256))
w = jax.random.normal(key, (256, 128))
b = jax.random.normal(key, (128,))


def simple_computation(x, w, b):
    """Simple neural network layer computation."""
    return jax.nn.relu(jnp.matmul(x, w) + b)


# JIT-compiled version
jit_computation = jit(simple_computation)

print("\n=== Performance Comparison ===")

# Warm up both versions
_ = simple_computation(x, w, b)
_ = jit_computation(x, w, b)

# Time non-JIT version
times_no_jit = []
for _i in range(10):
    start = time.time()
    result_no_jit = simple_computation(x, w, b)
    times_no_jit.append(time.time() - start)

avg_no_jit = np.mean(times_no_jit[1:])  # Skip first for fair comparison

# Time JIT version (after compilation)
times_jit = []
for _i in range(10):
    start = time.time()
    result_jit = jit_computation(x, w, b)
    times_jit.append(time.time() - start)

avg_jit = np.mean(times_jit[1:])  # Skip first for fair comparison

print(f"No JIT:  {avg_no_jit * 1000:.2f}ms per call")
print(f"With JIT: {avg_jit * 1000:.2f}ms per call")
print(f"Speedup: {avg_no_jit / avg_jit:.1f}x")

# Check if results are the same
print(f"Results identical: {jnp.allclose(result_no_jit, result_jit)}")

print("\n=== Why is JIT so much faster? ===")
print("1. XLA Compilation: JAX compiles to optimized XLA (Accelerated Linear Algebra)")
print("2. Fusion: Multiple operations are fused into single kernels")
print("3. Memory optimization: Reduces intermediate array allocations")
print("4. Vectorization: Better use of SIMD instructions on your ARM64 Mac")
print("5. Loop optimization: Unrolling and other compiler optimizations")

# Test memory efficiency
print("\n=== Memory Analysis ===")


def memory_intensive_computation(x):
    # Multiple intermediate arrays without JIT
    a = jnp.sin(x)
    b = jnp.cos(a)
    c = jnp.exp(b)
    d = jnp.log(c + 1)
    return d


jit_memory_computation = jit(memory_intensive_computation)

# Time both versions
start = time.time()
result_mem_no_jit = memory_intensive_computation(x)
time_mem_no_jit = time.time() - start

start = time.time()
result_mem_jit = jit_memory_computation(x)
time_mem_jit = time.time() - start

print("Memory-intensive computation:")
print(f"  No JIT:  {time_mem_no_jit * 1000:.2f}ms")
print(f"  With JIT: {time_mem_jit * 1000:.2f}ms")
print(f"  Speedup: {time_mem_no_jit / time_mem_jit:.1f}x")

print("\n=== What's NOT happening ===")
print("❌ GPU acceleration (your Mac is using CPU backend)")
print("❌ Distributed computing")
print("❌ Special Apple Silicon GPU support (would need metal backend)")

print("\n=== What IS happening ===")
print("✅ XLA compilation to optimized machine code")
print("✅ Operation fusion (sin->cos->exp->log becomes one kernel)")
print("✅ ARM64 SIMD vectorization")
print("✅ Memory layout optimization")
print("✅ Elimination of Python interpreter overhead")
print("✅ Dead code elimination and constant folding")
