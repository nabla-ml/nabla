#!/usr/bin/env python3
"""Performance test of JIT optimization without debug output."""

import time

import nabla as nb


# Simple test function
def simple_func(inputs):
    x, y = inputs
    return x * y + x


# Create JIT version without debug
jitted_func = nb.jit(simple_func)

# Test data
x = nb.array([1.0, 2.0, 3.0])
y = nb.array([4.0, 5.0, 6.0])
inputs = [x, y]

# Warmup run (compilation)
print("Warmup run...")
result = jitted_func(inputs)
print(f"Result: {result}")

# Time multiple runs
print("\nTiming 1000 fast runs...")
start_time = time.perf_counter()
for _ in range(1000):
    result = jitted_func(inputs)
end_time = time.perf_counter()

total_time = end_time - start_time
avg_time = total_time / 1000
print(f"Total time: {total_time:.6f}s")
print(f"Average per call: {avg_time:.6f}s")
print(f"Calls per second: {1 / avg_time:.0f}")
