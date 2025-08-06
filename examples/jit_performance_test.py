# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

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
