# Vectorization and JIT

This tutorial covers two essential performance optimization techniques in Nabla: vectorization with `vmap` and just-in-time compilation with `jit`.

## Vectorization with `vmap`

The `vmap` (vectorized map) transformation automatically vectorizes functions to operate over batches of data efficiently.

### Basic Vectorization

```python
import nabla as nb

def simple_function(x):
    """A function that operates on a single sample"""
    return nb.sum(x ** 2)

# Create batch data
batch_data = nb.randn((100, 10))  # 100 samples, each with 10 features

# Method 1: Manual batching (inefficient)
results_manual = []
for i in range(batch_data.shape[0]):
    result = simple_function(batch_data[i])
    results_manual.append(result)
results_manual = nb.array(results_manual)

# Method 2: Vectorized with vmap (efficient)
vectorized_fn = nb.vmap(simple_function)
results_vmap = vectorized_fn(batch_data)

print(f"Manual results shape: {results_manual.shape}")
print(f"Vmap results shape: {results_vmap.shape}")
```

### Specifying Vectorization Axes

You can control which axes to vectorize over:

```python
def matrix_vector_multiply(matrix, vector):
    return nb.matmul(matrix, vector)

# Batch of matrices and vectors
matrices = nb.randn((5, 3, 4))  # 5 matrices of shape (3, 4)
vectors = nb.randn((5, 4))      # 5 vectors of shape (4,)

# Vectorize over the first axis for both arguments
vmap_fn = nb.vmap(matrix_vector_multiply, in_axes=(0, 0))
results = vmap_fn(matrices, vectors)
print(f"Results shape: {results.shape}")  # (5, 3)

# Vectorize matrix over axis 0, broadcast vector
matrices = nb.randn((5, 3, 4))
vector = nb.randn((4,))  # Single vector
vmap_fn = nb.vmap(matrix_vector_multiply, in_axes=(0, None))
results = vmap_fn(matrices, vector)
print(f"Broadcast results shape: {results.shape}")  # (5, 3)
```

### Nested Vectorization

You can apply `vmap` multiple times for higher-dimensional batching:

```python
def pairwise_distance(x, y):
    """Compute L2 distance between two vectors"""
    return nb.sqrt(nb.sum((x - y) ** 2))

# Single distance
x = nb.array([1.0, 2.0])
y = nb.array([3.0, 4.0])
distance = pairwise_distance(x, y)

# Distance between each point in batch A and each point in batch B
batch_a = nb.randn((10, 2))  # 10 points
batch_b = nb.randn((5, 2))   # 5 points

# Create pairwise distance matrix
vmap_outer = nb.vmap(lambda x: nb.vmap(lambda y: pairwise_distance(x, y))(batch_b))
distance_matrix = vmap_outer(batch_a)
print(f"Distance matrix shape: {distance_matrix.shape}")  # (10, 5)
```

## Just-in-Time Compilation with `jit`

The `jit` transformation compiles functions for improved performance:

### Basic JIT Usage

```python
def expensive_computation(x):
    """A computationally expensive function"""
    result = x
    for _ in range(100):
        result = nb.sin(result) + nb.cos(result)
    return result

# Regular function
import time
x = nb.randn((1000,))

start_time = time.time()
result_normal = expensive_computation(x)
normal_time = time.time() - start_time

# JIT-compiled function
jit_fn = nb.jit(expensive_computation)

start_time = time.time()
result_jit = jit_fn(x)
jit_time = time.time() - start_time

print(f"Normal execution time: {normal_time:.4f}s")
print(f"JIT execution time: {jit_time:.4f}s")
print(f"Speedup: {normal_time / jit_time:.2f}x")
```

### JIT with Control Flow

JIT works with control flow, but performance is best with minimal branching:

```python
def conditional_computation(x, threshold=0.0):
    """Function with conditional logic"""
    positive_part = nb.where(x > threshold, x, 0.0)
    negative_part = nb.where(x <= threshold, -x, 0.0)
    return positive_part + negative_part

# JIT compile the conditional function
jit_conditional = nb.jit(conditional_computation)

x = nb.randn((1000,))
result = jit_conditional(x, threshold=0.5)
```

### JIT Best Practices

```python
# Good: Pure functions with minimal control flow
@nb.jit
def good_jit_function(x, y):
    return nb.matmul(x, y) + nb.sin(x)

# Avoid: Functions with side effects or complex control flow
def avoid_jit_function(x):
    # This won't JIT well due to side effects
    print(f"Processing {x.shape}")
    if x.shape[0] > 100:
        return complex_computation(x)
    else:
        return simple_computation(x)
```

## Combining `vmap` and `jit`

The most powerful pattern is combining vectorization and compilation:

```python
def neural_layer(x, weights, bias):
    """A single neural network layer"""
    return nb.relu(nb.matmul(x, weights) + bias)

def neural_network(x, all_params):
    """Full neural network"""
    output = x
    for i in range(0, len(all_params), 2):
        weights, bias = all_params[i], all_params[i + 1]
        output = neural_layer(output, weights, bias)
    return output

# Create parameters
layer_sizes = [10, 64, 32, 1]
params = []
for i in range(len(layer_sizes) - 1):
    w = nb.randn((layer_sizes[i], layer_sizes[i+1])) * 0.1
    b = nb.zeros((layer_sizes[i+1],))
    params.extend([w, b])

# Batch processing with vmap + jit
batch_network = nb.jit(nb.vmap(lambda x: neural_network(x, params)))

# Process entire batch efficiently
batch_x = nb.randn((1000, 10))  # 1000 samples
batch_output = batch_network(batch_x)
print(f"Batch output shape: {batch_output.shape}")  # (1000, 1)
```

## Performance Optimization Patterns

### 1. Reduce Array Allocations

```python
# Less efficient: Multiple temporary arrays
def inefficient_computation(x):
    temp1 = nb.sin(x)
    temp2 = nb.cos(x)
    temp3 = temp1 + temp2
    return temp3 ** 2

# More efficient: Fused computation
def efficient_computation(x):
    return (nb.sin(x) + nb.cos(x)) ** 2

# JIT will optimize both, but the second is better for memory
jit_efficient = nb.jit(efficient_computation)
```

### 2. Batch Similar Operations

```python
# Process multiple similar computations together
def batch_optimization_step(params_batch, gradients_batch, learning_rate):
    """Update multiple parameter sets at once"""
    return params_batch - learning_rate * gradients_batch

# Vectorize over different parameter initializations
def train_multiple_models(data, targets, num_models=10):
    # Initialize multiple parameter sets
    param_sets = [initialize_params() for _ in range(num_models)]
    param_batch = nb.stack(param_sets)
    
    # Training function for single model
    def single_model_loss(params):
        predictions = neural_network(data, params)
        return nb.mean((predictions - targets) ** 2)
    
    # Vectorized gradient computation
    grad_fn = nb.vmap(nb.grad(single_model_loss))
    
    # Training loop
    for epoch in range(100):
        grads = grad_fn(param_batch)
        param_batch = batch_optimization_step(param_batch, grads, 0.01)
    
    return param_batch
```

### 3. Minimize Host-Device Transfers

```python
@nb.jit
def gpu_intensive_computation(x):
    """Keep computation on device"""
    # Perform many operations without transferring back to host
    for _ in range(10):
        x = nb.matmul(x, x.T)
        x = nb.relu(x)
        x = x / nb.sum(x)  # Normalize
    return x

# Good: Single transfer back to host
result = gpu_intensive_computation(nb.randn((100, 100)))
final_result = nb.to_numpy(result)  # Only one host transfer

# Avoid: Multiple transfers
# x = nb.randn((100, 100))
# for _ in range(10):
#     x = some_operation(x)
#     print(nb.to_numpy(x).mean())  # Transfer each iteration
```

## Advanced Vectorization Patterns

### Scan Operations

```python
def scan_example():
    """Implement a scan (cumulative operation) with vmap"""
    def scan_step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    
    def manual_scan(xs):
        carry = 0.0
        results = []
        for x in xs:
            carry, result = scan_step(carry, x)
            results.append(result)
        return nb.array(results)
    
    # For now, implement with manual loop (future: native scan)
    xs = nb.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cumsum = manual_scan(xs)
    print(f"Cumulative sum: {cumsum}")
```

### Parallel Map-Reduce

```python
def parallel_reduction_example():
    """Combine vmap with reductions for parallel processing"""
    def process_chunk(chunk):
        # Process a chunk of data
        return nb.sum(chunk ** 2)
    
    # Large dataset
    data = nb.randn((10000,))
    
    # Split into chunks and process in parallel
    chunk_size = 1000
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    chunks_array = nb.stack(chunks)
    
    # Process all chunks in parallel
    chunk_results = nb.vmap(process_chunk)(chunks_array)
    final_result = nb.sum(chunk_results)
    
    print(f"Parallel result: {final_result}")
```

## Debugging JIT Functions

### Common Issues and Solutions

```python
# Issue: Shape polymorphism
def shape_dependent_function(x):
    # This might not JIT well if shapes vary
    if x.shape[0] > 100:
        return nb.mean(x)
    else:
        return nb.sum(x)

# Solution: Avoid shape-dependent control flow
def shape_agnostic_function(x):
    # Use mathematical operations instead
    weight = nb.minimum(1.0, x.shape[0] / 100.0)
    return weight * nb.mean(x) + (1 - weight) * nb.sum(x)

# Issue: Type instability
def type_unstable_function(x, flag):
    if flag:
        return x.astype(nb.float32)  # Returns float32
    else:
        return x.astype(nb.int32)    # Returns int32

# Solution: Ensure consistent return types
def type_stable_function(x, flag):
    result = nb.where(flag, x, x.astype(nb.int32).astype(nb.float32))
    return result.astype(nb.float32)  # Always returns float32
```

## Performance Profiling

```python
def profile_performance():
    """Compare different optimization strategies"""
    import time
    
    def baseline_function(x):
        return nb.sum(x ** 2)
    
    # Test data
    batch_data = nb.randn((1000, 100))
    
    # Baseline: No optimization
    start = time.time()
    results = [baseline_function(x) for x in batch_data]
    baseline_time = time.time() - start
    
    # Vmap only
    vmap_fn = nb.vmap(baseline_function)
    start = time.time()
    results_vmap = vmap_fn(batch_data)
    vmap_time = time.time() - start
    
    # JIT only
    jit_fn = nb.jit(baseline_function)
    start = time.time()
    results_jit = [jit_fn(x) for x in batch_data]
    jit_time = time.time() - start
    
    # Both vmap and JIT
    vmap_jit_fn = nb.jit(nb.vmap(baseline_function))
    start = time.time()
    results_both = vmap_jit_fn(batch_data)
    both_time = time.time() - start
    
    print(f"Baseline time: {baseline_time:.4f}s")
    print(f"Vmap time: {vmap_time:.4f}s (speedup: {baseline_time/vmap_time:.2f}x)")
    print(f"JIT time: {jit_time:.4f}s (speedup: {baseline_time/jit_time:.2f}x)")
    print(f"Both time: {both_time:.4f}s (speedup: {baseline_time/both_time:.2f}x)")

profile_performance()
```

## Next Steps

- Check out the {doc}`../examples/index` for real-world performance examples
- Combine with {doc}`automatic_differentiation` for efficient gradient computation
