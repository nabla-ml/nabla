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

"""
GPU vs CPU Performance Comparison for Nabla MLP Training

This script demonstrates when GPU acceleration provides significant benefits
vs when CPU and GPU performance are similar.
"""

import time
import numpy as np
from max import driver
import nabla as nb

# Configuration for GPU vs CPU comparison
# Small config (similar performance expected)
SMALL_CONFIG = {
    "BATCH_SIZE": 8,
    "LAYERS": [1, 256, 1024, 1024, 16, 1024, 256, 1],
    "LEARNING_RATE": 0.001,
    "NUM_EPOCHS": 20,
    "PRINT_INTERVAL": 10,
    "SIN_PERIODS": 8
}

# Large config (GPU should be much faster)
LARGE_CONFIG = {
    "BATCH_SIZE": 1024,  # Much larger batch size
    "LAYERS": [1, 2048, 4096, 4096, 2048, 2048, 4096, 2048, 1],  # Much larger model
    "LEARNING_RATE": 0.001,
    "NUM_EPOCHS": 20,
    "PRINT_INTERVAL": 5,
    "SIN_PERIODS": 8
}

def calculate_model_size(layers):
    """Calculate total number of parameters in the model."""
    total_params = 0
    for i in range(len(layers) - 1):
        # Weights: layers[i] * layers[i+1]
        # Biases: layers[i+1]
        total_params += layers[i] * layers[i + 1] + layers[i + 1]
    return total_params

def mlp_forward(x: nb.Array, params: list[nb.Array]) -> nb.Array:
    """MLP forward pass through all layers."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nb.matmul(output, w) + b
        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = nb.relu(output)
    return output

def mean_squared_error(predictions: nb.Array, targets: nb.Array, device) -> nb.Array:
    """Compute mean squared error loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = nb.array(predictions.shape[0], dtype=nb.DType.float32).to(device)
    loss = nb.sum(squared_errors) / batch_size
    return loss

def create_sin_dataset(batch_size: int, sin_periods: int, device) -> tuple[nb.Array, nb.Array]:
    """Create the sin dataset."""
    x = nb.rand((batch_size, 1), lower=0.0, upper=1.0, dtype=nb.DType.float32).to(device)
    targets = nb.sin(sin_periods * 2.0 * np.pi * x) / 2.0 + 0.5
    return x, targets

def initialize_params(layers: list[int], device, seed: int = 42) -> list[nb.Array]:
    """Initialize model parameters."""
    np.random.seed(seed)
    params = []
    for i in range(len(layers) - 1):
        fan_in, fan_out = layers[i], layers[i + 1]
        w = nb.he_normal((fan_in, fan_out), seed=seed).to(device)
        b = nb.zeros((fan_out,)).to(device)
        params.append(w)
        params.append(b)
    return params

@nb.jit
def train_step_simple(x: nb.Array, targets: nb.Array, params: list[nb.Array], learning_rate: float) -> tuple[list[nb.Array], nb.Array]:
    """Simple training step using SGD."""
    def loss_fn(*inner_params):
        param_list = list(inner_params)
        predictions = mlp_forward(x, param_list)
        diff = predictions - targets
        squared_errors = diff * diff
        batch_size = nb.array(predictions.shape[0], dtype=nb.DType.float32)
        loss = nb.sum(squared_errors) / batch_size
        return loss

    loss_value, param_gradients = nb.value_and_grad(
        loss_fn, argnums=list(range(len(params)))
    )(*params)

    # Simple SGD update
    updated_params = []
    for param, grad in zip(params, param_gradients):
        updated_params.append(param - learning_rate * grad)

    return updated_params, loss_value

def run_benchmark(config_name: str, config: dict, device) -> tuple[float, float]:
    """Run training benchmark and return timing results."""
    print(f"\n{'='*60}")
    print(f"RUNNING {config_name} on {device}")
    print(f"{'='*60}")
    
    # Extract config
    BATCH_SIZE = config["BATCH_SIZE"]
    LAYERS = config["LAYERS"] 
    LEARNING_RATE = config["LEARNING_RATE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    PRINT_INTERVAL = config["PRINT_INTERVAL"]
    SIN_PERIODS = config["SIN_PERIODS"]
    
    model_size = calculate_model_size(LAYERS)
    print(f"Model size: {model_size:,} parameters")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Architecture: {LAYERS}")
    
    # Initialize
    params = initialize_params(LAYERS, device)
    
    # Warm-up run (important for GPU timing)
    print("Warming up...")
    for _ in range(3):
        x, targets = create_sin_dataset(BATCH_SIZE, SIN_PERIODS, device)
        params, _ = train_step_simple(x, targets, params, LEARNING_RATE, device)
    
    print("Starting benchmark...")
    total_time = 0.0
    losses = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Create batch
        x, targets = create_sin_dataset(BATCH_SIZE, SIN_PERIODS, device)
        
        # Training step
        params, loss_value = train_step_simple(x, targets, params, LEARNING_RATE, device)
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        losses.append(loss_value.to_numpy().item())
        
        if epoch % PRINT_INTERVAL == 0:
            avg_loss = np.mean(losses[-PRINT_INTERVAL:])
            avg_time = total_time / epoch
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Avg time/epoch: {avg_time:.4f}s")
    
    avg_time_per_epoch = total_time / NUM_EPOCHS
    final_loss = losses[-1]
    
    print(f"\nCompleted {config_name}:")
    print(f"  Average time per epoch: {avg_time_per_epoch:.4f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Final loss: {final_loss:.6f}")
    
    return avg_time_per_epoch, final_loss

def main():
    """Main comparison function."""
    print("GPU vs CPU Performance Comparison for Nabla MLP Training")
    print("=" * 60)
    
    # Check available devices
    has_gpu = nb.accelerator_count() > 0
    print(f"GPU available: {has_gpu}")
    
    if not has_gpu:
        print("No GPU available. Running CPU-only comparison with different model sizes.")
        
        # CPU comparison with different sizes
        cpu_device = nb.cpu()
        
        print("\nTesting SMALL model on CPU...")
        small_time, small_loss = run_benchmark("SMALL CONFIG", SMALL_CONFIG, cpu_device)
        
        print("\nTesting LARGE model on CPU...")
        large_time, large_loss = run_benchmark("LARGE CONFIG", LARGE_CONFIG, cpu_device)
        
        print(f"\n{'='*60}")
        print("CPU COMPARISON RESULTS:")
        print(f"{'='*60}")
        print(f"Small model: {small_time:.4f}s per epoch")
        print(f"Large model: {large_time:.4f}s per epoch")
        print(f"Large/Small ratio: {large_time/small_time:.2f}x")
        
    else:
        cpu_device = nb.cpu()
        gpu_device = nb.accelerator()
        
        results = {}
        
        # Test small model on both devices
        print("\n" + "="*60)
        print("SMALL MODEL COMPARISON")
        print("="*60)
        
        cpu_small_time, cpu_small_loss = run_benchmark("SMALL - CPU", SMALL_CONFIG, cpu_device)
        gpu_small_time, gpu_small_loss = run_benchmark("SMALL - GPU", SMALL_CONFIG, gpu_device)
        
        # Test large model on both devices  
        print("\n" + "="*60)
        print("LARGE MODEL COMPARISON")
        print("="*60)
        
        cpu_large_time, cpu_large_loss = run_benchmark("LARGE - CPU", LARGE_CONFIG, cpu_device)
        gpu_large_time, gpu_large_loss = run_benchmark("LARGE - GPU", LARGE_CONFIG, gpu_device)
        
        # Results summary
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON RESULTS:")
        print(f"{'='*60}")
        print(f"SMALL MODEL:")
        print(f"  CPU: {cpu_small_time:.4f}s per epoch")
        print(f"  GPU: {gpu_small_time:.4f}s per epoch")
        print(f"  Speedup: {cpu_small_time/gpu_small_time:.2f}x")
        print(f"")
        print(f"LARGE MODEL:")
        print(f"  CPU: {cpu_large_time:.4f}s per epoch") 
        print(f"  GPU: {gpu_large_time:.4f}s per epoch")
        print(f"  Speedup: {cpu_large_time/gpu_large_time:.2f}x")
        print(f"")
        print(f"ANALYSIS:")
        small_speedup = cpu_small_time/gpu_small_time
        large_speedup = cpu_large_time/gpu_large_time
        
        if small_speedup < 1.5:
            print(f"  ✓ Small model shows minimal GPU advantage ({small_speedup:.1f}x) - as expected!")
        else:
            print(f"  ⚠ Small model shows unexpected GPU advantage ({small_speedup:.1f}x)")
            
        if large_speedup > 3.0:
            print(f"  ✓ Large model shows significant GPU advantage ({large_speedup:.1f}x) - excellent!")
        elif large_speedup > 1.5:
            print(f"  ✓ Large model shows moderate GPU advantage ({large_speedup:.1f}x) - good!")
        else:
            print(f"  ⚠ Large model shows minimal GPU advantage ({large_speedup:.1f}x) - unexpected")

if __name__ == "__main__":
    main()
