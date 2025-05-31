"""Integration test for MLP training with benchmarking."""

import gc
import os
import time
import tracemalloc

import numpy as np
import psutil

import nabla as nb

# Configuration constants
DEFAULT_BATCH_SIZE = 256
DEFAULT_LAYERS = [1, 64, 512, 512, 512, 64, 1]
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_NUM_EPOCHS = 200  # Shorter for detailed profiling
PRINT_INTERVAL = 10  # More frequent printing
SIN_PERIODS = 8


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


def mean_squared_error(predictions: nb.Array, targets: nb.Array) -> nb.Array:
    """Compute mean squared error loss."""
    diff = predictions - targets
    squared_errors = diff * diff
    batch_size = nb.array([np.float32(predictions.shape[0])])
    loss = nb.reduce_sum(squared_errors) / batch_size
    return loss


def mlp_forward_and_loss(inputs: list[nb.Array]) -> list[nb.Array]:
    """Combined forward pass and loss computation for VJP."""
    x, targets, *params = inputs
    predictions = mlp_forward(x, params)
    loss = mean_squared_error(predictions, targets)
    return [loss]


def create_sin_dataset(batch_size: int = 32) -> tuple[nb.Array, nb.Array]:
    """Create training data for learning sin function."""
    np_x = np.random.uniform(0.0, 1.0, (batch_size, 1)).astype(np.float32)
    np_targets = (np.sin(SIN_PERIODS * 2.0 * np.pi * np_x) / 2.0 + 0.5).astype(
        np.float32
    )

    x = nb.Array.from_numpy(np_x)
    targets = nb.Array.from_numpy(np_targets)

    return x, targets


def initialize_mlp_params(layers: list[int], seed: int = 42) -> list[nb.Array]:
    """Initialize MLP parameters with Xavier initialization."""
    np.random.seed(seed)
    params = []

    for i in range(len(layers) - 1):
        # Xavier initialization (matching Mojo's He initialization style)
        fan_in, fan_out = layers[i], layers[i + 1]
        std = np.sqrt(2.0 / fan_in)  # He initialization like Mojo

        w_np = np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)
        b_np = np.zeros((1, fan_out), dtype=np.float32)

        w = nb.Array.from_numpy(w_np)
        b = nb.Array.from_numpy(b_np)
        params.extend([w, b])

    return params


def sgd_momentum_step(
    params: list[nb.Array],
    gradients: list[nb.Array],
    velocities: list[nb.Array],
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> tuple[list[nb.Array], list[nb.Array]]:
    """Pure functional SGD with momentum step - no hidden state."""
    updated_params = []
    updated_velocities = []

    momentum_tensor = nb.array([np.float32(momentum)])
    lr_tensor = nb.array([np.float32(learning_rate)])

    for param, grad, velocity in zip(params, gradients, velocities, strict=False):
        # Compute new velocity: v = momentum * v - lr * grad
        new_velocity = momentum_tensor * velocity - lr_tensor * grad
        # Compute new parameter: param = param + velocity
        new_param = param + new_velocity

        updated_params.append(new_param)
        updated_velocities.append(new_velocity)

    return updated_params, updated_velocities


def init_sgd_momentum_state(params: list[nb.Array]) -> list[nb.Array]:
    """Initialize optimizer state (velocities) for SGD with momentum."""
    velocities = []
    for param in params:
        v_np = np.zeros_like(param.to_numpy())
        velocities.append(nb.Array.from_numpy(v_np))
    return velocities


def train_step_functional(
    x: nb.Array,
    targets: nb.Array,
    params: list[nb.Array],
    optimizer_state: list[nb.Array],
    learning_rate: float = DEFAULT_LEARNING_RATE,
    momentum: float = DEFAULT_MOMENTUM,
) -> tuple[list[nb.Array], list[nb.Array], float]:
    """Perform one training step using functional SGD with momentum."""
    # Forward pass + VJP for gradients
    all_inputs = [x, targets] + params
    loss_values, vjp_fn = nb.vjp(mlp_forward_and_loss, all_inputs)

    # Backward pass
    cotangent = [nb.array([np.float32(1.0)])]
    gradients = vjp_fn(cotangent)

    # Extract parameter gradients (skip x and targets)
    param_gradients = gradients[2:]

    # Functional optimizer update
    updated_params, updated_state = sgd_momentum_step(
        params, param_gradients, optimizer_state, learning_rate, momentum
    )

    loss_scalar = loss_values[0].to_numpy().item()
    return updated_params, updated_state, loss_scalar


def test_mlp_training_with_benchmark():
    """Test MLP training with functional optimizer and benchmarking."""
    print(
        "=== Testing MLP Training with Functional SGD Momentum: Learning Sin Function ==="
    )
    print(f"Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"Network architecture: {DEFAULT_LAYERS}")
    print(f"Learning rate: {DEFAULT_LEARNING_RATE}")
    print(f"Momentum: {DEFAULT_MOMENTUM}")
    print(f"Sin periods: {SIN_PERIODS}")
    print("Starting training...")

    # Start memory tracking
    tracemalloc.start()

    # Initialize model parameters
    params = initialize_mlp_params(DEFAULT_LAYERS)

    # Initialize functional optimizer state
    optimizer_state = init_sgd_momentum_state(params)

    # Tracking variables
    avg_loss = 0.0
    avg_time = 0.0
    initial_memory = None

    # Training loop with benchmarking
    for epoch in range(1, DEFAULT_NUM_EPOCHS + 1):
        start_time = time.perf_counter()

        # Create fresh batch each iteration (like Mojo)
        x, targets = create_sin_dataset(DEFAULT_BATCH_SIZE)

        # Functional training step
        params, optimizer_state, loss = train_step_functional(
            x, targets, params, optimizer_state, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM
        )

        end_time = time.perf_counter()

        # Accumulate metrics
        avg_loss += loss
        avg_time += end_time - start_time

        # Memory tracking
        if epoch == 1:
            initial_memory = tracemalloc.get_traced_memory()[0]
            process = psutil.Process(os.getpid())
            initial_rss = process.memory_info().rss

        # Force garbage collection periodically to see if it helps
        if epoch % 100 == 0:
            gc.collect()

        # Print progress every PRINT_INTERVAL epochs
        if epoch % PRINT_INTERVAL == 0:
            gc.collect()  # Force garbage collection

            # Detailed memory tracking
            current_traced, peak_traced = tracemalloc.get_traced_memory()
            memory_growth = (current_traced - initial_memory) / 1024 / 1024  # MB

            process = psutil.Process(os.getpid())
            current_rss = process.memory_info().rss
            rss_growth = (current_rss - initial_rss) / 1024 / 1024  # MB

            # Get garbage collection stats
            gc_stats = gc.get_stats()

            print(f"\nITERATION: {epoch}")
            print(f"LOSS: {avg_loss / PRINT_INTERVAL:.6f}")
            print(f"TIME: {avg_time / PRINT_INTERVAL:.6f} seconds")
            print(f"TRACED MEMORY GROWTH: {memory_growth:.2f} MB")
            print(f"RSS MEMORY GROWTH: {rss_growth:.2f} MB")
            print(f"PEAK TRACED MEMORY: {peak_traced / 1024 / 1024:.2f} MB")
            print(
                f"GC STATS: Gen0={gc_stats[0]['collections']}, Gen1={gc_stats[1]['collections']}, Gen2={gc_stats[2]['collections']}"
            )

            # Count objects of different types
            object_counts = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

            # Show count of nabla-related objects
            nabla_objects = {
                k: v
                for k, v in object_counts.items()
                if "nabla" in k.lower() or "array" in k.lower()
            }
            if nabla_objects:
                print(f"NABLA OBJECTS: {nabla_objects}")

            # Reset averages
            avg_loss = 0.0
            avg_time = 0.0

    print("\nTraining completed!")

    # Stop memory tracking and show final stats
    tracemalloc.stop()


if __name__ == "__main__":
    test_mlp_training_with_benchmark()
