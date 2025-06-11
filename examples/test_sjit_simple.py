#!/usr/bin/env python3
"""Simple test to compare static JIT vs regular JIT with the same setup."""

import numpy as np

import nabla as nb

# Configuration - use simpler setup to isolate the issue
BATCH_SIZE = 128
LAYERS = [1, 4, 1]  # Very simple network
LEARNING_RATE = 0.001
NUM_EPOCHS = 5  # Just a few epochs to see the issue
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
    loss = nb.sum(squared_errors) / batch_size
    return loss


def mlp_forward_and_loss(inputs: list[nb.Array]) -> nb.Array:
    """Combined forward pass and loss computation for VJP."""
    x, targets, *params = inputs
    predictions = mlp_forward(x, params)
    loss = mean_squared_error(predictions, targets)
    return loss


def create_sin_dataset(batch_size: int = 256) -> tuple[nb.Array, nb.Array]:
    """Create the COMPLEX 8-period sin dataset."""
    x = nb.rand((batch_size, 1), lower=0.0, upper=1.0, dtype=nb.DType.float32)
    targets = nb.sin(SIN_PERIODS * 2.0 * np.pi * x) / 2.0 + 0.5
    return x, targets


def initialize_params(layers: list[int], seed: int = 42) -> list[nb.Array]:
    """Initialize parameters."""
    np.random.seed(seed)
    params = []

    for i in range(len(layers) - 1):
        fan_in, fan_out = layers[i], layers[i + 1]
        std = (2.0 / fan_in) ** 0.5
        w_np = np.random.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)
        b_np = np.zeros((1, fan_out), dtype=np.float32)

        w = nb.Array.from_numpy(w_np)
        b = nb.Array.from_numpy(b_np)
        params.extend([w, b])

    return params


def adamw_step(
    params: list[nb.Array],
    gradients: list[nb.Array],
    m_states: list[nb.Array],
    v_states: list[nb.Array],
    step: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[list[nb.Array], list[nb.Array], list[nb.Array]]:
    """AdamW optimizer step with weight decay."""
    updated_params = []
    updated_m = []
    updated_v = []

    for param, grad, m, v in zip(params, gradients, m_states, v_states, strict=False):
        # Update moments
        new_m = beta1 * m + (1.0 - beta1) * grad
        new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

        # Bias correction
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        # Corrected moments
        m_corrected = new_m / bias_correction1
        v_corrected = new_v / bias_correction2

        # Parameter update
        new_param = param - learning_rate * (
            m_corrected / (v_corrected**0.5 + eps) + weight_decay * param
        )

        updated_params.append(new_param)
        updated_m.append(new_m)
        updated_v.append(new_v)

    return updated_params, updated_m, updated_v


def init_adamw_state(params: list[nb.Array]) -> tuple[list[nb.Array], list[nb.Array]]:
    """Initialize AdamW state."""
    m_states = []
    v_states = []
    for param in params:
        m_np = np.zeros_like(param.to_numpy())
        v_np = np.zeros_like(param.to_numpy())
        m_states.append(nb.Array.from_numpy(m_np))
        v_states.append(nb.Array.from_numpy(v_np))
    return m_states, v_states


@nb.jit
def train_step_regular_jit(
    x: nb.Array,
    targets: nb.Array,
    params: list[nb.Array],
    m_states: list[nb.Array],
    v_states: list[nb.Array],
    step: int,
    learning_rate: float,
) -> tuple[list[nb.Array], list[nb.Array], list[nb.Array], nb.Array]:
    """Regular JIT-compiled training step."""
    # Prepare inputs for value_and_grad
    all_inputs = [x, targets] + params
    param_indices = list(range(2, 2 + len(params)))

    # Forward pass + gradients using value_and_grad
    loss_value, param_gradients = nb.value_and_grad(
        mlp_forward_and_loss, argnums=param_indices
    )(all_inputs)

    # AdamW optimizer update
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, learning_rate
    )

    return updated_params, updated_m, updated_v, loss_value


@nb.sjit
def train_step_static_jit(
    x: nb.Array,
    targets: nb.Array,
    params: list[nb.Array],
    m_states: list[nb.Array],
    v_states: list[nb.Array],
    step: int,
    learning_rate: float,
) -> tuple[list[nb.Array], list[nb.Array], list[nb.Array], nb.Array]:
    """Static JIT-compiled training step."""
    # Prepare inputs for value_and_grad
    all_inputs = [x, targets] + params
    param_indices = list(range(2, 2 + len(params)))

    # Forward pass + gradients using value_and_grad
    loss_value, param_gradients = nb.value_and_grad(
        mlp_forward_and_loss, argnums=param_indices
    )(all_inputs)

    # AdamW optimizer update
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, learning_rate
    )

    return updated_params, updated_m, updated_v, loss_value


def test_jit_comparison():
    """Test regular JIT vs static JIT with identical setup."""
    print("=== Comparing Regular JIT vs Static JIT ===")
    print(f"Architecture: {LAYERS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")

    # Use the same data for both tests
    x, targets = create_sin_dataset(BATCH_SIZE)

    for jit_type in ["Regular JIT", "Static JIT"]:
        print(f"\n=== {jit_type} ===")

        # Initialize parameters (same seed for both)
        params = initialize_params(LAYERS, seed=42)
        m_states, v_states = init_adamw_state(params)

        # Choose the JIT function
        if jit_type == "Regular JIT":
            train_step = train_step_regular_jit
        else:
            train_step = train_step_static_jit

        # Training loop
        for epoch in range(1, NUM_EPOCHS + 1):
            # Use the same data for each epoch for exact comparison
            updated_params, updated_m, updated_v, loss_values = train_step(
                x, targets, params, m_states, v_states, epoch, LEARNING_RATE
            )

            params, m_states, v_states = updated_params, updated_m, updated_v
            loss_value = loss_values.to_numpy().item()

            print(f"Epoch {epoch} - Loss: {loss_value:.6f}")


if __name__ == "__main__":
    test_jit_comparison()
