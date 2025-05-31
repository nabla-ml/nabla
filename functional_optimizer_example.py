"""Recommended functional optimizer approach for Nabla."""

import numpy as np

import nabla as nb

# =============================================================================
# Pure Functional SGD with Momentum
# =============================================================================


def sgd_momentum_step(
    params: list[nb.Array],
    gradients: list[nb.Array],
    velocities: list[nb.Array],
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> tuple[list[nb.Array], list[nb.Array]]:
    """Pure functional optimizer step - no hidden state."""
    updated_params = []
    updated_velocities = []

    momentum_tensor = nb.array([np.float32(momentum)])
    lr_tensor = nb.array([np.float32(learning_rate)])

    for param, grad, velocity in zip(params, gradients, velocities, strict=False):
        # Compute new velocity
        new_velocity = momentum_tensor * velocity - lr_tensor * grad
        # Compute new parameter
        new_param = param + new_velocity

        updated_params.append(new_param)
        updated_velocities.append(new_velocity)

    return updated_params, updated_velocities


def init_sgd_momentum_state(params: list[nb.Array]) -> list[nb.Array]:
    """Initialize optimizer state (velocities)."""
    velocities = []
    for param in params:
        v_np = np.zeros_like(param.to_numpy())
        velocities.append(nb.Array.from_numpy(v_np))
    return velocities


# =============================================================================
# Updated MLP Training with Functional Optimizer
# =============================================================================


def functional_mlp_training():
    """Your MLP training refactored to use functional optimizer."""

    # Initialize model parameters
    input_size, hidden_size, output_size = 2, 3, 1

    # Weight initialization (Xavier/Glorot)
    w1_np = np.random.normal(
        0, np.sqrt(2.0 / input_size), (input_size, hidden_size)
    ).astype(np.float32)
    b1_np = np.zeros((1, hidden_size), dtype=np.float32)
    w2_np = np.random.normal(
        0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size)
    ).astype(np.float32)
    b2_np = np.zeros((1, output_size), dtype=np.float32)

    params = [
        nb.Array.from_numpy(w1_np),
        nb.Array.from_numpy(b1_np),
        nb.Array.from_numpy(w2_np),
        nb.Array.from_numpy(b2_np),
    ]

    # Initialize optimizer state
    optimizer_state = init_sgd_momentum_state(params)

    # Training data
    X = nb.Array.from_numpy(np.random.randn(100, input_size).astype(np.float32))
    y = nb.Array.from_numpy(np.random.randn(100, output_size).astype(np.float32))

    def mlp_forward_and_loss(inputs):
        x, targets, w1, b1, w2, b2 = inputs
        # Forward pass
        h = nb.relu(nb.matmul(x, w1) + b1)
        output = nb.matmul(h, w2) + b2
        # MSE loss
        loss = nb.reduce_sum((output - targets) ** 2) / (
            nb.array([np.float32(X.shape[0])]) * nb.array([np.float32(output_size)])
        )
        return [loss]

    # Training loop
    learning_rate = 0.01
    momentum = 0.9

    for epoch in range(100):
        # Create input tuple for vjp
        all_inputs = [X, y] + params

        # Compute loss and gradients
        loss_values, vjp_fn = nb.vjp(mlp_forward_and_loss, all_inputs)
        cotangent = [nb.array([1.0])]  # Gradient w.r.t. loss
        gradients = vjp_fn(cotangent)

        # Extract parameter gradients (skip X, y)
        param_gradients = gradients[2:]

        # Functional optimizer update
        params, optimizer_state = sgd_momentum_step(
            params, param_gradients, optimizer_state, learning_rate, momentum
        )

        # Print progress
        if epoch % 20 == 0:
            loss_value = loss_values[0].to_numpy()[0]
            print(f"Epoch {epoch}, Loss: {loss_value:.6f}")

    return params


if __name__ == "__main__":
    trained_params = functional_mlp_training()
    print("Training completed with functional optimizer!")
