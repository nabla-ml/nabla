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
Neural Network Function Approximation with Nabla

Demonstrates how a neural network learns to approximate a complex mathematical function.
Features real-time visualization with 3Blue1Brown-style dark theme.
"""

import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import nabla as nb

# =============================================================================
# DATA GENERATION
# =============================================================================


def target_function_numpy(x):
    """Complex multi-modal target function with varied behaviors."""
    # Main wave patterns
    base_wave = np.sin(x) + 0.6 * np.sin(2.1 * x)
    modulated = 0.4 * np.sin(4 * x) * np.exp(-0.08 * x**2)  # Gentler modulation
    chirp = 0.3 * np.sin(x**2 * 0.3)  # Slower frequency increase
    damped_osc = 0.5 * np.cos(3.5 * x) * np.exp(-0.03 * np.abs(x))  # Slower decay

    # Add some interesting features without too much noise
    sharp_feature = 0.4 * np.tanh(8 * (x + 1.5)) - 0.4 * np.tanh(8 * (x - 1.5))
    local_bump = (
        0.3 * np.exp(-1.5 * (x - 0.5) ** 2) * np.sin(8 * x)
    )  # Less frantic oscillation

    return base_wave + modulated + chirp + damped_osc + sharp_feature + local_bump


def generate_regression_data_numpy(num_samples=300, noise=0.0, seed=42):
    """Generate training data from the target function."""
    np.random.seed(seed)

    x_range = 3 * np.pi
    x_data = np.random.uniform(-x_range, x_range, num_samples)
    y_data = target_function_numpy(x_data)

    if noise > 0:
        y_data = y_data + np.random.normal(0, noise, num_samples)
        print(f"🔊 Noise level: {noise:.3f}")
    else:
        print("🎯 Using clean target function for training")

    print(
        f"📊 Data range: [{x_data.min():.2f}, {x_data.max():.2f}] | Target range: [{y_data.min():.2f}, {y_data.max():.2f}]"
    )

    return x_data.astype(np.float32), y_data.astype(np.float32)


def create_evaluation_grid_numpy(x_data, resolution=1000, padding=0.5):
    """Create dense x grid for smooth function plotting."""
    x_min = x_data.min() - padding
    x_max = x_data.max() + padding

    x_grid = np.linspace(x_min, x_max, resolution)
    y_true_grid = target_function_numpy(x_grid)

    return x_grid.astype(np.float32), y_true_grid.astype(np.float32), (x_min, x_max)


def cosine_annealing_schedule_numpy(epoch, base_lr, total_epochs, min_lr_ratio=0.1):
    """Cosine annealing learning rate schedule."""
    cosine_factor = (1 + np.cos(np.pi * epoch / total_epochs)) / 2
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_factor)


# =============================================================================
# NEURAL NETWORK
# =============================================================================


def init_network_params_nabla(layer_sizes, seed=42):
    """Initialize network weights using He initialization optimized for Leaky ReLU."""
    np.random.seed(seed)
    params = []
    for i, (in_size, out_size) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
    ):
        if i == len(layer_sizes) - 2:  # Output layer
            # Use smaller initialization for output layer to prevent large initial outputs
            W_np = np.random.normal(
                0, np.sqrt(1.0 / in_size), (in_size, out_size)
            ).astype(np.float32)
            b_np = np.zeros(out_size, dtype=np.float32)
        else:  # Hidden layers
            # He initialization for Leaky ReLU (accounts for alpha=0.01)
            fan_in = in_size
            # For leaky ReLU with alpha=0.01, effective fan_in is larger
            effective_fan_in = fan_in * (1 + 0.01**2) / 2
            std = np.sqrt(2.0 / effective_fan_in)
            W_np = np.random.normal(0, std, (in_size, out_size)).astype(np.float32)

            # Small positive bias for hidden layers to help with dead neurons
            b_np = np.random.normal(0, 0.01, out_size).astype(np.float32)

        W = nb.Array.from_numpy(W_np)
        b = nb.Array.from_numpy(b_np)
        params.append((W, b))
    return params


def swish_nabla(x):
    """Swish activation function (x * sigmoid(x)) - often better than ReLU for function approximation."""
    return x * nb.sigmoid(x)


def leaky_relu_nabla(x, alpha=0.01):
    """Leaky ReLU activation function."""
    zeros = nb.zeros_like(x)
    alpha_val = nb.array(alpha, dtype=x.dtype)
    return nb.where(x > zeros, x, alpha_val * x)


def forward_pass_nabla(params, x):
    """Forward pass through the neural network."""
    h = x
    for W, b in params[:-1]:
        h = leaky_relu_nabla(nb.matmul(h, W) + b)  # Back to Leaky ReLU

    output = nb.matmul(h, params[-1][0]) + params[-1][1]
    return output


def mse_loss_nabla(params, x, y):
    """Mean squared error loss function."""
    predictions = forward_pass_nabla(params, x)

    y_reshaped = y.reshape((-1, 1)) if len(y.shape) == 1 else y
    predictions_reshaped = (
        predictions.reshape((-1, 1)) if len(predictions.shape) == 1 else predictions
    )

    diff = predictions_reshaped - y_reshaped
    mse = nb.mean(diff * diff)
    return mse


def init_adam_state_nabla(params):
    """Initialize Adam optimizer state."""
    m_state = []
    v_state = []
    for W, b in params:
        m_W = nb.zeros_like(W)
        m_b = nb.zeros_like(b)
        v_W = nb.zeros_like(W)
        v_b = nb.zeros_like(b)
        m_state.append((m_W, m_b))
        v_state.append((v_W, v_b))
    return m_state, v_state


@nb.jit
def train_step_adam_nabla(
    params, m_state, v_state, x, y, lr, step, beta1=0.9, beta2=0.999, eps=1e-8
):
    """Adam optimizer training step with JIT compilation."""

    def loss_fn(params_inner):
        return mse_loss_nabla(params_inner, x, y)

    loss, grads = nb.value_and_grad(loss_fn)(params)

    updated_params = []
    updated_m = []
    updated_v = []

    for (W, b), (dW, db), (mW, mb), (vW, vb) in zip(
        params, grads, m_state, v_state, strict=False
    ):
        new_mW = beta1 * mW + (1 - beta1) * dW
        new_mb = beta1 * mb + (1 - beta1) * db

        new_vW = beta2 * vW + (1 - beta2) * (dW * dW)
        new_vb = beta2 * vb + (1 - beta2) * (db * db)

        bias_corr1 = 1 - nb.pow(beta1, step)
        bias_corr2 = 1 - nb.pow(beta2, step)

        mW_hat = new_mW / bias_corr1
        mb_hat = new_mb / bias_corr1
        vW_hat = new_vW / bias_corr2
        vb_hat = new_vb / bias_corr2

        new_W = W - lr * mW_hat / (nb.sqrt(vW_hat) + eps)
        new_b = b - lr * mb_hat / (nb.sqrt(vb_hat) + eps)

        updated_params.append((new_W, new_b))
        updated_m.append((new_mW, new_mb))
        updated_v.append((new_vW, new_vb))

    return updated_params, updated_m, updated_v, loss


def predict_function_nabla(params, x):
    """Get function predictions from the model."""
    return forward_pass_nabla(params, x)


# =============================================================================
# VISUALIZATION
# =============================================================================

plt.style.use("default")
fig, (ax_main, ax_loss) = plt.subplots(1, 2, figsize=(16, 9), width_ratios=[2, 1])

# 3Blue1Brown color scheme
background_color = "#000000"
fig.patch.set_facecolor(background_color)  # type: ignore
ax_main.set_facecolor(background_color)
ax_loss.set_facecolor(background_color)

ax_main.set_position([0.10, 0.15, 0.48, 0.70])
ax_loss.set_position([0.68, 0.15, 0.26, 0.70])

color_true = "#58C4DD"
color_pred = "#FC6255"
color_loss = "#83C167"
color_text = "#ECECEC"
color_grid = "#2A2A3E"

loss_history = []
epoch_history = []
global_num_epochs = 500


def plot_function_approximation(
    params, x_data, y_data, x_grid, y_true_grid, layer_sizes, epoch=0, loss=0.0
):
    """Plot function approximation with 3Blue1Brown styling."""

    ax_main.clear()
    ax_main.set_facecolor(background_color)

    x_grid_nabla = nb.Array.from_numpy(x_grid.reshape(-1, 1))
    y_pred_nabla = predict_function_nabla(params, x_grid_nabla)
    y_pred_grid = y_pred_nabla.to_numpy().flatten()

    ax_main.plot(
        x_grid,
        y_true_grid,
        color=color_true,
        linewidth=4,
        label="Target Function",
        alpha=0.95,
        zorder=3,
    )
    ax_main.plot(
        x_grid,
        y_pred_grid,
        color=color_pred,
        linewidth=3,
        label="NN Prediction",
        alpha=0.9,
        zorder=4,
    )

    ax_main.set_xlabel("x", fontsize=14, color=color_text, fontweight="bold")
    ax_main.set_ylabel("f(x)", fontsize=14, color=color_text, fontweight="bold")
    ax_main.set_title(
        f"Neural Network Function Approximation\nArchitecture: {' → '.join(map(str, layer_sizes))} | Epoch {epoch:03d} | MSE: {loss:.6f}",
        fontsize=16,
        color=color_text,
        fontweight="bold",
        pad=25,
    )

    legend = ax_main.legend(loc="upper right", fontsize=12, framealpha=0.9)
    legend.get_frame().set_facecolor("#1A1A2E")
    legend.get_frame().set_edgecolor(color_text)
    for text in legend.get_texts():
        text.set_color(color_text)

    ax_main.grid(True, alpha=0.3, color=color_grid, linewidth=0.8)
    ax_main.set_xlim(-10, 10)
    ax_main.set_ylim(-3.0, 3.0)

    ax_main.tick_params(colors=color_text, which="both", labelsize=11)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    loss_history.append(loss)
    epoch_history.append(epoch)

    ax_loss.clear()
    ax_loss.set_facecolor(background_color)

    if len(loss_history) > 1:
        ax_loss.plot(
            epoch_history, loss_history, color=color_loss, linewidth=3, alpha=0.9
        )
        ax_loss.fill_between(epoch_history, loss_history, alpha=0.2, color=color_loss)

        ax_loss.set_xlabel("Epoch", fontsize=12, color=color_text, fontweight="bold")
        ax_loss.set_ylabel("MSE Loss", fontsize=12, color=color_text, fontweight="bold")
        ax_loss.set_title(
            "Training Loss", fontsize=14, color=color_text, fontweight="bold", pad=20
        )
        ax_loss.grid(True, alpha=0.3, color=color_grid, linewidth=0.8)
        ax_loss.set_yscale("log")

        ax_loss.tick_params(colors=color_text, which="both", labelsize=10)
        for spine in ax_loss.spines.values():
            spine.set_visible(False)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    video_only = "--video" in sys.argv

    # Hyperparameters - back to original working version
    layer_sizes = [1, 4, 4, 1]
    learning_rate = 0.01
    num_epochs = 600

    global global_num_epochs
    global_num_epochs = num_epochs

    if video_only:
        print("🎬 Video-only mode: Creating MP4 file...")
    else:
        print("🎮 Live animation mode...")

    print("🚀 Generating regression dataset...")
    x_data, y_data = generate_regression_data_numpy(num_samples=400, noise=0.0, seed=42)
    x_grid, y_true_grid, x_bounds = create_evaluation_grid_numpy(
        x_data, resolution=1000, padding=0.5
    )

    x_data_nabla = nb.Array.from_numpy(x_data.reshape(-1, 1))
    y_data_nabla = nb.Array.from_numpy(y_data.reshape(-1, 1))

    print("🧠 Initializing neural network...")
    params = init_network_params_nabla(layer_sizes, seed=42)
    m_state, v_state = init_adam_state_nabla(params)

    total_params = sum(W.to_numpy().size + b.to_numpy().size for W, b in params)
    print(f"📊 Dataset: {len(x_data)} samples | Network: {layer_sizes}")
    print(f"⚙️  Parameters: {total_params:,} | Optimizer: Adam")
    print(f"🎯 Learning rate: {learning_rate} | Epochs: {num_epochs}")
    print("=" * 60)

    params_container = [params]
    m_container = [m_state]
    v_container = [v_state]
    step_container = [1]

    initial_loss = (
        mse_loss_nabla(params_container[0], x_data_nabla, y_data_nabla)
        .to_numpy()
        .item()
    )
    plot_function_approximation(
        params_container[0],
        x_data,
        y_data,
        x_grid,
        y_true_grid,
        layer_sizes,
        epoch=0,
        loss=initial_loss,
    )

    def update(frame):
        current_lr = cosine_annealing_schedule_numpy(
            frame, learning_rate, num_epochs, min_lr_ratio=0.6
        )

        params_container[0], m_container[0], v_container[0], loss = (
            train_step_adam_nabla(
                params_container[0],
                m_container[0],
                v_container[0],
                x_data_nabla,
                y_data_nabla,
                current_lr,
                step_container[0],
            )
        )
        step_container[0] += 1

        if frame % 2 == 0:
            loss_val = loss.to_numpy().item()
            print(f"Epoch {frame:03d}, MSE Loss: {loss_val:.6f}, LR: {current_lr:.6f}")
            plot_function_approximation(
                params_container[0],
                x_data,
                y_data,
                x_grid,
                y_true_grid,
                layer_sizes,
                epoch=frame,
                loss=loss_val,
            )

        return []

    ani = animation.FuncAnimation(
        fig, update, frames=num_epochs, repeat=False, interval=50, blit=False
    )

    if video_only:
        print("💾 Saving animation to 'nonlinear_regression.mp4'...")
        ani.save("nonlinear_regression.mp4", writer="ffmpeg", fps=20, bitrate=1800)
        print("✅ Animation saved successfully!")
    else:
        plt.show()


if __name__ == "__main__":
    main()
