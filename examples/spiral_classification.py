"""
Spiral Classification with Neural Networks

A demonstration of separating concerns between NumPy (data generation/visualization) 
and Nabla (neural network training) for classifying complex spiral datasets.

NumPy handles: Data generation, preprocessing, visualization, learning rate scheduling
Nabla handles: Neural network operations, automatic differentiation, JIT compilation
"""

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.animation as animation  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore
import sys

import nabla as nb
from nabla.nn.layers.activations import log_softmax

# =============================================================================
# DATA GENERATION & PREPROCESSING (NumPy)
# =============================================================================

def generate_spiral_data_numpy(num_samples=600, noise=0.08, seed=42):
    """Generate two intertwined oval spirals with complex variations."""
    np.random.seed(seed)
    n_per_class = num_samples // 2
    
    theta = np.linspace(0, 5 * np.pi, n_per_class)
    
    # Radius function with sinusoidal variation
    base_radius = 0.26 + theta / (5 * np.pi) * 4.55
    radius_variation = 0.325 * np.sin(theta * 1.5) * np.exp(-theta / (10 * np.pi))
    
    # Oval scaling
    ellipse_scale_x = 2
    ellipse_scale_y = 1.8
    
    # Spiral 1
    r1 = base_radius + radius_variation
    x1 = r1 * np.cos(theta) * ellipse_scale_x + np.random.normal(0, noise, n_per_class)
    y1 = r1 * np.sin(theta) * ellipse_scale_y + np.random.normal(0, noise, n_per_class)
    
    # Spiral 2 with spacing offset
    phase_offset = np.pi + 0.3 * np.sin(theta * 0.3)
    radial_offset = 0.195
    r2 = base_radius + radius_variation * np.cos(theta * 0.1) + radial_offset
    
    x2 = r2 * np.cos(theta + phase_offset) * ellipse_scale_x + np.random.normal(0, noise, n_per_class)
    y2 = r2 * np.sin(theta + phase_offset) * ellipse_scale_y + np.random.normal(0, noise, n_per_class)
    
    # Debug noise
    if noise > 0:
        noise_x1 = np.random.normal(0, noise, n_per_class)
        noise_y1 = np.random.normal(0, noise, n_per_class)
        print(f"🔊 Noise level: {noise:.3f} | X1 noise std: {np.std(noise_x1):.3f} | Y1 noise std: {np.std(noise_y1):.3f}")
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X, y.astype(int)

def create_mesh_grid_numpy(X, resolution=250, padding=1.5):
    """Create mesh grid for decision boundary visualization - force square."""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    
    # Force square by using the larger range for both dimensions
    data_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Create square bounds
    x_min = x_center - data_range / 2
    x_max = x_center + data_range / 2
    y_min = y_center - data_range / 2
    y_max = y_center + data_range / 2
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    return xx, yy, (x_min, x_max, y_min, y_max)

def cosine_annealing_schedule_numpy(epoch, base_lr, total_epochs, min_lr_ratio=0.1):
    """Cosine annealing learning rate schedule with minimum learning rate."""
    cosine_factor = (1 + np.cos(np.pi * epoch / total_epochs)) / 2
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_factor)

# =============================================================================
# NEURAL NETWORK TRAINING & OPTIMIZATION (Nabla)
# =============================================================================

def init_network_params_nabla(layer_sizes, seed=42):
    """Initialize network weights using Xavier initialization."""
    np.random.seed(seed)
    params = []
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        W_np = np.random.normal(0, np.sqrt(2.0 / (in_size + out_size)), (in_size, out_size)).astype(np.float32)
        b_np = np.zeros(out_size, dtype=np.float32)
        
        W = nb.Array.from_numpy(W_np)
        b = nb.Array.from_numpy(b_np)
        params.append((W, b))
    return params

def leaky_relu_nabla(x, alpha=0.01):
    """Leaky ReLU activation function."""
    zeros = nb.zeros_like(x)
    alpha_val = nb.array(alpha, dtype=x.dtype)
    return nb.where(x > zeros, x, alpha_val * x)

def forward_pass_nabla(params, x):
    """Forward pass through the neural network."""
    h = x
    for W, b in params[:-1]:
        h = leaky_relu_nabla(nb.matmul(h, W) + b)
    logits = nb.matmul(h, params[-1][0]) + params[-1][1]
    return logits

def create_one_hot_nabla(y, num_classes):
    """Create one-hot encoding using Nabla operations."""
    batch_size = y.shape[0]
    y_expanded = y.reshape((batch_size, 1))
    class_indices = nb.ndarange((num_classes,), dtype=y.dtype).reshape((1, num_classes))
    one_hot = nb.equal(y_expanded, class_indices).astype(nb.DType.float32)
    return one_hot

def cross_entropy_loss_nabla(params, x, y):
    """Cross-entropy loss function."""
    logits = forward_pass_nabla(params, x)
    one_hot_y = create_one_hot_nabla(y, num_classes=2)
    log_probs = log_softmax(logits, axis=-1)
    
    cross_entropy = -nb.sum(one_hot_y * log_probs)
    batch_size = nb.array(logits.shape[0], dtype=nb.DType.float32)
    return cross_entropy / batch_size

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
def train_step_adam_nabla(params, m_state, v_state, x, y, lr, step, 
                         beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer training step with JIT compilation."""
    
    def loss_fn(params_inner):
        return cross_entropy_loss_nabla(params_inner, x, y)
    
    loss, grads = nb.value_and_grad(loss_fn)(params)
    
    updated_params = []
    updated_m = []
    updated_v = []
    
    for (W, b), (dW, db), (mW, mb), (vW, vb) in zip(params, grads, m_state, v_state):
        # Update biased first moment estimate
        new_mW = beta1 * mW + (1 - beta1) * dW
        new_mb = beta1 * mb + (1 - beta1) * db

        # Update biased second raw moment estimate
        new_vW = beta2 * vW + (1 - beta2) * (dW * dW)
        new_vb = beta2 * vb + (1 - beta2) * (db * db)
        
        # Bias correction
        bias_corr1 = 1 - nb.pow(beta1, step)
        bias_corr2 = 1 - nb.pow(beta2, step)

        mW_hat = new_mW / bias_corr1
        mb_hat = new_mb / bias_corr1
        vW_hat = new_vW / bias_corr2
        vb_hat = new_vb / bias_corr2
        
        # Update parameters
        new_W = W - lr * mW_hat / (nb.sqrt(vW_hat) + eps)
        new_b = b - lr * mb_hat / (nb.sqrt(vb_hat) + eps)

        updated_params.append((new_W, new_b))
        updated_m.append((new_mW, new_mb))
        updated_v.append((new_vW, new_vb))
    
    return updated_params, updated_m, updated_v, loss

def predict_probabilities_nabla(params, x):
    """Get probability predictions from the model."""
    logits = forward_pass_nabla(params, x)
    return nb.softmax(logits, axis=-1)

# =============================================================================
# VISUALIZATION (Matplotlib)
# =============================================================================

# Setup matplotlib styling - 1:1 aspect ratio, no labels
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 10))  # Square format

background_color = "#FFFFFF"
fig.patch.set_facecolor(background_color)  # type: ignore
ax.set_facecolor(background_color)

# Color scheme
color_class_0 = "#4CB2FF"  # Blue
color_class_1 = "#F1AF79"  # Orange
class_colors = [color_class_0, color_class_1]

boundary_cmap = LinearSegmentedColormap.from_list(
    'direct_gradient', [color_class_0, color_class_1], N=256
)

def plot_decision_boundary(params, X, y, epoch=0, loss=0.0):
    """Plot data points and decision boundary - clean, no labels."""
    ax.clear()
    ax.set_facecolor('#FFFFFF')
    
    # Create mesh for visualization
    xx, yy, bounds = create_mesh_grid_numpy(X, resolution=250, padding=1.5)
    x_min, x_max, y_min, y_max = bounds
    
    # Get predictions
    grid_np = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_nabla = nb.Array.from_numpy(grid_np)
    probs_nabla = predict_probabilities_nabla(params, grid_nabla)[:, 1]
    
    Z = probs_nabla.to_numpy().reshape(xx.shape)
    
    # Plot decision boundary
    contour_filled = ax.contourf(xx, yy, Z, levels=8, cmap=boundary_cmap, alpha=0.99)
    ax.contour(xx, yy, Z, levels=8, colors=["white"], linewidths=1.0, alpha=0.6)
    
    # Plot data points
    for i, color in enumerate(class_colors):
        mask = y == i
        ax.scatter(
            X[mask, 0], X[mask, 1], 
            c=color, s=60, alpha=0.9,
            zorder=5, edgecolors='white', linewidths=1.5
        )
    
    # Clean styling - no labels, axes, or anything
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Remove all axes, labels, ticks, spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove margins for clean edge-to-edge display
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Check for video-only mode
    video_only = '--video' in sys.argv
    
    # Hyperparameters
    layer_sizes = [2, 32, 64, 32, 2]
    learning_rate = 0.003
    num_epochs = 400
    
    if video_only:
        print("🎬 Video-only mode: Creating MP4 file...")
    else:
        print("🎮 Live animation mode...")
    
    # Generate data
    print("🚀 Generating spiral dataset...")
    X_np, y_np = generate_spiral_data_numpy(num_samples=800, noise=0.08, seed=42)
    
    X_nabla = nb.Array.from_numpy(X_np.astype(np.float32))
    y_nabla = nb.Array.from_numpy(y_np.astype(np.int32))
    
    # Initialize network
    print("🧠 Initializing neural network...")
    params = init_network_params_nabla(layer_sizes, seed=42)
    m_state, v_state = init_adam_state_nabla(params)
    
    total_params = sum(W.to_numpy().size + b.to_numpy().size for W, b in params)
    print(f"📊 Dataset: 800 samples | Network: {layer_sizes}")
    print(f"⚙️  Parameters: {total_params:,} | Optimizer: Adam")
    print(f"🎯 Learning rate: {learning_rate} | Epochs: {num_epochs}")
    print("=" * 50)

    # Animation containers
    params_container = [params]
    m_container = [m_state]
    v_container = [v_state]
    step_container = [1]
    
    plot_decision_boundary(params_container[0], X_np, y_np, epoch=0, loss=1.0)
    
    def update(frame):
        current_lr = cosine_annealing_schedule_numpy(frame, learning_rate, num_epochs, min_lr_ratio=0.6)
        
        params_container[0], m_container[0], v_container[0], loss = train_step_adam_nabla(
            params_container[0], m_container[0], v_container[0], 
            X_nabla, y_nabla, current_lr, step_container[0]
        )
        step_container[0] += 1

        if frame % 2 == 0:
            loss_val = loss.to_numpy().item()
            print(f"Epoch {frame:03d}, Loss: {loss_val:.4f}, LR: {current_lr:.6f}")
            plot_decision_boundary(params_container[0], X_np, y_np, epoch=frame, loss=loss_val)
        
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=num_epochs, repeat=False, interval=40, blit=False
    )
    
    if video_only:
        # Save animation as MP4 file
        print("💾 Saving animation to 'spiral_classification.mp4'...")
        ani.save('spiral_classification.mp4', writer='ffmpeg', fps=25, bitrate=1800)
        print("✅ Animation saved successfully!")
    else:
        plt.show()

if __name__ == "__main__":
    main()
