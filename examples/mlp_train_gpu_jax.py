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

# ===----------------------------------------------------------------------=== #
# JAX Implementation for Direct Comparison with Nabla
# ===----------------------------------------------------------------------=== #

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad

# Configuration - EXACTLY the same as Nabla version
BATCH_SIZE = 2048  # Much larger batch size to benefit GPU
LAYERS = [1, 4096, 8192, 8192, 4096, 1]  # Much larger model for GPU advantage
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # Fewer epochs since larger model
PRINT_INTERVAL = 10
SIN_PERIODS = 8

# Test both CPU and GPU
def test_device(device_name):
    """Test JAX training on specified device."""
    
    if device_name == "gpu":
        # Try to use GPU if available
        try:
            device = jax.devices("gpu")[0]
            jax.config.update("jax_default_device", device)
            print(f"Using JAX GPU device: {device}")
        except:
            print("GPU not available, falling back to CPU")
            device_name = "cpu"
            device = jax.devices("cpu")[0]
            jax.config.update("jax_default_device", device)
    else:
        device = jax.devices("cpu")[0]
        jax.config.update("jax_default_device", device)
        print(f"Using JAX CPU device: {device}")

    def mlp_forward(x, params):
        """MLP forward pass through all layers."""
        output = x
        for i in range(0, len(params) - 1, 2):
            w, b = params[i], params[i + 1]
            output = jnp.dot(output, w) + b
            # Apply ReLU to all layers except the last
            if i < len(params) - 2:
                output = jax.nn.relu(output)
        return output

    def mean_squared_error(predictions, targets):
        """Compute mean squared error loss."""
        diff = predictions - targets
        squared_errors = diff * diff
        batch_size = predictions.shape[0]
        loss = jnp.sum(squared_errors) / batch_size
        return loss

    def create_sin_dataset(batch_size=256, key=None):
        """Create the COMPLEX 8-period sin dataset."""
        if key is None:
            key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (batch_size, 1), dtype=jnp.float32)
        targets = jnp.sin(SIN_PERIODS * 2.0 * jnp.pi * x) / 2.0 + 0.5
        return x, targets

    def initialize_for_complex_function(layers, seed=42):
        """Initialize specifically for learning complex high-frequency functions."""
        key = jax.random.PRNGKey(seed)
        params = []

        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            key, subkey = jax.random.split(key)
            # He normal initialization
            std = jnp.sqrt(2.0 / fan_in)
            w = jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32) * std
            b = jnp.zeros((fan_out,), dtype=jnp.float32)
            params.append(w)
            params.append(b)
            key, subkey = jax.random.split(key)

        return params

    def adamw_step(params, gradients, m_states, v_states, step, 
                   learning_rate=0.001, beta1=0.9, beta2=0.999, 
                   eps=1e-8, weight_decay=0.01):
        """AdamW optimizer step with weight decay."""
        updated_params = []
        updated_m = []
        updated_v = []

        for param, grad, m, v in zip(params, gradients, m_states, v_states):
            # Update moments
            new_m = beta1 * m + (1.0 - beta1) * grad
            new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

            # Bias correction
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

            # Corrected moments
            m_corrected = new_m / bias_correction1
            v_corrected = new_v / bias_correction2

            # Parameter update with weight decay
            new_param = param - learning_rate * (
                m_corrected / (jnp.sqrt(v_corrected) + eps) + weight_decay * param
            )

            # Append updated values
            updated_params.append(new_param)
            updated_m.append(new_m)
            updated_v.append(new_v)

        return updated_params, updated_m, updated_v

    def init_adamw_state(params):
        """Initialize AdamW state."""
        m_states = []
        v_states = []
        for param in params:
            m_states.append(jnp.zeros_like(param))
            v_states.append(jnp.zeros_like(param))
        return m_states, v_states

    def learning_rate_schedule(epoch, initial_lr=0.001, decay_factor=0.95, decay_every=1000):
        """Learning rate schedule for complex function learning."""
        return initial_lr * (decay_factor ** (epoch // decay_every))

    @jit
    def train_step(x, targets, params, m_states, v_states, step, learning_rate):
        """JIT-compiled training step combining gradient computation and optimizer update."""
        
        def loss_fn(inner_params):
            predictions = mlp_forward(x, inner_params)
            loss = mean_squared_error(predictions, targets)
            return loss

        loss_value, param_gradients = value_and_grad(loss_fn)(params)

        # AdamW optimizer update
        updated_params, updated_m, updated_v = adamw_step(
            params, param_gradients, m_states, v_states, step, learning_rate
        )

        return updated_params, updated_m, updated_v, loss_value

    def test_jax_complex_sin():
        """Test JAX implementation with JIT for complex sin learning."""
        print(f"=== Learning COMPLEX 8-Period Sin Function with JAX JIT ({device_name.upper()}) ===")
        print(f"Architecture: {LAYERS}")
        print(f"Initial learning rate: {LEARNING_RATE}")
        print(f"Sin periods: {SIN_PERIODS}")
        print(f"Batch size: {BATCH_SIZE}")

        # Initialize for complex function learning
        params = initialize_for_complex_function(LAYERS)
        m_states, v_states = init_adamw_state(params)

        # Initial analysis
        key = jax.random.PRNGKey(42)
        x_init, targets_init = create_sin_dataset(BATCH_SIZE, key)
        predictions_init = mlp_forward(x_init, params)
        initial_loss = mean_squared_error(predictions_init, targets_init)

        pred_init_np = np.array(predictions_init)
        target_init_np = np.array(targets_init)

        print(f"Initial loss: {float(initial_loss):.6f}")
        print(f"Initial predictions range: [{pred_init_np.min():.3f}, {pred_init_np.max():.3f}]")
        print(f"Targets range: [{target_init_np.min():.3f}, {target_init_np.max():.3f}]")

        print("\nStarting training...")

        # Training loop
        avg_loss = 0.0
        avg_time = 0.0
        avg_data_time = 0.0
        avg_jit_time = 0.0

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start_time = time.time()

            # Learning rate schedule
            current_lr = learning_rate_schedule(epoch, LEARNING_RATE)

            # Create fresh batch
            data_start = time.time()
            key, subkey = jax.random.split(key)
            x, targets = create_sin_dataset(BATCH_SIZE, subkey)
            data_time = time.time() - data_start

            # Training step using JIT-compiled function
            jit_start = time.time()

            # Use JIT-compiled training step
            updated_params, updated_m, updated_v, loss_values = train_step(
                x, targets, params, m_states, v_states, epoch, current_lr
            )

            jit_time = time.time() - jit_start

            # Update return values
            params, m_states, v_states = updated_params, updated_m, updated_v

            # Loss extraction and conversion
            loss_value = float(loss_values)

            epoch_time = time.time() - epoch_start_time
            avg_loss += loss_value
            avg_time += epoch_time
            avg_data_time += data_time
            avg_jit_time += jit_time

            if epoch % PRINT_INTERVAL == 0:
                avg_time_per_epoch = avg_time / PRINT_INTERVAL if avg_time > 0 else 0
                data_perc = (avg_data_time / avg_time) * 100 if avg_time > 0 else 0
                jit_perc = (avg_jit_time / avg_time) * 100 if avg_time > 0 else 0

                print(f"\n{'=' * 60}")
                print(f"JAX Device: {device}")
                print(f"Epoch {epoch:4d} | Loss: {avg_loss / PRINT_INTERVAL:.6f} | Time: {avg_time_per_epoch:.4f}s")
                print(f"{'=' * 60}")
                print(f"  ├─ Data Gen:   {avg_data_time / PRINT_INTERVAL:.4f}s ({data_perc:.1f}%)")
                print(f"  └─ JIT Step:   {avg_jit_time / PRINT_INTERVAL:.4f}s ({jit_perc:.1f}%)")

                avg_loss = 0.0
                avg_time = 0.0
                avg_data_time = 0.0
                avg_jit_time = 0.0

        print(f"\nJAX JIT training completed on {device_name.upper()}!")
        return avg_time_per_epoch

    return test_jax_complex_sin()


if __name__ == "__main__":
    print("Testing JAX performance on CPU and GPU for MLP training")
    print("=" * 80)
    
    # Test CPU
    cpu_time = test_device("cpu")
    print("\n" + "=" * 80)
    
    # Test GPU
    gpu_time = test_device("gpu")
    
    print("\n" + "=" * 80)
    print("JAX PERFORMANCE COMPARISON:")
    print(f"CPU time per epoch: {cpu_time:.4f}s")
    print(f"GPU time per epoch: {gpu_time:.4f}s")
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"GPU Speedup: {speedup:.2f}x")
    print("=" * 80)
