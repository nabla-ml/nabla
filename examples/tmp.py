"""
Ablation Test: This script tests ONLY the RMSProp part of the Adam optimizer.
It removes momentum entirely to isolate the buggy GPU kernel.
"""
import time
import numpy as np
from max import driver
import nabla as nb

# --- 1. Configuration ---
LEARNING_RATE = 0.001 # RMSProp can be sensitive
NUM_EPOCHS = 300
PRINT_INTERVAL = 20
LAYERS = [1, 64, 1]
BATCH_SIZE = 16

# --- 2. Device Setup ---
# We force the GPU to find the bug.
try:
    if driver.accelerator_count() == 0:
        print("❌ No GPU found. This test requires a GPU.")
        exit()
    device = driver.Accelerator()
    print(f"✅ Forcing device: {device}")
except Exception as e:
    print(f"❌ Failed to initialize device. Error: {e}")
    exit()

# --- 3. Optimizer and State Initialization Functions ---
def rmsprop_step(
    params: list[nb.Array],
    gradients: list[nb.Array],
    v_states: list[nb.Array],
    step: int,
    learning_rate: float,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[list[nb.Array], list[nb.Array]]:
    """Performs an RMSProp update step."""
    updated_params, updated_v = [], []
    for param, grad, v in zip(params, gradients, v_states):
        # grad.realize()
        # print(grad)
        # Calculate the second moment (v_state)
        new_v = beta2 * v + (1.0 - beta2) * (grad * grad)
        
        # Bias correction for v
        bias_correction2 = 1.0 - beta2**step
        v_corrected = new_v / bias_correction2
        
        # Update parameters by scaling the gradient
        new_param = param - learning_rate * (grad / (v_corrected**0.5 + eps))
        
        updated_params.append(new_param)
        updated_v.append(new_v)
    return updated_params, updated_v

def init_v_state(params: list[nb.Array]) -> list[nb.Array]:
    """Initialize RMSProp state (v_states) with zeros."""
    v_states = []
    for param in params:
        v_np = np.zeros_like(param.to_numpy())
        v_states.append(nb.Array.from_numpy(v_np).to(param.device))
    return v_states

# --- 4. The Core JIT-Compiled Training Step ---
@nb.djit
def train_step_rmsprop(x, targets, params, v_states, step, lr):
    """
    Performs a JIT-compiled training step with RMSProp only.
    """
    w1, b1, w2, b2 = params[0], params[1], params[2], params[3]

    def loss_fn(iw1, ib1, iw2, ib2):
        hidden = nb.matmul(x, iw1) + ib1
        activated = nb.relu(hidden)
        predictions = nb.matmul(activated, iw2) + ib2
        diff = predictions - targets
        n = nb.array(predictions.shape[0], dtype=nb.DType.float32).to(predictions.device)
        return nb.sum(diff * diff) / n

    loss, grads = nb.value_and_grad(loss_fn, argnums=[0, 1, 2, 3])(w1, b1, w2, b2)

    updated_params, updated_v = rmsprop_step(params, list(grads), v_states, step, lr)
    return updated_params, updated_v, loss

# --- 5. Main Execution Logic ---
if __name__ == "__main__":
    # --- Data and Model Initialization ---
    print("\n--- Preparing Data and Model ---")
    X_np = np.linspace(-1, 1, BATCH_SIZE).reshape(BATCH_SIZE, 1).astype(np.float32)
    Y_np = 2.0 * X_np + 1.0
    x_train = nb.Array.from_numpy(X_np).to(device)
    y_train = nb.Array.from_numpy(Y_np).to(device)
    params = [
        nb.he_normal((LAYERS[0], LAYERS[1]), seed=42).to(device),
        nb.zeros((LAYERS[1],)).to(device),
        nb.he_normal((LAYERS[1], LAYERS[2]), seed=43).to(device),
        nb.zeros((LAYERS[2],)).to(device)
    ]
    print(f"Model with {len(params)} parameter tensors initialized.")

    # --- Initialize RMSProp State ---
    v_states = init_v_state(params)
    print("RMSProp v_states initialized.")

    # --- Training Loop ---
    print("\n--- Starting Training (MLP with ReLU and RMSProp only) ---")
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        params, v_states, loss = train_step_rmsprop(
            x_train, y_train, params, v_states, epoch, LEARNING_RATE
        )
        if epoch % PRINT_INTERVAL == 0 or epoch == 1:
            loss_val = loss.to_numpy().item()
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.8f}")
            if np.isnan(loss_val):
                print("\n" + "="*60)
                print(">>> 🎯 BUG FOUND. The NaN is produced by the RMSProp update logic.")
                print(">>> The faulty kernel is multiply, sqrt, or division.")
                print("="*60)
                exit()

    total_time = time.time() - start_time
    print("--- Training Complete ---")
    print(f"Total time: {total_time:.2f} seconds")

    # --- Final Evaluation ---
    print("\n--- Final Evaluation ---")
    X_test_np = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
    Y_test_np = 2.0 * X_test_np + 1.0
    x_test = nb.Array.from_numpy(X_test_np).to(device)
    
    w1, b1, w2, b2 = params[0], params[1], params[2], params[3]
    hidden = nb.matmul(x_test, w1) + b1
    activated = nb.relu(hidden)
    final_preds = nb.matmul(activated, w2) + b2

    final_preds_np = final_preds.to_numpy()
    final_loss = np.mean((final_preds_np - Y_test_np)**2)
    print(f"Final Test MSE: {final_loss:.8f}")

    if final_loss < 0.01:
        print("\n✅ SUCCESS: The full system is stable and learns correctly!")
    else:
        print("\n❌ FAILURE: The model did not learn correctly. Check for bugs or tuning.")