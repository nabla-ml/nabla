import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from typing import Any, Tuple, Dict

# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 64
    
    # Titan (High Freq) Settings
    d_memory: int = 16
    num_heads: int = 4
    
    # Neural Optimizer Settings
    optimizer_hidden_dim: int = 64
    
    # Continuum Memory Settings
    freq_mlp_hidden: int = 64
    update_freq: int = 4 
    mid_freq_lr: float = 0.1

# ==========================================
# 2. BASIC LAYERS (Pure JAX)
# ==========================================

def init_linear(rng, in_dim, out_dim):
    k1, k2 = jax.random.split(rng)
    # LeCun Normal-ish initialization
    limit = jnp.sqrt(1.0 / in_dim)
    w = jax.random.uniform(k1, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,))
    return {'w': w, 'b': b}

def linear(params, x):
    return jnp.dot(x, params['w']) + params['b']

def init_layer_norm(dim):
    return {'scale': jnp.ones((dim,)), 'bias': jnp.zeros((dim,))}

def layer_norm(params, x, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    return norm * params['scale'] + params['bias']

def swish(x):
    return x * jax.nn.sigmoid(x)

# ==========================================
# 3. MODULE: NEURAL MOMENTUM (Level 1: Deep Optimizer)
# ==========================================

def init_neural_momentum(rng, config: ModelConfig):
    k1, k2 = jax.random.split(rng)
    k1, k2 = jax.random.split(rng)
    
    # Input: Flattened gradients + Flattened momentum per head
    # Shape: 2 * (d_memory * d_memory)
    input_dim = 2 * (config.d_memory * config.d_memory)
    
    # Layer Norms
    ln_dim = config.d_memory * config.d_memory
    
    return {
        'ln_grad': init_layer_norm(ln_dim),
        'ln_mom': init_layer_norm(ln_dim),
        'l1': init_linear(k1, 2 * ln_dim, config.optimizer_hidden_dim),
        'l2': init_linear(k2, config.optimizer_hidden_dim, ln_dim) # Output matches flattened shape
    }

def neural_momentum(params, grad_signal, momentum_state, config: ModelConfig):
    # grad_signal: [Batch, Heads, D_mem, D_mem]
    batch, heads, d1, d2 = grad_signal.shape
    
    flat_grad = grad_signal.reshape(batch, heads, -1)
    flat_mom = momentum_state.reshape(batch, heads, -1)
    
    # Normalize
    norm_grad = layer_norm(params['ln_grad'], flat_grad)
    norm_mom = layer_norm(params['ln_mom'], flat_mom)
    
    inp = jnp.concatenate([norm_grad, norm_mom], axis=-1)
    
    # MLP
    hidden = linear(params['l1'], inp)
    hidden = swish(hidden)
    
    # Output update vector
    update_vector = linear(params['l2'], hidden)
    new_momentum = update_vector.reshape(grad_signal.shape)
    
    # Residual update
    combined_momentum = 0.9 * momentum_state + new_momentum + 0.1 * grad_signal
    return combined_momentum

# ==========================================
# 4. MODULE: CONTINUUM MEMORY (Level 2: Mid-Freq)
# ==========================================

def init_continuum_memory(rng, config: ModelConfig):
    # These are the "Meta-Parameters" (Initializations for the plastic weights)
    k1, k2 = jax.random.split(rng)
    
    # We use standard init for these meta-params
    w1 = jax.random.normal(k1, (config.d_model, config.freq_mlp_hidden)) * jnp.sqrt(1/config.d_model)
    b1 = jnp.zeros((config.freq_mlp_hidden,))
    w2 = jax.random.normal(k2, (config.freq_mlp_hidden, config.d_model)) * jnp.sqrt(1/config.freq_mlp_hidden)
    b2 = jnp.zeros((config.d_model,))
    
    return {
        'init_w1': w1, 'init_b1': b1,
        'init_w2': w2, 'init_b2': b2
    }

def continuum_memory_forward(params, x, carry_state, config: ModelConfig):
    # Note: Weights come from 'carry_state', so 'params' are unused here.
    
    # Unpack Plastic Weights
    (w1, b1, w2, b2), buffer, counter = carry_state
    
    # --- 1. Forward Pass ---
    h = jax.nn.relu(jnp.dot(x, w1) + b1)
    y_t = jnp.dot(h, w2) + b2
    
    # --- 2. Buffer Management ---
    idx = counter % config.update_freq
    new_buffer = buffer.at[idx].set(x)
    should_update = (idx == config.update_freq - 1)
    
    # --- 3. Inner Loop Update ---
    def update_weights(curr_params, buff):
        (cw1, cb1, cw2, cb2) = curr_params
        
        targets = jnp.roll(buff, -1, axis=0)
        targets = jax.lax.stop_gradient(targets)
        
        def local_loss(p_w1, p_b1, p_w2, p_b2):
            h_buf = jax.nn.relu(jnp.dot(buff, p_w1) + p_b1)
            preds = jnp.dot(h_buf, p_w2) + p_b2
            return jnp.mean((preds - targets) ** 2)
        
        grads = jax.grad(local_loss, argnums=(0,1,2,3))(cw1, cb1, cw2, cb2)
        
        lr = config.mid_freq_lr
        new_params = (
            cw1 - lr * grads[0],
            cb1 - lr * grads[1],
            cw2 - lr * grads[2],
            cb2 - lr * grads[3]
        )
        return new_params

    new_weights = jax.lax.cond(
        should_update,
        update_weights,
        lambda p, b: p,
        (w1, b1, w2, b2),
        new_buffer
    )
    
    return y_t, (new_weights, new_buffer, counter + 1)

# ==========================================
# 5. MODULE: HOPE BLOCK (Level 1 + 2 Integration)
# ==========================================

def init_hope(rng, config: ModelConfig):
    k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    dim = config.num_heads * config.d_memory
    
    return {
        'W_q': init_linear(k1, config.d_model, dim),
        'W_k': init_linear(k2, config.d_model, dim),
        'W_v': init_linear(k3, config.d_model, dim),
        'W_o': init_linear(k4, dim, config.d_model),
        'ln1': init_layer_norm(config.d_model),
        'ln2': init_layer_norm(config.d_model),
        'deep_opt': init_neural_momentum(k5, config),
        'continuum': init_continuum_memory(k6, config),
        'decay': jnp.array([0.95]) # Learnable decay
    }

def hope_forward(params, x_seq, config: ModelConfig):
    # x_seq: [Batch, Time, Dim]
    batch_size, seq_len, _ = x_seq.shape
    
    # --- 1. Projections ---
    x_norm = layer_norm(params['ln1'], x_seq)
    
    # Apply linear layers (vmap over time and batch implicitly via dot broadcast)
    # But our linear is dot(x, w) + b. x is [B, T, D], w is [D, Out]. Result [B, T, Out]. Correct.
    
    q_flat = linear(params['W_q'], x_norm)
    k_flat = linear(params['W_k'], x_norm)
    v_flat = linear(params['W_v'], x_norm)
    
    # Reshape for heads: [Batch, Time, Heads, D_mem]
    def split(x):
        return x.reshape(batch_size, seq_len, config.num_heads, config.d_memory)
    
    Q = split(q_flat)
    K = split(k_flat)
    V = split(v_flat)
    
    # --- 2. Initialize States ---
    init_M = jnp.zeros((batch_size, config.num_heads, config.d_memory, config.d_memory))
    init_Mom = jnp.zeros_like(init_M)
    
    # Init Continuum State
    # Get initial plastic weights from meta-params
    c_params = params['continuum']
    init_plastic = (c_params['init_w1'], c_params['init_b1'], c_params['init_w2'], c_params['init_b2'])
    
    # Broadcast to batch
    init_plastic_batch = jax.tree.map(
        lambda p: jnp.repeat(p[None, ...], batch_size, axis=0),
        init_plastic
    )
    init_buffer = jnp.zeros((batch_size, config.update_freq, config.d_model))
    init_counter = jnp.zeros((batch_size,), dtype=jnp.int32)
    
    init_mid_carry = (init_plastic_batch, init_buffer, init_counter)
    
    # --- 3. Scan Prep ---
    # Transpose to [Time, Batch, ...] for scan
    Q_T = jnp.swapaxes(Q, 0, 1)
    K_T = jnp.swapaxes(K, 0, 1)
    V_T = jnp.swapaxes(V, 0, 1)
    x_T = jnp.swapaxes(x_seq, 0, 1)
    
    def scan_step(carry, inputs):
        (M, Mom), (mlp_weights, buf, ctr) = carry
        q, k, v, x_raw = inputs # [Batch, ...]
        
        # A. Mid-Frequency (Continuum)
        # vmap over batch
        mlp_out, new_mid_carry = jax.vmap(continuum_memory_forward, in_axes=(None, 0, (0,0,0), None))(
            None, x_raw, (mlp_weights, buf, ctr), config
        )
        
        # B. High-Frequency (Titan)
        # 1. Retrieval
        q_expanded = jnp.expand_dims(q, axis=-1)
        mem_out = jnp.matmul(M, q_expanded).squeeze(-1)
        
        # 2. Gradient of L2 Error
        k_expanded = jnp.expand_dims(k, axis=-1)
        reconstruction = jnp.matmul(M, k_expanded).squeeze(-1)
        
        error = reconstruction - v
        
        # 3. Compute Gradient Signal (Outer Product)
        error_expanded = jnp.expand_dims(error, axis=-1)
        k_transposed = jnp.expand_dims(k, axis=-2) 
        grad_signal = jnp.matmul(error_expanded, k_transposed)
        
        # 3. Deep Optimizer
        new_Mom = neural_momentum(params['deep_opt'], grad_signal, Mom, config)
        
        # 4. Update Memory
        decay_rate = jax.nn.sigmoid(params['decay'])
        new_M = (1.0 - decay_rate) * M - new_Mom
        
        new_high_carry = (new_M, new_Mom)
        
        return (new_high_carry, new_mid_carry), (mem_out, mlp_out)

    carry_init = ((init_M, init_Mom), init_mid_carry)
    
    _, (mem_out_seq, mlp_out_seq) = jax.lax.scan(
        scan_step,
        carry_init,
        (Q_T, K_T, V_T, x_T)
    )
    
    # --- 4. Fusion ---
    # Swap axes back: [Time, Batch, ...] -> [Batch, Time, ...]
    mem_out_seq = jnp.swapaxes(mem_out_seq, 0, 1)
    mlp_out_seq = jnp.swapaxes(mlp_out_seq, 0, 1)
    
    titan_out = mem_out_seq.reshape(batch_size, seq_len, -1)
    titan_out = linear(params['W_o'], titan_out)
    
    total_out = x_seq + titan_out + mlp_out_seq
    return layer_norm(params['ln2'], total_out)

# ==========================================
# 6. MODEL WRAPPER
# ==========================================

def init_model(rng, config: ModelConfig):
    k1, k2, k3 = jax.random.split(rng, 3)
    return {
        'embed': jax.random.normal(k1, (config.vocab_size, config.d_model)) * 0.02,
        'hope': init_hope(k2, config),
        'head': init_linear(k3, config.d_model, config.vocab_size)
    }

def model_forward(params, x_seq, config: ModelConfig):
    # Embedding
    x = params['embed'][x_seq] # [Batch, Time, D_model]
    
    # HOPE Block
    x = hope_forward(params['hope'], x, config)
    
    # Head
    logits = linear(params['head'], x)
    return logits

# ==========================================
# 7. TRAINING UTILS
# ==========================================

def get_toy_batch(rng, batch_size=32, seq_len=30, vocab=128):
    data = jax.random.randint(rng, (batch_size, seq_len), 0, vocab)
    k_rng, v_rng = jax.random.split(rng, 2)
    keys = jax.random.randint(k_rng, (batch_size,), 0, vocab)
    vals = jax.random.randint(v_rng, (batch_size,), 0, vocab)
    
    # Simple Association Task
    data = data.at[:, 5].set(keys)
    data = data.at[:, 6].set(vals)
    data = data.at[:, 20].set(keys) # Recall Query
    
    targets = jnp.roll(data, -1, axis=1)
    targets = targets.at[:, 20].set(vals) # Expect Value
    return data, targets

def train():
    print("Initializing Pure JAX HOPE Model...")
    config = ModelConfig()
    rng = jax.random.PRNGKey(42)
    
    # Init Params
    params = init_model(rng, config)
    
    # Optimizer (Outer Loop)
    tx = optax.adamw(learning_rate=0.001)
    opt_state = tx.init(params)
    
    @jax.jit
    def train_step(params, opt_state, batch, targets, rng):
        def loss_fn(p):
            logits = model_forward(p, batch, config)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    print("Starting Training...")
    t_rng = jax.random.PRNGKey(0)
    
    for i in range(301):
        t_rng, s_rng, st_rng = jax.random.split(t_rng, 3)
        b, t = get_toy_batch(s_rng)
        params, opt_state, loss = train_step(params, opt_state, b, t, st_rng)
        
        if i % 50 == 0:
            print(f"Step {i:4d} | Loss: {loss:.4f}")

    # Inference Test
    print("\n=== Inference Test: Key=10, Val=99 -> Query=10 ===")
    seq = jnp.zeros((1, 30), dtype=jnp.int32)
    seq = seq.at[0, 5].set(10) 
    seq = seq.at[0, 6].set(99) 
    seq = seq.at[0, 20].set(10)
    
    logits = model_forward(params, seq, config)
    pred = jnp.argmax(logits[0, 20])
    print(f"Prediction at step 20: {pred} (Expected: 99)")

if __name__ == "__main__":
    train()