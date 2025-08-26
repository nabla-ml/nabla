# --- IMPORTS AND MODEL DEFINITIONS (Same as before) ---
import jax
import jax.numpy as jnp
import optax
from functools import partial
import numpy as np
import time

# --- HYPERPARAMETERS ---
SEQUENCE_LENGTH = 14
HIDDEN_DIM = 256
NUM_HEADS = 4
D_FF = 4 * HIDDEN_DIM
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

HIGH_LEVEL_CYCLES = 10
LOW_LEVEL_CYCLES = 3


# --- (All model definition functions remain the same) ---
def generate_counting_comparison_batch(batch_size, seq_len):
    assert seq_len % 2 == 0, "Sequence length must be even."
    x = np.random.randint(0, 2, size=(batch_size, seq_len, 1))
    half = seq_len // 2
    first_half_sum = np.sum(x[:, :half, :], axis=1)
    second_half_sum = np.sum(x[:, half:, :], axis=1)
    y = (first_half_sum > second_half_sum).astype(np.int32)
    return jnp.array(x), jnp.array(y)


def init_dense(key, in_dim, out_dim):
    limit = jnp.sqrt(6 / (in_dim + out_dim))
    w = jax.random.uniform(key, shape=(in_dim, out_dim), minval=-limit, maxval=limit)
    return {"weights": w}


def dense_forward(params, x):
    return jnp.dot(x, params["weights"])


def init_rms(key, dim):
    return {"weight": jnp.ones((dim,))}


def rms_norm(params, x, eps=1e-5):
    var = jnp.mean(x**2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * params["weight"]


def precompute_rope(dim, seq_len, base=10000.0):
    theta = base ** (-2.0 * jnp.arange(0, dim // 2) / dim)
    m = jnp.arange(seq_len)
    angles = m[:, None] * theta[None, :]
    sin, cos = jnp.sin(angles), jnp.cos(angles)
    return jnp.expand_dims(sin, (0, 2)), jnp.expand_dims(cos, (0, 2))


def apply_rope(x, sin, cos):
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    sin_b, cos_b = (
        jnp.concatenate([sin, sin], axis=-1),
        jnp.concatenate([cos, cos], axis=-1),
    )
    return x * cos_b + rotated * sin_b


def init_self_attn(key, dim, num_heads):
    keys = jax.random.split(key, 4)
    return {
        "q_proj": init_dense(keys[0], dim, dim),
        "k_proj": init_dense(keys[1], dim, dim),
        "v_proj": init_dense(keys[2], dim, dim),
        "o_proj": init_dense(keys[3], dim, dim),
    }


def self_attention(params, x, sin, cos):
    batch_size, seq_len, dim = x.shape
    d_head = dim // NUM_HEADS
    q = dense_forward(params["q_proj"], x).reshape(
        batch_size, seq_len, NUM_HEADS, d_head
    )
    k = dense_forward(params["k_proj"], x).reshape(
        batch_size, seq_len, NUM_HEADS, d_head
    )
    v = dense_forward(params["v_proj"], x).reshape(
        batch_size, seq_len, NUM_HEADS, d_head
    )
    q, k = apply_rope(q, sin, cos), apply_rope(k, sin, cos)
    logits = jnp.einsum("bshd,bthd->bsht", q, k) / jnp.sqrt(d_head)
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bsht,bthd->bshd", attn, v).reshape(batch_size, seq_len, dim)
    return dense_forward(params["o_proj"], out)


def init_ff(key, dim, d_ff):
    keys = jax.random.split(key, 3)
    return {
        "w1": init_dense(keys[0], dim, d_ff),
        "w2": init_dense(keys[1], dim, d_ff),
        "w3": init_dense(keys[2], d_ff, dim),
    }


def glu_ff(params, x):
    gate = jax.nn.gelu(dense_forward(params["w1"], x))
    val = dense_forward(params["w2"], x)
    return dense_forward(params["w3"], gate * val)


def init_transformer_block(key, dim, num_heads, d_ff):
    keys = jax.random.split(key, 4)
    return {
        "norm1": init_rms(keys[0], dim),
        "attn": init_self_attn(keys[1], dim, num_heads),
        "norm2": init_rms(keys[2], dim),
        "ff": init_ff(keys[3], dim, d_ff),
    }


def transformer_block(params, x, sin, cos):
    attn = self_attention(params["attn"], rms_norm(params["norm1"], x), sin, cos)
    x = x + attn
    ff = glu_ff(params["ff"], rms_norm(params["norm2"], x))
    return x + ff


def init_hrm_params(key, hidden_dim):
    keys = jax.random.split(key, 4)
    return {
        "input_net": init_dense(keys[0], 1, hidden_dim),
        "L_cell": init_transformer_block(keys[1], hidden_dim, NUM_HEADS, D_FF),
        "H_cell": init_transformer_block(keys[2], hidden_dim, NUM_HEADS, D_FF),
        "output_net": init_dense(keys[3], hidden_dim, 1),
    }


# --- Reasoning Phase with explicit `for` loops for clarity ---
def reasoning_phase(params, x_tilde, sin, cos):
    # Initialize the states we will update in the loops
    zH = jnp.zeros_like(x_tilde)
    zL = jnp.zeros_like(x_tilde)

    # This is the OUTER loop for the high-level "Manager"
    for _ in range(HIGH_LEVEL_CYCLES - 1):
        # This is the INNER loop for the low-level "Worker"
        # The worker will update its state `LOW_LEVEL_CYCLES` times
        # while the manager's state (zH) is held constant.
        zL_inner_carry = zL
        for _ in range(LOW_LEVEL_CYCLES):
            l_input = zL_inner_carry + zH + x_tilde
            zL_inner_carry = transformer_block(params["L_cell"], l_input, sin, cos)

        # After the inner loop, this is the worker's refined state
        zL_refined = zL_inner_carry

        # Now, the manager updates its state based on the worker's refined output
        h_input = zH + zL_refined
        zH_new = transformer_block(params["H_cell"], h_input, sin, cos)

        # Update the states for the next iteration of the outer loop
        zH = zH_new
        zL = zL_refined

    return zH, zL


# --- The Learning Phase (This was already correct and simple) ---
def learning_phase_loss(params, x_tilde, zH_warmed, zL_warmed, sin, cos, y_batch):
    # ONE L-cell update
    l_input_final = zL_warmed + zH_warmed + x_tilde
    zL_final = transformer_block(params["L_cell"], l_input_final, sin, cos)
    # ONE H-cell update
    h_input_final = zH_warmed + zL_final
    zH_final = transformer_block(params["H_cell"], h_input_final, sin, cos)
    # Final loss calculation
    pooled = jnp.mean(zH_final, axis=1)
    logits = dense_forward(params["output_net"], pooled).squeeze()
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y_batch.squeeze()))


# --- MAIN EXECUTION SCRIPT (Identical to before) ---
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    )
    key, hrm_key = jax.random.split(key)
    hrm_params = init_hrm_params(hrm_key, HIDDEN_DIM)
    hrm_opt_state = optimizer.init(hrm_params)

    NUM_ITERATIONS = 6
    reasoning_times, grad_times, optimizer_times = [], [], []
    grad_fn = jax.value_and_grad(learning_phase_loss)

    total_L_steps_reason = (HIGH_LEVEL_CYCLES - 1) * LOW_LEVEL_CYCLES
    total_H_steps_reason = HIGH_LEVEL_CYCLES - 1

    print(f"--- Profiling Eager Execution (with explicit FOR LOOPS) ---")
    print(
        f"Reasoning Phase Workload: {total_H_steps_reason} H-steps, {total_L_steps_reason} L-steps"
    )
    print(f"Learning Phase Workload:  1 H-step, 1 L-step")
    print(f"Running {NUM_ITERATIONS} iterations (first is warmup)...")

    for i in range(NUM_ITERATIONS):
        x_batch, y_batch = generate_counting_comparison_batch(
            BATCH_SIZE, SEQUENCE_LENGTH
        )
        sin, cos = precompute_rope(HIDDEN_DIM // NUM_HEADS, SEQUENCE_LENGTH)
        x_tilde = jax.vmap(dense_forward, in_axes=(None, 0))(
            hrm_params["input_net"], x_batch
        )

        t0 = time.perf_counter()
        zH_warmed, zL_warmed = reasoning_phase(hrm_params, x_tilde, sin, cos)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), (zH_warmed, zL_warmed))
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        loss_val, grads = grad_fn(
            hrm_params, x_tilde, zH_warmed, zL_warmed, sin, cos, y_batch
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), (loss_val, grads))
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        updates, new_opt_state = optimizer.update(grads, hrm_opt_state, hrm_params)
        new_params = optax.apply_updates(hrm_params, updates)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), new_params)
        t5 = time.perf_counter()

        reasoning_times.append(t1 - t0)
        grad_times.append(t3 - t2)
        optimizer_times.append(t5 - t4)
        hrm_params = new_params
        print(f"  Iteration {i + 1} done.")

    avg_reasoning = np.mean(reasoning_times[1:])
    avg_grad = np.mean(grad_times[1:])
    avg_optimizer = np.mean(optimizer_times[1:])
    total_avg = avg_reasoning + avg_grad + avg_optimizer

    print("\n" + "=" * 55)
    print("--- Average Eager Timings (post-warmup) ---")
    print(
        f"1. Reasoning Phase  : {avg_reasoning:.4f}s  ({(avg_reasoning / total_avg) * 100:.1f}%)"
    )
    print(
        f"2. Grad Calculation : {avg_grad:.4f}s  ({(avg_grad / total_avg) * 100:.1f}%)"
    )
    print(
        f"3. Optimizer Update : {avg_optimizer:.4f}s  ({(avg_optimizer / total_avg) * 100:.1f}%)"
    )
    print("-" * 55)
    print(f"Total Average Step Time : {total_avg:.4f}s")
    print("=" * 55)
