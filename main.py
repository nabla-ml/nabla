import numpy as np
import jax
import jax.numpy as jnp
import optax

# =========================
# Hyperparameters
# =========================
SEED = 0
np.random.seed(SEED)

SEQ_LEN = 32
HIDDEN = 64
HEADS = 4
D_FF = 4 * HIDDEN
BATCH = 64

LR = 3e-4
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0

HIGH_CYCLES = 3
LOW_CYCLES = 2

M_SLOTS_L = 4
M_SLOTS_H = 2

AUX_LOSS_W = 0.25

RESIDUAL_SCALE = 0.5
DTYPE = jnp.float32

# =========================
# Data: adding problem
# =========================
def make_batch(n, seq_len):
    values = np.random.rand(n, seq_len).astype(np.float32)
    markers = np.zeros((n, seq_len), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        p1, p2 = np.random.choice(seq_len, 2, replace=False)
        markers[i, p1] = 1.0
        markers[i, p2] = 1.0
        y[i, 0] = values[i, p1] + values[i, p2]
    x = np.stack([values, markers], axis=-1)
    return jnp.array(x, DTYPE), jnp.array(y, DTYPE)

# =========================
# NN utils
# =========================
def tree_l2(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(jnp.asarray(x) ** 2) for x in leaves]))

def init_dense(k, i, o, scale=1.0):
    lim = scale * jnp.sqrt(6.0 / (i + o))
    w = jax.random.uniform(k, (i, o), minval=-lim, maxval=lim, dtype=DTYPE)
    b = jnp.zeros((o,), DTYPE)
    return {"w": w, "b": b}

def linear(p, x):
    return x @ p["w"] + p["b"]

def init_rms(k, d):
    return {"g": jnp.ones((d,), DTYPE)}

def rms_norm(p, x, eps=1e-6):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    xhat = x * jax.lax.rsqrt(var + eps)
    return xhat * p["g"][None, None, :]

def init_ff(k, d, dff):
    k1, k2, k3 = jax.random.split(k, 3)
    return {
        "w1": init_dense(k1, d, dff, scale=0.5),
        "w2": init_dense(k2, d, dff, scale=0.5),
        "w3": init_dense(k3, dff, d, scale=0.5),
    }

def glu_ff(p, x):
    g = jax.nn.gelu(linear(p["w1"], x))
    v = linear(p["w2"], x)
    return linear(p["w3"], g * v)

def init_mha(k, d):
    kq, kk, kv, ko = jax.random.split(k, 4)
    return {
        "q": init_dense(kq, d, d),
        "k": init_dense(kk, d, d),
        "v": init_dense(kv, d, d),
        "o": init_dense(ko, d, d, scale=0.5),
    }

def split_heads(x, heads):
    b, s, d = x.shape
    dh = d // heads
    return x.reshape(b, s, heads, dh)

def merge_heads(x):
    b, s, h, dh = x.shape
    return x.reshape(b, s, h * dh)

def mha(p, x_q, x_kv, heads):
    q = split_heads(linear(p["q"], x_q), heads)
    k = split_heads(linear(p["k"], x_kv), heads)
    v = split_heads(linear(p["v"], x_kv), heads)
    dh = q.shape[-1]
    att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(dh)
    att = jax.nn.softmax(att, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", att, v)
    out = merge_heads(out)
    return linear(p["o"], out)

def init_block(k, d, dff):
    k1, k2, k3 = jax.random.split(k, 3)
    return {
        "norm1": init_rms(k1, d),
        "attn": init_mha(k2, d),
        "norm2": init_rms(k1, d),
        "ff": init_ff(k3, d, dff),
    }

def block_forward(p, x, heads):
    a = mha(p["attn"], rms_norm(p["norm1"], x), x, heads)
    x = x + RESIDUAL_SCALE * a
    f = glu_ff(p["ff"], rms_norm(p["norm2"], x))
    x = x + RESIDUAL_SCALE * f
    return x

# =========================
# Memory modules (HRM core)
# =========================
def init_memory_params(k, d):
    kr, kw, kg = jax.random.split(k, 3)
    return {
        "read_attn": init_mha(kr, d),
        "write_attn": init_mha(kw, d),
        "gate_w": init_dense(kg, 2 * d, d, scale=0.25),
        "gate_e": init_dense(kg, 2 * d, d, scale=0.25),
        "norm_seq": init_rms(kg, d),
        "norm_mem": init_rms(kg, d),
    }

def memory_read(p, seq, mem, heads):
    seq_n = rms_norm(p["norm_seq"], seq)
    mem_n = rms_norm(p["norm_mem"], mem)
    read = mha(p["read_attn"], seq_n, mem_n, heads)
    return seq + RESIDUAL_SCALE * read

def memory_write(p, seq, mem, heads):
    seq_n = rms_norm(p["norm_seq"], seq)
    mem_n = rms_norm(p["norm_mem"], mem)
    cand = mha(p["write_attn"], mem_n, seq_n, heads)
    cat = jnp.concatenate([mem_n, cand], axis=-1)
    gw = jax.nn.sigmoid(linear(p["gate_w"], cat))
    ge = jax.nn.sigmoid(linear(p["gate_e"], cat))
    mem_new = (1.0 - ge) * mem + gw * cand
    return mem_new

# =========================
# HRM cell (low / high)
# =========================
def init_hrm(k, d, dff):
    kL, kH, kML, kMH, kBL, kBH, kCLS, kIN, kAUX, kOUT = jax.random.split(k, 10)
    return {
        "low_block": init_block(kBL, d, dff),
        "high_block": init_block(kBH, d, dff),
        "mem_low": init_memory_params(kML, d),
        "mem_high": init_memory_params(kMH, d),
        "cls": jax.random.normal(kCLS, (d,), DTYPE) * 0.02,
        "input_net": init_dense(kIN, 2, d, scale=0.5),
        "aux_head": init_dense(kAUX, d, 1, scale=0.25),
        "out_head": init_dense(kOUT, d, 1, scale=0.25),
    }

def prepend_cls(p, x):
    B = x.shape[0]
    cls = jnp.broadcast_to(p["cls"], (B, 1, x.shape[-1]))
    return jnp.concatenate([cls, x], axis=1)

def init_mem_states(B, d):
    mL = jnp.zeros((B, M_SLOTS_L, d), DTYPE)
    mH = jnp.zeros((B, M_SLOTS_H, d), DTYPE)
    return mL, mH

def hrm_forward(params, x):
    B, S, _ = x.shape
    h = jax.vmap(linear, in_axes=(None, 0))(params["input_net"], x)
    h = prepend_cls(params, h)
    mL, mH = init_mem_states(B, HIDDEN)

    for _ in range(HIGH_CYCLES):
        for _ in range(LOW_CYCLES):
            h = block_forward(params["low_block"], h, HEADS)
            h = memory_read(params["mem_low"], h, mL, HEADS)
            mL = memory_write(params["mem_low"], h, mL, HEADS)

        h = block_forward(params["high_block"], h, HEADS)
        h = memory_read(params["mem_high"], h, mH, HEADS)
        mH = memory_write(params["mem_high"], h, mH, HEADS)

    cls_vec = h[:, 0, :]
    y_hat = linear(params["out_head"], cls_vec)
    tok = h[:, 1:, :]
    aux_logits = linear(params["aux_head"], tok)
    return y_hat, aux_logits

# =========================
# Loss
# =========================
def compute_loss(params, x, y):
    values = x[:, :, 0:1]
    markers = x[:, :, 1:2]
    target_token = values * markers
    y_hat, aux_logits = hrm_forward(params, x)
    main = jnp.mean((y_hat - y) ** 2)
    aux = jnp.mean((aux_logits - target_token) ** 2)
    return main + AUX_LOSS_W * aux, (main, aux, y_hat)

# =========================
# Init and training
# =========================
def init_params(key):
    return init_hrm(key, HIDDEN, D_FF)

def train_step(params, opt_state, x, y, optimizer):
    (loss, (main_loss, aux_loss, pred)), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    gnorm = tree_l2(grads)
    pnorm = tree_l2(params)
    return params, opt_state, loss, main_loss, aux_loss, gnorm, pnorm, pred

def main():
    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(CLIP_NORM),
        optax.adamw(LR, weight_decay=WEIGHT_DECAY),
    )
    opt_state = optimizer.init(params)

    print("--- HRM training (gated hierarchical memory, CLS, aux) ---")
    for it in range(1, 201):
        x, y = make_batch(BATCH, SEQ_LEN)
        params, opt_state, loss, main_loss, aux_loss, gnorm, pnorm, pred = train_step(params, opt_state, x, y, optimizer)

        if it % 10 == 0:
            print(
                f"Iter {it:3d} | loss {float(loss):.6e} | main {float(main_loss):.6e} | "
                f"aux {float(aux_loss):.6e} | gnorm {float(gnorm):.3e} | pnorm {float(pnorm):.3e}"
            )

    print("Done.")

if __name__ == "__main__":
    main()
