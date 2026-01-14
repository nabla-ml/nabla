import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from typing import Tuple, List, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import time


# --- 1. Configuration (Unchanged) ---
@dataclass
class TransformerConfig:
    num_layers: int = 4
    hidden_size: int = 256
    num_heads: int = 4
    expansion: float = 4.0
    norm_epsilon: float = 1e-5
    rope_theta: float = 10000.0


@dataclass
class ACTConfig:
    halt_max_steps: int = 16
    halt_exploration_probability: float = 0.1


@dataclass
class HRMACTModelConfig:
    seq_len: int = 81
    vocab_size: int = 10
    high_level_cycles: int = 2
    low_level_cycles: int = 2
    transformers: TransformerConfig = field(default_factory=TransformerConfig)
    act: ACTConfig = field(default_factory=ACTConfig)
    dtype: jnp.dtype = jnp.bfloat16


# --- 2. Helper Functions & Model Layers (Unchanged) ---
def rms_norm(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)


def trunc_normal_init(
    key: jax.random.PRNGKey, shape: Tuple, std: float, dtype: jnp.dtype
) -> jnp.ndarray:
    return std * jrandom.truncated_normal(key, -2, 2, shape=shape, dtype=dtype)


class RotaryPositionEmbedding(eqx.Module):
    sin: jnp.ndarray
    cos: jnp.ndarray

    def __init__(self, dim: int, max_length: int, base: float, dtype: jnp.dtype):
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))
        t = jnp.arange(max_length, dtype=dtype)
        freqs = jnp.outer(t, inv_freq)
        self.sin, self.cos = jnp.sin(freqs), jnp.cos(freqs)

    def __call__(self, length: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.sin[:length, None, :], self.cos[:length, None, :]


def apply_rotary_emb(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    r_x1 = x1 * cos - x2 * sin
    r_x2 = x1 * sin + x2 * cos
    return jnp.concatenate([r_x1, r_x2], axis=-1)


class Attention(eqx.Module):
    num_heads: int
    head_dim: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear

    def __init__(
        self, dim: int, num_heads: int, head_dim: int, key: jax.random.PRNGKey
    ):
        keys = jrandom.split(key, 4)
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = eqx.nn.Linear(
            dim, num_heads * head_dim, use_bias=False, key=keys[0]
        )
        self.k_proj = eqx.nn.Linear(
            dim, num_heads * head_dim, use_bias=False, key=keys[1]
        )
        self.v_proj = eqx.nn.Linear(
            dim, num_heads * head_dim, use_bias=False, key=keys[2]
        )
        self.o_proj = eqx.nn.Linear(
            num_heads * head_dim, dim, use_bias=False, key=keys[3]
        )

    def __call__(self, x: jnp.ndarray, rope: Tuple) -> jnp.ndarray:
        seq_len, _ = x.shape
        q_p, k_p, v_p, o_p = map(
            jax.vmap, (self.q_proj, self.k_proj, self.v_proj, self.o_proj)
        )
        q = q_p(x).reshape(seq_len, self.num_heads, self.head_dim)
        k = k_p(x).reshape(seq_len, self.num_heads, self.head_dim)
        v = v_p(x).reshape(seq_len, self.num_heads, self.head_dim)
        sin, cos = rope
        q, k = apply_rotary_emb(q, sin, cos), apply_rotary_emb(k, sin, cos)
        attn = jnp.einsum("lhd,khd->h l k", q, k) * (self.head_dim**-0.5)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("h l k,khd->lhd", attn, v).reshape(seq_len, -1)
        return o_p(out)


class SwiGLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, dim: int, expansion: float, key: jax.random.PRNGKey):
        h = int(dim * expansion)
        keys = jrandom.split(key, 3)
        self.w1 = eqx.nn.Linear(dim, h, use_bias=False, key=keys[0])
        self.w3 = eqx.nn.Linear(dim, h, use_bias=False, key=keys[1])
        self.w2 = eqx.nn.Linear(h, dim, use_bias=False, key=keys[2])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w1, w3, w2 = map(jax.vmap, (self.w1, self.w3, self.w2))
        return w2(jax.nn.silu(w1(x)) * w3(x))


class HRMACTBlock(eqx.Module):
    self_attn: Attention
    mlp: SwiGLU
    norm_epsilon: float

    def __init__(
        self,
        h_size: int,
        n_heads: int,
        exp: float,
        norm_eps: float,
        key: jax.random.PRNGKey,
    ):
        keys = jrandom.split(key, 2)
        self.self_attn = Attention(h_size, n_heads, h_size // n_heads, keys[0])
        self.mlp = SwiGLU(h_size, exp, keys[1])
        self.norm_epsilon = norm_eps

    def __call__(self, x: jnp.ndarray, rope: Tuple) -> jnp.ndarray:
        x = rms_norm(x + self.self_attn(x, rope), self.norm_epsilon)
        x = rms_norm(x + self.mlp(x), self.norm_epsilon)
        return x


class HRMACTReasoner(eqx.Module):
    blocks: List[HRMACTBlock]

    def __init__(
        self,
        n_layers: int,
        h_size: int,
        n_heads: int,
        exp: float,
        norm_eps: float,
        key: jax.random.PRNGKey,
    ):
        keys = jrandom.split(key, n_layers)
        self.blocks = [HRMACTBlock(h_size, n_heads, exp, norm_eps, k) for k in keys]

    def __call__(
        self, h_state: jnp.ndarray, i_inject: jnp.ndarray, rope: Tuple
    ) -> jnp.ndarray:
        h_state += i_inject
        [h_state := block(h_state, rope) for block in self.blocks]
        return h_state


# --- 3. Main Model & 4. Loss Function (Unchanged) ---
class ModelOutput(NamedTuple):
    next_high_level: jnp.ndarray
    next_low_level: jnp.ndarray
    output_logits: jnp.ndarray
    q_act_halt: jnp.ndarray
    q_act_continue: jnp.ndarray


class HRMACTInner(eqx.Module):
    config: HRMACTModelConfig = eqx.field(static=True)
    cls_token: jnp.ndarray
    input_embedding: eqx.nn.Embedding
    output_head: eqx.nn.Linear
    q_act_head: eqx.nn.Linear
    rotary_emb: RotaryPositionEmbedding
    high_level_reasoner: HRMACTReasoner
    low_level_reasoner: HRMACTReasoner
    initial_high_level: jnp.ndarray
    initial_low_level: jnp.ndarray

    def __init__(self, config: HRMACTModelConfig, key: jax.random.PRNGKey):
        self.config = config
        keys = jrandom.split(key, 7)
        h_size = config.transformers.hidden_size
        std_embed = 1.0 / jnp.sqrt(h_size)
        self.cls_token = trunc_normal_init(keys[0], (h_size,), std_embed, config.dtype)
        self.input_embedding = eqx.nn.Embedding(config.vocab_size, h_size, key=keys[1])
        self.output_head = eqx.nn.Linear(
            h_size, config.vocab_size, use_bias=False, key=keys[2]
        )
        q_act_head = eqx.nn.Linear(h_size, 2, use_bias=True, key=keys[3])
        self.q_act_head = eqx.tree_at(
            lambda m: m.bias, q_act_head, jnp.array([-5.0, -5.0])
        )
        rope_dim = h_size // config.transformers.num_heads
        self.rotary_emb = RotaryPositionEmbedding(
            rope_dim, config.seq_len + 1, config.transformers.rope_theta, config.dtype
        )
        r_args = (
            config.transformers.num_layers,
            h_size,
            config.transformers.num_heads,
            config.transformers.expansion,
            config.transformers.norm_epsilon,
        )
        self.high_level_reasoner = HRMACTReasoner(*r_args, key=keys[4])
        self.low_level_reasoner = HRMACTReasoner(*r_args, key=keys[5])
        hl_key, ll_key = jrandom.split(keys[6])
        self.initial_high_level = trunc_normal_init(
            hl_key, (h_size,), 1.0, config.dtype
        )
        self.initial_low_level = trunc_normal_init(ll_key, (h_size,), 1.0, config.dtype)

    def __call__(
        self, high_level: jnp.ndarray, low_level: jnp.ndarray, inputs: jnp.ndarray
    ) -> ModelOutput:
        embeds = jnp.concatenate(
            [self.cls_token[None, :], jax.vmap(self.input_embedding)(inputs)], axis=0
        )
        embeds *= jnp.sqrt(self.config.transformers.hidden_size).astype(embeds.dtype)
        rope = self.rotary_emb(embeds.shape[0])
        hl_z, ll_z = high_level, low_level
        total_cycles = self.config.high_level_cycles * self.config.low_level_cycles
        for cycle in range(1, total_cycles):
            ll_z = self.low_level_reasoner(ll_z, hl_z + embeds, rope)
            if cycle % self.config.low_level_cycles == 0:
                hl_z = self.high_level_reasoner(hl_z, ll_z, rope)
        hl_z, ll_z = map(jax.lax.stop_gradient, (hl_z, ll_z))
        ll_z = self.low_level_reasoner(ll_z, hl_z + embeds, rope)
        hl_z = self.high_level_reasoner(hl_z, ll_z, rope)
        logits = jax.vmap(self.output_head)(hl_z[1:, :])
        q_act_logits = self.q_act_head(hl_z[0, :])
        return ModelOutput(
            next_high_level=hl_z,
            next_low_level=ll_z,
            output_logits=logits,
            q_act_halt=q_act_logits[0],
            q_act_continue=q_act_logits[1],
        )


def sudoku_loss_fn(
    model: HRMACTInner,
    hl_z: jnp.ndarray,
    ll_z: jnp.ndarray,
    boards: jnp.ndarray,
    targets: jnp.ndarray,
    segments: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, dict]:
    output = jax.vmap(model)(hl_z, ll_z, boards)
    mask = (boards == 0).astype(jnp.float32)
    output_loss = (
        optax.softmax_cross_entropy_with_integer_labels(output.output_logits, targets)
        * mask
    ).sum() / jnp.maximum(mask.sum(), 1)
    output_correct = jnp.all(
        (output.output_logits.argmax(-1) == targets) | (boards != 0), axis=-1
    )
    q_halt_target = output_correct.astype(jnp.int32)
    next_out = jax.vmap(model)(output.next_high_level, output.next_low_level, boards)
    is_last = (segments + 1) > model.config.act.halt_max_steps
    q_cont_target = jax.nn.sigmoid(
        jnp.where(
            is_last,
            next_out.q_act_halt,
            jnp.maximum(next_out.q_act_halt, next_out.q_act_continue),
        )
    )
    halt_loss = optax.sigmoid_binary_cross_entropy(output.q_act_halt, q_halt_target)
    cont_loss = optax.sigmoid_binary_cross_entropy(output.q_act_continue, q_cont_target)
    q_act_loss = (halt_loss + cont_loss).mean() / 2.0
    total_loss = output_loss + q_act_loss
    h_key, m_key = jrandom.split(key)
    is_halted = is_last | (output.q_act_halt > output.q_act_continue)
    halt_expl = (
        jrandom.uniform(h_key, is_halted.shape)
        < model.config.act.halt_exploration_probability
    )
    min_segs = jrandom.randint(
        m_key, segments.shape, 2, model.config.act.halt_max_steps + 1
    )
    is_halted &= (segments + 1) > min_segs * halt_expl.astype(jnp.int32)
    blanks_ok = ((output.output_logits.argmax(-1) == targets) * mask).sum()
    num_blanks = mask.sum()
    blanks_acc = jnp.where(num_blanks > 0, blanks_ok / num_blanks, 1.0)
    q_acc = jnp.mean((output.q_act_halt >= 0) == output_correct)
    return total_loss, {
        "is_halted": is_halted,
        "blanks_acc": blanks_acc,
        "q_act_halt_acc": q_acc,
        "next_hl_z": output.next_high_level,
        "next_ll_z": output.next_low_level,
        "total_loss": total_loss,
    }


# --- 5. Data Handling & Training Step (MODIFIED for On-the-Fly Generation) ---
@eqx.filter_jit
def generate_and_replace_halted(
    hl_z,
    ll_z,
    board_inputs,
    board_targets,
    segments,
    is_halted,
    key,
    initial_hl,
    initial_ll,
):
    """
    Generates new puzzles ONLY for the halted items and replaces them in the batch.
    This entire function runs as a single, efficient kernel on the GPU.
    """
    batch_size = is_halted.shape[0]

    # --- 1. Generate new puzzles FOR THE ENTIRE BATCH ---
    # This is simpler than trying to generate a dynamic number. We will only use the ones we need.
    REMOVE_COUNTS = jnp.array([40, 50, 55, 60])
    BASE_SOLUTION = jnp.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ],
        dtype=jnp.int32,
    )

    # vmap the generation function over a new set of keys for the batch
    @jax.vmap
    def _generate_one_sudoku(key):
        shuffle_key, diff_key, remove_key = jrandom.split(key, 3)
        perm = jrandom.permutation(shuffle_key, jnp.arange(1, 10))
        solution = perm[BASE_SOLUTION - 1].flatten()
        difficulty_index = jrandom.choice(
            diff_key, jnp.arange(4), p=jnp.array([0.25] * 4)
        )
        remove_count = REMOVE_COUNTS[difficulty_index]
        shuffled_indices = jrandom.permutation(remove_key, jnp.arange(81))
        mask_to_remove = shuffled_indices < remove_count
        puzzle = jnp.where(mask_to_remove, 0, solution)
        return puzzle, solution

    # Create a new puzzle for every slot in the batch.
    new_puzzle_keys = jrandom.split(key, batch_size)
    new_boards, new_targets = _generate_one_sudoku(new_puzzle_keys)

    # --- 2. Use `jnp.where` to select between old and new data based on the `is_halted` mask ---
    halted_mask_b = is_halted[:, None]  # (B, 1) for boards
    halted_mask_bl = is_halted[:, None, None]  # (B, 1, 1) for states

    board_inputs = jnp.where(halted_mask_b, new_boards, board_inputs)
    board_targets = jnp.where(halted_mask_b, new_targets, board_targets)

    # --- 3. Reset states and segments for the halted items ---
    hl_z = jnp.where(
        halted_mask_bl, jnp.repeat(initial_hl[None, None, :], batch_size, axis=0), hl_z
    )
    ll_z = jnp.where(
        halted_mask_bl, jnp.repeat(initial_ll[None, None, :], batch_size, axis=0), ll_z
    )
    segments = jnp.where(is_halted, 0, segments)

    return hl_z, ll_z, board_inputs, board_targets, segments


@eqx.filter_jit
def training_step(
    model: HRMACTInner,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch_tuple: Tuple,
) -> Tuple:
    (loss, aux), grads = eqx.filter_value_and_grad(sudoku_loss_fn, has_aux=True)(
        model, *batch_tuple
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, aux


# --- 6. Main Training Loop (MODIFIED) ---
def train(
    config: HRMACTModelConfig = HRMACTModelConfig(),
    steps: int = 10000,
    batch_size: int = 64,
    lr: float = 3e-4,
):
    key = jrandom.PRNGKey(42)
    model_key, batch_key, train_key = jrandom.split(key, 3)

    model = HRMACTInner(config, model_key)
    model = jax.tree_util.tree_map(
        lambda x: x.astype(config.dtype) if eqx.is_array(x) else x, model
    )
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- Initialize the first batch of data on-the-fly ---
    print("Generating initial batch of data...")
    # A dummy `is_halted` mask with all True to generate the first batch
    initial_halt_mask = jnp.ones((batch_size,), dtype=bool)
    # Dummy empty arrays to be filled
    hl_z = jnp.zeros(
        (batch_size, config.seq_len + 1, config.transformers.hidden_size),
        dtype=config.dtype,
    )
    ll_z = jnp.zeros(
        (batch_size, config.seq_len + 1, config.transformers.hidden_size),
        dtype=config.dtype,
    )
    board_inputs = jnp.zeros((batch_size, config.seq_len), dtype=jnp.int32)
    board_targets = jnp.zeros((batch_size, config.seq_len), dtype=jnp.int32)
    segments = jnp.zeros((batch_size,), dtype=jnp.int32)

    hl_z, ll_z, board_inputs, board_targets, segments = generate_and_replace_halted(
        hl_z,
        ll_z,
        board_inputs,
        board_targets,
        segments,
        initial_halt_mask,
        batch_key,
        model.initial_high_level,
        model.initial_low_level,
    )

    print("\nStarting training...")
    last_log_time = time.time()
    steps_since_log = 0
    compilation_done = False
    for step_num in range(1, steps + 1):
        if not compilation_done:
            print("First step: Compiling and running (this may take a minute)...")

        step_key, replace_key = jrandom.split(train_key)
        train_key = step_key

        batch_tuple = (hl_z, ll_z, board_inputs, board_targets, segments, step_key)
        model, opt_state, aux = training_step(model, opt_state, optimizer, batch_tuple)

        if not compilation_done:
            print(
                f"Compilation finished. Total time for first step: {time.time() - last_log_time:.2f}s"
            )
            last_log_time = time.time()
            compilation_done = True

        hl_z, ll_z = aux["next_hl_z"], aux["next_ll_z"]
        is_halted = aux["is_halted"]
        segments = segments + 1

        # This is now a single, JIT-ed call that performs the entire replacement on the GPU
        hl_z, ll_z, board_inputs, board_targets, segments = generate_and_replace_halted(
            hl_z,
            ll_z,
            board_inputs,
            board_targets,
            segments,
            is_halted,
            replace_key,
            model.initial_high_level,
            model.initial_low_level,
        )

        steps_since_log += 1
        if step_num % 1 == 0:
            end_time = time.time()
            steps_per_sec = steps_since_log / (end_time - last_log_time)
            last_log_time, steps_since_log = end_time, 0
            # Note: A proper puzzle counter is complex inside JIT, so we omit it for this performance-focused version
            print(
                f"Step {step_num:5d} | Loss: {aux['total_loss'].item():.3f} | Blanks Acc: {aux['blanks_acc'].item():.3f} | Q-ACT Acc: {aux['q_act_halt_acc'].item():.3f} | SPS: {steps_per_sec:.2f}"
            )


if __name__ == "__main__":
    try:
        print("Available JAX devices:", jax.devices())
        device = jax.devices()[0]
        print(f"Using device: {device.device_kind}")
        if "A100" in device.device_kind:
            print("Successfully running on an A100 GPU.")
        else:
            print("Warning: Not running on an A100 GPU.")
    except Exception as e:
        print("Could not verify JAX device:", e)
    train(batch_size=64)
