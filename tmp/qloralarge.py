import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
from typing import Dict, Tuple, NamedTuple, Generator, List, Union

# --- 1. CONFIGURATION: OFFICIAL GEMMA 3 270M ---

# Architectural Constants from Model Card
VOCAB_SIZE: int = 262144
EMBED_DIM: int = 640                  # hidden_size
NUM_LAYERS: int = 18                  # num_hidden_layers
NUM_HEADS: int = 4                    # num_attention_heads
NUM_KV_HEADS: int = 1                 # num_key_value_heads (Multi-Query Attention)
HEAD_DIM: int = 256                   # head_dim
FFN_DIM: int = 2048                   # intermediate_size
MAX_CONTEXT_LEN: int = 32768          # max_position_embeddings

# --- Architectural Parameters ---
HIDDEN_ACTIVATION: str = "gelu_pytorch_tanh" # hidden_activation
INITIALIZER_RANGE: float = 0.02              # initializer_range
QUERY_PRE_ATTN_SCALAR: float = 256.0         # query_pre_attn_scalar
ROPE_THETA: float = 1000000.0                # rope_theta
ROPE_LOCAL_BASE_FREQ: float = 10000.0        # rope_local_base_freq


# Sliding Window Attention Configuration from Model Card
SLIDING_WINDOW_SIZE: int = 512
LAYER_TYPES: List[str] = [
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "full_attention"
]

# --- Simulation & Training Constants (Unchanged) ---
SEQ_LEN: int = 32 # Reduced for local simulation
COMPUTE_DTYPE = jnp.bfloat16
LORA_RANK: int = 8
QLORA_BLOCK_SIZE: int = 64
LORA_ALPHA: float = 16.0
BATCH_SIZE: int = 4
NUM_EPOCHS: int = 5
STEPS_PER_EPOCH: int = 1
WEIGHT_DECAY: float = 0.01
MAX_LEARNING_RATE: float = 1e-4
MIN_LEARNING_RATE: float = 1e-5
WARMUP_STEPS: int = 1
TOTAL_TRAIN_STEPS: int = NUM_EPOCHS * STEPS_PER_EPOCH
GRAD_CLIP_NORM: float = 1.0
NUM_TOKENS_TO_GENERATE: int = 8


# --- 2. QLoRA & STATIC CONFIGURATIONS ---
# This module implements the core logic for QLoRA and defines static
# configuration structures for JIT compilation. No architectural changes here.

# Type definitions
JaxArray = jax.Array
FrozenParams = Dict[str, Union[JaxArray, Dict]]
LoRAParams = Dict[str, Union[JaxArray, Dict]]
OptimizerState = Dict[str, Union[int, Dict]]

class QuantConfig(NamedTuple):
    original_shape: Tuple[int, ...]
    original_numel: int
    block_size: int
    padded_size: int

class AttentionConfig(NamedTuple):
    q_proj: QuantConfig
    k_proj: QuantConfig
    v_proj: QuantConfig
    o_proj: QuantConfig
    attention_type: str

class FeedForwardConfig(NamedTuple):
    gate_proj: QuantConfig
    up_proj: QuantConfig
    down_proj: QuantConfig

class LayerConfig(NamedTuple):
    attention: AttentionConfig
    feed_forward: FeedForwardConfig

class StaticModelConfig(NamedTuple):
    layers: Tuple[LayerConfig, ...]

NF4_QUANTIZATION_VALUES: JaxArray = jnp.array([
    -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
    0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000
], dtype=jnp.bfloat16)

# (Quantization and Dequantization functions remain unchanged)
def pack_4bit(x: JaxArray) -> JaxArray:
    x_flat = x.flatten()
    if x_flat.size % 2 != 0: x_flat = jnp.pad(x_flat, (0, 1))
    x_pairs = x_flat.reshape(-1, 2)
    return (x_pairs[:, 0] | (x_pairs[:, 1] << 4)).astype(jnp.uint8)

def unpack_4bit(x_packed: JaxArray, original_size: int) -> JaxArray:
    unpacked_pairs = jnp.stack([x_packed & 0x0F, x_packed >> 4], axis=-1)
    unpacked_flat = unpacked_pairs.flatten()
    return unpacked_flat[:original_size]

def quantize_scales(scales: JaxArray) -> Dict[str, JaxArray]:
    absmax = jnp.max(jnp.abs(scales))
    normalized_scales = scales / absmax
    q_scales = jnp.round(normalized_scales * 127).astype(jnp.int8)
    return {'q_scales': q_scales, 'meta_scale': absmax}

def dequantize_scales(q_scales_dict: Dict[str, JaxArray]) -> JaxArray:
    q_scales = q_scales_dict['q_scales'].astype(jnp.float32)
    meta_scale = q_scales_dict['meta_scale']
    return (q_scales / 127.0) * meta_scale

def quantize_nf4(w: JaxArray, block_size: int = QLORA_BLOCK_SIZE) -> Tuple[Dict[str, JaxArray], QuantConfig]:
    original_shape, w_flat = w.shape, w.flatten()
    original_numel = w_flat.size
    pad_len = (block_size - (original_numel % block_size)) % block_size
    if pad_len > 0: w_flat = jnp.pad(w_flat, (0, pad_len))
    w_blocks = w_flat.reshape(-1, block_size)
    scales = jnp.max(jnp.abs(w_blocks), axis=-1, keepdims=True)
    scales = scales.at[scales == 0].set(1.0)
    w_normalized = w_blocks / scales
    diffs = jnp.abs(w_normalized.reshape(-1, 1) - NF4_QUANTIZATION_VALUES)
    quantized_indices = jnp.argmin(diffs, axis=-1).astype(jnp.uint8)
    packed_w = pack_4bit(quantized_indices)
    quantized_scales_dict = quantize_scales(scales.flatten())
    weights = {'qw': packed_w, 'scales_quantized': quantized_scales_dict}
    config = QuantConfig(original_shape, original_numel, block_size, w_flat.size)
    return weights, config

def dequantize_nf4(q_weights: Dict[str, JaxArray], q_config: QuantConfig) -> JaxArray:
    scales = dequantize_scales(q_weights['scales_quantized'])
    indices = unpack_4bit(q_weights['qw'], q_config.padded_size)
    dequantized_flat_normalized = NF4_QUANTIZATION_VALUES[indices]
    dequantized_blocks = dequantized_flat_normalized.reshape(-1, q_config.block_size)
    scaled_blocks = dequantized_blocks * scales.reshape(-1, 1)
    w_flat = scaled_blocks.flatten()
    w = w_flat[:q_config.original_numel]
    return w.reshape(q_config.original_shape)

# --- 3. MODEL ARCHITECTURE (CORRECTED) ---
# This module is updated with the corrected activation, RoPE base frequency,
# and query scaling to match the model card.

def rms_norm(x: JaxArray, w: JaxArray, eps: float = 1e-6) -> JaxArray:
    norm_factor = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return x * norm_factor * w

def precompute_rope_freqs(
    dim: int,
    max_len: int,
    theta_global: float = ROPE_THETA,
    theta_local: float = ROPE_LOCAL_BASE_FREQ
) -> Dict[str, Tuple[JaxArray, JaxArray]]:
    """
    CORRECTED: Pre-computes two sets of RoPE frequencies.
    - 'global': For full attention layers, using the primary `rope_theta`.
    - 'local': For sliding attention layers, using `rope_local_base_freq`.
    """
    def _compute_freqs(theta: float) -> Tuple[JaxArray, JaxArray]:
        inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        return jnp.sin(freqs), jnp.cos(freqs)

    sin_global, cos_global = _compute_freqs(theta_global)
    sin_local, cos_local = _compute_freqs(theta_local)

    return {
        'global': (sin_global, cos_global),
        'local': (sin_local, cos_local)
    }

def apply_rotary_embeddings(x: JaxArray, start_pos: Union[int, JaxArray], rope_sin: JaxArray, rope_cos: JaxArray) -> JaxArray:
    seq_len, head_dim = x.shape[-2:]
    is_batched = jnp.ndim(start_pos) > 0
    if is_batched:
        def get_freqs(start: int):
            sin = jax.lax.dynamic_slice(rope_sin, [start, 0], [seq_len, head_dim // 2])
            cos = jax.lax.dynamic_slice(rope_cos, [start, 0], [seq_len, head_dim // 2])
            return sin, cos
        sin, cos = jax.vmap(get_freqs)(start_pos)
        sin, cos = sin[:, None, :, :], cos[:, None, :, :]
    else:
        sin = jax.lax.dynamic_slice(rope_sin, [start_pos, 0], [seq_len, head_dim // 2])
        cos = jax.lax.dynamic_slice(rope_cos, [start_pos, 0], [seq_len, head_dim // 2])
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
    x_real, x_imag = x[..., 0::2], x[..., 1::2]
    x_rotated_real = x_real * cos - x_imag * sin
    x_rotated_imag = x_real * sin + x_imag * cos
    y = jnp.empty_like(x)
    y = y.at[..., 0::2].set(x_rotated_real)
    y = y.at[..., 1::2].set(x_rotated_imag)
    return y

def qlora_linear(x: JaxArray, frozen_q_w: Dict, q_config: QuantConfig, lora_a: JaxArray, lora_b: JaxArray) -> JaxArray:
    dequantized_w = dequantize_nf4(frozen_q_w, q_config)
    x_compute = x.astype(COMPUTE_DTYPE)
    dequantized_w_compute = dequantized_w.astype(COMPUTE_DTYPE)
    lora_a_compute = lora_a.astype(COMPUTE_DTYPE)
    lora_b_compute = lora_b.astype(COMPUTE_DTYPE)
    lora_scaling = LORA_ALPHA / LORA_RANK
    frozen_out = x_compute @ dequantized_w_compute
    lora_out = (x_compute @ lora_a_compute) @ lora_b_compute
    return (frozen_out + lora_out * lora_scaling).astype(jnp.float32)

def attention(x: JaxArray, rope_sin: JaxArray, rope_cos: JaxArray, start_pos: Union[int, JaxArray],
              frozen_p: FrozenParams, config_p: AttentionConfig, lora_p: LoRAParams) -> JaxArray:
    """
    Uses `query_pre_attn_scalar` for score normalization.
    """
    bsz, seq_len, _ = x.shape
    q = qlora_linear(x, frozen_p['q_proj'], config_p.q_proj, lora_p['q_proj_A'], lora_p['q_proj_B'])
    k = qlora_linear(x, frozen_p['k_proj'], config_p.k_proj, lora_p['k_proj_A'], lora_p['k_proj_B'])
    v = qlora_linear(x, frozen_p['v_proj'], config_p.v_proj, lora_p['v_proj_A'], lora_p['v_proj_B'])

    q = q.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    q = apply_rotary_embeddings(q, start_pos, rope_sin, rope_cos)
    k = apply_rotary_embeddings(k, start_pos, rope_sin, rope_cos)

    n_rep = NUM_HEADS // NUM_KV_HEADS
    k_repeated = jnp.repeat(k, n_rep, axis=1)
    v_repeated = jnp.repeat(v, n_rep, axis=1)

    if config_p.attention_type == 'full_attention':
        mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=bool))
    else: # 'sliding_attention'
        q_pos = jnp.arange(seq_len)[None, :]
        k_pos = jnp.arange(seq_len)[:, None]
        causal_mask = k_pos <= q_pos
        window_mask = (q_pos - k_pos) < SLIDING_WINDOW_SIZE
        mask = causal_mask & window_mask
        mask = mask[None, None, :, :]

    scores = (q @ k_repeated.transpose(0, 1, 3, 2)) / jnp.sqrt(QUERY_PRE_ATTN_SCALAR)
    scores = jnp.where(mask, scores, -jnp.inf)
    scores = jax.nn.softmax(scores, axis=-1)

    attn_out = (scores @ v_repeated).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
    return qlora_linear(attn_out, frozen_p['o_proj'], config_p.o_proj, lora_p['o_proj_A'], lora_p['o_proj_B'])

def feed_forward(x: JaxArray, frozen_p: FrozenParams, config_p: FeedForwardConfig, lora_p: LoRAParams) -> JaxArray:
    """
    Uses the specified `gelu_pytorch_tanh` activation function.
    """
    gate = qlora_linear(x, frozen_p['gate_proj'], config_p.gate_proj, lora_p['gate_proj_A'], lora_p['gate_proj_B'])
    up = qlora_linear(x, frozen_p['up_proj'], config_p.up_proj, lora_p['up_proj_A'], lora_p['up_proj_B'])
    ffn_out = jax.nn.gelu(gate, approximate=True) * up
    return qlora_linear(ffn_out, frozen_p['down_proj'], config_p.down_proj, lora_p['down_proj_A'], lora_p['down_proj_B'])

def transformer_layer(x: JaxArray, rope_sin: JaxArray, rope_cos: JaxArray, start_pos: Union[int, JaxArray],
                      frozen_p: FrozenParams, config_p: LayerConfig, lora_p: LoRAParams) -> JaxArray:
    h = x + attention(rms_norm(x, frozen_p['attn_norm_scale']), rope_sin, rope_cos, start_pos,
                      frozen_p['attention'], config_p.attention, lora_p['attention'])
    out = h + feed_forward(rms_norm(h, frozen_p['ffn_norm_scale']),
                           frozen_p['feed_forward'], config_p.feed_forward, lora_p['feed_forward'])
    return out

def model_forward(tokens: JaxArray, start_pos: JaxArray, frozen_params: FrozenParams,
                  model_config: StaticModelConfig, lora_params: LoRAParams) -> JaxArray:
    x = frozen_params['tok_embeddings'][tokens]
    rope_freqs_map = frozen_params['rope_freqs']

    for i in range(NUM_LAYERS):
        # Select the correct RoPE frequencies (local or global) for the current layer
        # based on its specified attention type.
        if LAYER_TYPES[i] == 'full_attention':
            rope_sin, rope_cos = rope_freqs_map['global']
        else: # sliding_attention
            rope_sin, rope_cos = rope_freqs_map['local']

        x = transformer_layer(x, rope_sin, rope_cos, start_pos, frozen_params[f'layer_{i}'],
                              model_config.layers[i], lora_params[f'layer_{i}'])

    x = rms_norm(x, frozen_params['output_norm_scale'])
    return x @ frozen_params['tok_embeddings'].T


# --- 4. DATA & INITIALIZATION (CORRECTED) ---
# Initialization now respects the `initializer_range` from the model card.

def data_generator(data_size: int, batch_size: int, seq_len: int, vocab_size: int) -> Generator[Tuple[JaxArray, JaxArray, JaxArray], None, None]:
    data = np.random.randint(0, vocab_size, size=data_size, dtype=np.int32)
    while True:
        idxs = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
        x = np.stack([data[i:i+seq_len] for i in idxs])
        y = np.stack([data[i+1:i+seq_len+1] for i in idxs])
        yield jnp.array(x), jnp.array(y), jnp.array(idxs)

def init_gemma3_270m_params(key: JaxArray, vocab_size: int) -> Tuple[FrozenParams, StaticModelConfig, LoRAParams]:
    """
    CORRECTED: Initializes weights using the specified `initializer_range` and
    pre-computes both sets of RoPE frequencies.
    """
    def normal(key: JaxArray, shape: Tuple[int, ...]) -> JaxArray:
        return jax.random.normal(key, shape, dtype=jnp.float32) * INITIALIZER_RANGE
        
    key, tok_key = jax.random.split(key)
    frozen_params: FrozenParams = {
        'tok_embeddings': normal(tok_key, (vocab_size, EMBED_DIM)),
        'output_norm_scale': jnp.ones(EMBED_DIM, dtype=jnp.float32),
        'rope_freqs': precompute_rope_freqs(HEAD_DIM, MAX_CONTEXT_LEN)
    }
    lora_params: LoRAParams = {}
    layer_configs: List[LayerConfig] = []
    layer_keys = jax.random.split(key, NUM_LAYERS)
    q_proj_dim, kv_proj_dim = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM

    for i in range(NUM_LAYERS):
        layer_key = layer_keys[i]
        q_key, k_key, v_key, o_key, g_key, u_key, d_key, lora_key = jax.random.split(layer_key, 8)
        weights_to_quantize = {
            'attention': {
                'q_proj': normal(q_key, (EMBED_DIM, q_proj_dim)), 'k_proj': normal(k_key, (EMBED_DIM, kv_proj_dim)),
                'v_proj': normal(v_key, (EMBED_DIM, kv_proj_dim)), 'o_proj': normal(o_key, (q_proj_dim, EMBED_DIM)),
            },
            'feed_forward': {
                'gate_proj': normal(g_key, (EMBED_DIM, FFN_DIM)), 'up_proj': normal(u_key, (EMBED_DIM, FFN_DIM)),
                'down_proj': normal(d_key, (FFN_DIM, EMBED_DIM)),
            },
        }
        frozen_params[f'layer_{i}'] = {'attn_norm_scale': jnp.ones(EMBED_DIM), 'ffn_norm_scale': jnp.ones(EMBED_DIM), 'attention': {}, 'feed_forward': {}}
        attn_configs, ffn_configs = {}, {}
        for name, weight in weights_to_quantize['attention'].items():
            q_w, q_c = quantize_nf4(weight)
            frozen_params[f'layer_{i}']['attention'][name] = q_w
            attn_configs[name] = q_c
        for name, weight in weights_to_quantize['feed_forward'].items():
            q_w, q_c = quantize_nf4(weight)
            frozen_params[f'layer_{i}']['feed_forward'][name] = q_w
            ffn_configs[name] = q_c
        
        final_attn_config = AttentionConfig(**attn_configs, attention_type=LAYER_TYPES[i])
        layer_configs.append(LayerConfig(attention=final_attn_config, feed_forward=FeedForwardConfig(**ffn_configs)))
        
        lora_attn_key, lora_ffn_key = jax.random.split(lora_key)
        lora_params[f'layer_{i}'] = {'attention': {}, 'feed_forward': {}}
        attn_lora_shapes = {'q_proj': (EMBED_DIM, q_proj_dim), 'k_proj': (EMBED_DIM, kv_proj_dim), 'v_proj': (EMBED_DIM, kv_proj_dim), 'o_proj': (q_proj_dim, EMBED_DIM)}
        for k, (in_dim, out_dim) in attn_lora_shapes.items():
            key, subkey = jax.random.split(lora_attn_key)
            lora_params[f'layer_{i}']['attention'][f'{k}_A'] = jax.random.normal(subkey, (in_dim, LORA_RANK)) * 0.01
            lora_params[f'layer_{i}']['attention'][f'{k}_B'] = jnp.zeros((LORA_RANK, out_dim))
        for k in ['gate_proj', 'up_proj', 'down_proj']:
            key, subkey = jax.random.split(lora_ffn_key)
            in_dim, out_dim = (EMBED_DIM, FFN_DIM) if k != 'down_proj' else (FFN_DIM, EMBED_DIM)
            lora_params[f'layer_{i}']['feed_forward'][f'{k}_A'] = jax.random.normal(subkey, (in_dim, LORA_RANK)) * 0.01
            lora_params[f'layer_{i}']['feed_forward'][f'{k}_B'] = jnp.zeros((LORA_RANK, out_dim))

    model_config = StaticModelConfig(layers=tuple(layer_configs))
    return frozen_params, model_config, lora_params


# --- 5. LOSS, OPTIMIZER, AND TRAINING ---
# No architectural changes in this section.

def loss_fn(lora_params: LoRAParams, frozen_params: FrozenParams, model_config: StaticModelConfig, x: JaxArray, y: JaxArray, start_pos: JaxArray) -> JaxArray:
    logits = model_forward(x, start_pos, frozen_params, model_config, lora_params)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, y[..., None], axis=-1))

def init_optimizer_state(params: LoRAParams) -> OptimizerState:
    return {'m': jax.tree_util.tree_map(jnp.zeros_like, params), 'v': jax.tree_util.tree_map(jnp.zeros_like, params), 'step': 0}

def get_learning_rate(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> JaxArray:
    lr_warmup = (step / warmup_steps) * max_lr
    is_warmup = step < warmup_steps
    decay_ratio = jnp.clip((step - warmup_steps) / (total_steps - warmup_steps), 0.0, 1.0)
    coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
    lr_decay = min_lr + coeff * (max_lr - min_lr)
    return jnp.where(is_warmup, lr_warmup, lr_decay)

def clip_grads_by_global_norm(grads: LoRAParams, max_norm: float) -> LoRAParams:
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves))
    clip_coefficient = jnp.minimum(max_norm / (total_norm + 1e-6), 1.0)
    clipped_leaves = [leaf * clip_coefficient for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, clipped_leaves)

def optimizer_update(grads: LoRAParams, state: OptimizerState, params: LoRAParams, learning_rate: float) -> Tuple[LoRAParams, OptimizerState]:
    step = state['step'] + 1
    m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, state['m'], grads)
    v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * (g ** 2), state['v'], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - 0.9 ** step), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - 0.999 ** step), v)
    new_params = jax.tree_util.tree_map(lambda p, m, v: p - learning_rate * (m / (jnp.sqrt(v) + 1e-8) + WEIGHT_DECAY * p), params, m_hat, v_hat)
    new_state = {'m': m, 'v': v, 'step': step}
    return new_params, new_state

@partial(jax.jit, static_argnums=(2,))
def train_step(lora_params: LoRAParams, frozen_params: FrozenParams, model_config: StaticModelConfig, optimizer_state: OptimizerState, x: JaxArray, y: JaxArray, start_pos: JaxArray) -> Tuple[JaxArray, LoRAParams, OptimizerState, JaxArray]:
    loss, grads = jax.value_and_grad(loss_fn, argnums=0)(lora_params, frozen_params, model_config, x, y, start_pos)
    clipped_grads = clip_grads_by_global_norm(grads, GRAD_CLIP_NORM)
    current_lr = get_learning_rate(optimizer_state['step'], WARMUP_STEPS, TOTAL_TRAIN_STEPS, MAX_LEARNING_RATE, MIN_LEARNING_RATE)
    new_lora_params, new_optimizer_state = optimizer_update(clipped_grads, optimizer_state, lora_params, current_lr)
    return loss, new_lora_params, new_optimizer_state, current_lr


# --- 6. INFERENCE WITH KV CACHING ---
# Inference logic now uses corrected architectural components implicitly.

def transformer_layer_inference(
    x: JaxArray, start_pos: int, rope_sin: JaxArray, rope_cos: JaxArray, kv_cache: Tuple[JaxArray, JaxArray],
    layer_params: FrozenParams, layer_config: LayerConfig, lora_layer_params: LoRAParams
) -> Tuple[JaxArray, Tuple[JaxArray, JaxArray]]:
    attn_p, attn_c, attn_l = layer_params['attention'], layer_config.attention, lora_layer_params['attention']
    ffn_p, ffn_c, ffn_l = layer_params['feed_forward'], layer_config.feed_forward, lora_layer_params['feed_forward']

    x_norm = rms_norm(x, layer_params['attn_norm_scale'])
    q = qlora_linear(x_norm, attn_p['q_proj'], attn_c.q_proj, attn_l['q_proj_A'], attn_l['q_proj_B'])
    k = qlora_linear(x_norm, attn_p['k_proj'], attn_c.k_proj, attn_l['k_proj_A'], attn_l['k_proj_B'])
    v = qlora_linear(x_norm, attn_p['v_proj'], attn_c.v_proj, attn_l['v_proj_A'], attn_l['v_proj_B'])

    bsz, seq_len, _ = x.shape
    q = q.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    q = apply_rotary_embeddings(q, start_pos, rope_sin, rope_cos)
    k = apply_rotary_embeddings(k, start_pos, rope_sin, rope_cos)

    k_cache, v_cache = kv_cache
    k_cache = jax.lax.dynamic_update_slice(k_cache, k.astype(k_cache.dtype), [0, 0, start_pos, 0])
    v_cache = jax.lax.dynamic_update_slice(v_cache, v.astype(v_cache.dtype), [0, 0, start_pos, 0])

    n_rep = NUM_HEADS // NUM_KV_HEADS
    k_repeated = jnp.repeat(k_cache, n_rep, axis=1)
    v_repeated = jnp.repeat(v_cache, n_rep, axis=1)

    scores = (q @ k_repeated.transpose(0, 1, 3, 2)) / jnp.sqrt(QUERY_PRE_ATTN_SCALAR)
    
    key_positions = jnp.arange(MAX_CONTEXT_LEN)
    causal_mask = key_positions <= start_pos
    if attn_c.attention_type == 'sliding_attention':
        window_start = start_pos - SLIDING_WINDOW_SIZE + 1
        sliding_mask = key_positions >= window_start
        mask = causal_mask & sliding_mask
    else: # 'full_attention'
        mask = causal_mask
    
    mask = mask[None, None, None, :]
    scores = jnp.where(mask, scores, -jnp.inf)
    scores = jax.nn.softmax(scores, axis=-1)
    
    attn_out = (scores @ v_repeated).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
    
    h = x + qlora_linear(attn_out, attn_p['o_proj'], attn_c.o_proj, attn_l['o_proj_A'], attn_l['o_proj_B'])
    ffn_out = feed_forward(rms_norm(h, layer_params['ffn_norm_scale']), ffn_p, ffn_c, ffn_l)
    return h + ffn_out, (k_cache, v_cache)

@partial(jax.jit, static_argnums=(1, 3))
def generate_impl(frozen_params: FrozenParams, model_config: StaticModelConfig, lora_params: LoRAParams,
                  num_tokens_to_generate: int, prompt_tokens: JaxArray) -> JaxArray:
    kv_cache_shape = (1, NUM_KV_HEADS, MAX_CONTEXT_LEN, HEAD_DIM)
    kv_caches = [(jnp.zeros(kv_cache_shape, dtype=COMPUTE_DTYPE), jnp.zeros(kv_cache_shape, dtype=COMPUTE_DTYPE)) for _ in range(NUM_LAYERS)]
    prompt = jnp.array([prompt_tokens])
    prompt_len = prompt.shape[1]
    rope_freqs_map = frozen_params['rope_freqs']
    x = frozen_params['tok_embeddings'][prompt]
    
    # Process prompt
    for i in range(NUM_LAYERS):
        if LAYER_TYPES[i] == 'full_attention':
            rope_sin, rope_cos = rope_freqs_map['global']
        else:
            rope_sin, rope_cos = rope_freqs_map['local']
        x, kv_caches[i] = transformer_layer_inference(x, 0, rope_sin, rope_cos, kv_caches[i], frozen_params[f'layer_{i}'], model_config.layers[i], lora_params[f'layer_{i}'])
    
    logits = rms_norm(x, frozen_params['output_norm_scale']) @ frozen_params['tok_embeddings'].T
    next_token = jnp.argmax(logits[:, -1, :], axis=-1)

    def body_fun(i: int, state: Tuple) -> Tuple:
        kv_caches, current_token, generated_seq = state
        start_pos = prompt_len + i
        token_input = current_token.reshape(1, 1)
        x = frozen_params['tok_embeddings'][token_input]
        
        # Generate next token
        for j in range(NUM_LAYERS):
            if LAYER_TYPES[j] == 'full_attention':
                rope_sin, rope_cos = rope_freqs_map['global']
            else:
                rope_sin, rope_cos = rope_freqs_map['local']
            x, kv_caches[j] = transformer_layer_inference(x, start_pos, rope_sin, rope_cos, kv_caches[j], frozen_params[f'layer_{j}'], model_config.layers[j], lora_params[f'layer_{j}'])
        
        logits = rms_norm(x, frozen_params['output_norm_scale']) @ frozen_params['tok_embeddings'].T
        next_tok = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_seq = generated_seq.at[i].set(next_tok[0])
        return kv_caches, next_tok, generated_seq

    generated_sequence = jnp.zeros(num_tokens_to_generate - 1, dtype=jnp.int32)
    initial_state = (kv_caches, next_token, generated_sequence)
    _, _, final_generated_sequence = jax.lax.fori_loop(0, num_tokens_to_generate - 1, body_fun, initial_state)
    return jnp.concatenate([next_token, final_generated_sequence])

def generate(prompt_tokens: List[int], frozen_params: FrozenParams, model_config: StaticModelConfig, lora_params: LoRAParams, num_new_tokens: int) -> List[int]:
    print(f"\n--- Generating Sequence (Corrected Hybrid Sliding/Full Attention with Dual RoPE) ---")
    start_time = time.time()
    generated_ids = generate_impl(frozen_params, model_config, lora_params, num_new_tokens, jnp.array(prompt_tokens))
    generated_ids.block_until_ready()
    end_time = time.time()
    duration = end_time - start_time
    tokens_per_sec = num_new_tokens / duration if duration > 0 else float('inf')
    print(f"Generation took {duration:.4f} seconds ({tokens_per_sec:.2f} tokens/sec).")
    return generated_ids.tolist()


# --- 7. MAIN SCRIPT & ARCHITECTURE VALIDATION ---
# Unchanged

def calculate_parameter_count(config: Dict[str, int]) -> Tuple[int, int, int]:
    embedding_params = config['VOCAB_SIZE'] * config['EMBED_DIM']
    q_proj_dim, kv_proj_dim = config['NUM_HEADS'] * config['HEAD_DIM'], config['NUM_KV_HEADS'] * config['HEAD_DIM']
    q_proj_params, k_proj_params = config['EMBED_DIM'] * q_proj_dim, config['EMBED_DIM'] * kv_proj_dim
    v_proj_params, o_proj_params = config['EMBED_DIM'] * kv_proj_dim, q_proj_dim * config['EMBED_DIM']
    attn_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params
    ffn_params_per_layer = (config['EMBED_DIM'] * config['FFN_DIM']) * 2 + (config['FFN_DIM'] * config['EMBED_DIM'])
    norm_params_per_layer = 2 * config['EMBED_DIM']
    transformer_params_per_layer = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    total_transformer_params = config['NUM_LAYERS'] * transformer_params_per_layer
    total_params = embedding_params + total_transformer_params + config['EMBED_DIM']
    return total_params, embedding_params, total_transformer_params

def main() -> None:
    key = jax.random.PRNGKey(42)
    print("--- Validating Simulated Gemma 3 270M Architecture (Rigorously Corrected) ---")
    model_config_dict = {
        'VOCAB_SIZE': VOCAB_SIZE, 'EMBED_DIM': EMBED_DIM, 'NUM_LAYERS': NUM_LAYERS, 'FFN_DIM': FFN_DIM,
        'NUM_HEADS': NUM_HEADS, 'NUM_KV_HEADS': NUM_KV_HEADS, 'HEAD_DIM': HEAD_DIM
    }
    total, embed, transformer = calculate_parameter_count(model_config_dict)
    print(f"Official Vocabulary Size: {VOCAB_SIZE:,}")
    print(f"Calculated Total Parameters: {total / 1e6:.2f}M")
    print(f"  - Embedding/Output Parameters: {embed / 1e6:.2f}M")
    print(f"  - Transformer Block Parameters: {transformer / 1e6:.2f}M")
    print("-" * 50)

    print("--- Initializing model with corrected Gemma 3 270M config ---")
    key, subkey = jax.random.split(key)
    frozen_params, model_config, lora_params = init_gemma3_270m_params(subkey, VOCAB_SIZE)
    optimizer_state = init_optimizer_state(lora_params)
    dataloader = data_generator(data_size=100000, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    
    print(f"\n--- Simulating Finetuning (SeqLen: {SEQ_LEN}, Batch: {BATCH_SIZE}) ---")
    for epoch in range(NUM_EPOCHS):
        x_batch, y_batch, start_pos_batch = next(dataloader)
        loss, lora_params, optimizer_state, current_lr = train_step(lora_params, frozen_params, model_config, optimizer_state, x_batch, y_batch, start_pos_batch)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | LR: {current_lr:.6f}")
    
    print("\n--- Simulated Finetuning complete ---")
    simulated_prompt = list(range(1, 31))
    generated_ids = generate(simulated_prompt, frozen_params, model_config, lora_params, NUM_TOKENS_TO_GENERATE)
    print(f"\nOriginal prompt (token IDs): {simulated_prompt}")
    print(f"Model Output (token IDs): {generated_ids}")

if __name__ == '__main__':
    main()