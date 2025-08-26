import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
from collections import namedtuple

# --- 1. CONFIGURATION ---
VOCAB_SIZE = 26
SEQ_LEN = 32
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = EMBED_DIM * 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
COMPUTE_DTYPE = jnp.bfloat16
LORA_RANK = 8
QLORA_BLOCK_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500
WEIGHT_DECAY = 0.01
NUM_TOKENS_TO_GENERATE = 64
MAX_CONTEXT_LEN = SEQ_LEN + NUM_TOKENS_TO_GENERATE

# --- 2. QLoRA & HASHABLE CONFIG ---
QuantConfig = namedtuple('QuantConfig', ['original_shape', 'original_numel', 'block_size', 'padded_size'])
AttentionConfig = namedtuple('AttentionConfig', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
FeedForwardConfig = namedtuple('FeedForwardConfig', ['gate_proj', 'up_proj', 'down_proj'])
LayerConfig = namedtuple('LayerConfig', ['attention', 'feed_forward'])
StaticModelConfig = namedtuple('StaticModelConfig', ['layers'])

# Note: The official NF4 values have 16 entries, with 0 repeated.
# Using 15 is fine but we'll use the full 16 to be perfectly aligned with some implementations.
# This doesn't change the logic as 0 will map to itself.
NF4_QUANTIZATION_VALUES = jnp.array([
    -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, -0.0000,
     0.0000,  0.0911,  0.1848,  0.2844,  0.3949,  0.5251,  0.6962,  1.0000
], dtype=jnp.float32)

def pack_4bit(x):
    # Ensure x is a flat array of integers
    x_flat = x.flatten()
    if x_flat.size % 2 != 0:
        # Pad with a zero if the size is odd
        x_flat = jnp.pad(x_flat, (0, 1))
    # Reshape to pair up numbers
    x_pairs = x_flat.reshape(-1, 2)
    # Pack two 4-bit numbers into one uint8
    # The first number goes into the lower 4 bits, the second into the upper 4 bits
    return (x_pairs[:, 0] | (x_pairs[:, 1] << 4)).astype(jnp.uint8)

def unpack_4bit(x_packed, original_size):
    # Unpack the lower 4 bits (x & 15) and the upper 4 bits (x >> 4)
    unpacked_pairs = jnp.stack([x_packed & 0x0F, x_packed >> 4], axis=-1)
    # Flatten back into a 1D array of 4-bit numbers
    unpacked_flat = unpacked_pairs.flatten()
    # Trim any padding that was added during packing
    return unpacked_flat[:original_size]

def quantize_nf4(w, block_size=QLORA_BLOCK_SIZE):
    original_shape = w.shape
    w_flat = w.flatten()
    original_numel = w_flat.size
    pad_len = (block_size - (original_numel % block_size)) % block_size
    if pad_len > 0:
        w_flat = jnp.pad(w_flat, (0, pad_len))
    w_blocks = w_flat.reshape(-1, block_size)
    scales = jnp.max(jnp.abs(w_blocks), axis=-1, keepdims=True)
    scales = scales.at[scales == 0].set(1.0) # Avoid division by zero
    w_normalized = w_blocks / scales
    # Find the closest NF4 value for each weight
    # This uses broadcasting to compute differences for all values at once
    diffs = jnp.abs(w_normalized.reshape(-1, 1) - NF4_QUANTIZATION_VALUES)
    quantized_indices = jnp.argmin(diffs, axis=-1).astype(jnp.uint8)
    packed_w = pack_4bit(quantized_indices)
    weights = {'qw': packed_w, 'scales': scales.flatten()}
    config = QuantConfig(original_shape, original_numel, block_size, w_flat.size)
    return weights, config

def dequantize_nf4(q_weights, q_config):
    indices = unpack_4bit(q_weights['qw'], q_config.padded_size)
    dequantized_flat = NF4_QUANTIZATION_VALUES[indices]
    dequantized_blocks = dequantized_flat.reshape(-1, q_config.block_size)
    # Rescale the blocks
    scaled_blocks = dequantized_blocks * q_weights['scales'].reshape(-1, 1)
    w_flat = scaled_blocks.flatten()
    # Remove padding and restore original shape
    w = w_flat[:q_config.original_numel]
    return w.reshape(q_config.original_shape)

# --- 3. ROTARY POSITIONAL ENCODING (RoPE) ---
def precompute_rope_freqs(dim, max_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.sin(freqs), jnp.cos(freqs)

def apply_rotary_embeddings(x, start_pos, full_rope_sin, full_rope_cos):
    seq_len, head_dim = x.shape[-2:]
    is_batched = jnp.ndim(start_pos) > 0
    if is_batched:
        def get_freqs(start):
            sin = jax.lax.dynamic_slice(full_rope_sin, [start, 0], [seq_len, head_dim // 2])
            cos = jax.lax.dynamic_slice(full_rope_cos, [start, 0], [seq_len, head_dim // 2])
            return sin, cos
        sin, cos = jax.vmap(get_freqs)(start_pos)
        sin, cos = sin[:, None, :, :], cos[:, None, :, :]
    else:
        sin = jax.lax.dynamic_slice(full_rope_sin, [start_pos, 0], [seq_len, head_dim // 2])
        cos = jax.lax.dynamic_slice(full_rope_cos, [start_pos, 0], [seq_len, head_dim // 2])
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
    x_real, x_imag = x[..., 0::2], x[..., 1::2]
    x_rotated_real = x_real * cos - x_imag * sin
    x_rotated_imag = x_real * sin + x_imag * cos
    y = jnp.empty_like(x)
    y = y.at[..., 0::2].set(x_rotated_real)
    y = y.at[..., 1::2].set(x_rotated_imag)
    return y

# --- 4. MODEL ARCHITECTURE ---
def rms_norm(x, w, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps) * w

def qlora_linear(x, frozen_q_w, q_config, lora_a, lora_b):
    dequantized_w = dequantize_nf4(frozen_q_w, q_config)
    # Cast to compute dtype for matmuls
    x_compute = x.astype(COMPUTE_DTYPE)
    dequantized_w_compute = dequantized_w.astype(COMPUTE_DTYPE)
    lora_a_compute = lora_a.astype(COMPUTE_DTYPE)
    lora_b_compute = lora_b.astype(COMPUTE_DTYPE)
    # Frozen path
    frozen_out = x_compute @ dequantized_w_compute
    # LoRA path
    lora_out = (x_compute @ lora_a_compute) @ lora_b_compute
    return (frozen_out + lora_out).astype(jnp.float32)

def attention(x, mask, rope_sin, rope_cos, start_pos, frozen_p, config_p, lora_p):
    bsz, seq_len, _ = x.shape
    q = qlora_linear(x, frozen_p['q_proj'], config_p.q_proj, lora_p['q_proj_A'], lora_p['q_proj_B'])
    k = qlora_linear(x, frozen_p['k_proj'], config_p.k_proj, lora_p['k_proj_A'], lora_p['k_proj_B'])
    v = qlora_linear(x, frozen_p['v_proj'], config_p.v_proj, lora_p['v_proj_A'], lora_p['v_proj_B'])
    q = q.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    q = apply_rotary_embeddings(q, start_pos, rope_sin, rope_cos)
    k = apply_rotary_embeddings(k, start_pos, rope_sin, rope_cos)
    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(HEAD_DIM)
    scores = jnp.where(mask, scores, -jnp.inf)
    scores = jax.nn.softmax(scores, axis=-1)
    attn_out = (scores @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, EMBED_DIM)
    return qlora_linear(attn_out, frozen_p['o_proj'], config_p.o_proj, lora_p['o_proj_A'], lora_p['o_proj_B'])

def feed_forward(x, frozen_p, config_p, lora_p):
    gate = qlora_linear(x, frozen_p['gate_proj'], config_p.gate_proj, lora_p['gate_proj_A'], lora_p['gate_proj_B'])
    up = qlora_linear(x, frozen_p['up_proj'], config_p.up_proj, lora_p['up_proj_A'], lora_p['up_proj_B'])
    ffn_out = jax.nn.silu(gate) * up
    return qlora_linear(ffn_out, frozen_p['down_proj'], config_p.down_proj, lora_p['down_proj_A'], lora_p['down_proj_B'])

def transformer_layer(x, mask, rope_sin, rope_cos, start_pos, frozen_p, config_p, lora_p):
    h = x + attention(rms_norm(x, frozen_p['attn_norm_scale']), mask, rope_sin, rope_cos, start_pos, frozen_p['attention'], config_p.attention, lora_p['attention'])
    out = h + feed_forward(rms_norm(h, frozen_p['ffn_norm_scale']), frozen_p['feed_forward'], config_p.feed_forward, lora_p['feed_forward'])
    return out

def model_forward(tokens, start_pos, frozen_params, model_config, lora_params):
    x = frozen_params['tok_embeddings'][tokens]
    rope_sin, rope_cos = frozen_params['rope_freqs']
    mask = jnp.tril(jnp.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=bool))
    for i in range(NUM_LAYERS):
        x = transformer_layer(x, mask, rope_sin, rope_cos, start_pos, frozen_params[f'layer_{i}'], model_config.layers[i], lora_params[f'layer_{i}'])
    x = rms_norm(x, frozen_params['output_norm_scale'])
    # The output projection is not quantized in this setup
    return x @ frozen_params['output_proj'].T

# --- 5. DATA & INIT ---
def get_dummy_data():
    vocab = "abcdefghijklmnopqrstuvwxyz"
    char_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_char = {i: c for i, c in enumerate(vocab)}
    data = "thequickbrownfoxjumpsoverthelazydog"
    tokenize = lambda s: [char_to_int.get(c, 0) for c in s if c in char_to_int]
    detokenize = lambda ids: "".join([int_to_char.get(i, "") for i in ids])
    tokenized_data = tokenize(data * 100)
    return np.array(tokenized_data), tokenize, detokenize, len(vocab)

def data_generator(data, batch_size, seq_len):
    while True:
        idxs = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
        x = np.stack([data[i:i+seq_len] for i in idxs])
        y = np.stack([data[i+1:i+seq_len+1] for i in idxs])
        yield jnp.array(x), jnp.array(y), jnp.array(idxs)

def init_params(key, vocab_size):
    def normal(key, shape): return jax.random.normal(key, shape, dtype=jnp.float32)
    key, tok_key, out_key = jax.random.split(key, 3)
    frozen_params = {
        'tok_embeddings': normal(tok_key, (vocab_size, EMBED_DIM)),
        'output_proj': normal(out_key, (vocab_size, EMBED_DIM)),
        'output_norm_scale': jnp.ones(EMBED_DIM),
        'rope_freqs': precompute_rope_freqs(HEAD_DIM, MAX_CONTEXT_LEN)
    }
    lora_params = {}
    layer_configs = []
    layer_keys = jax.random.split(key, NUM_LAYERS)
    for i in range(NUM_LAYERS):
        layer_key = layer_keys[i]
        q_key, k_key, v_key, o_key, g_key, u_key, d_key = jax.random.split(layer_key, 7)
        # Using standard (in_dim, out_dim) convention now
        weights_to_quantize = {
            'attention': {'q_proj': normal(q_key, (EMBED_DIM, EMBED_DIM)), 'k_proj': normal(k_key, (EMBED_DIM, EMBED_DIM)), 'v_proj': normal(v_key, (EMBED_DIM, EMBED_DIM)), 'o_proj': normal(o_key, (EMBED_DIM, EMBED_DIM)),},
            'feed_forward': {'gate_proj': normal(g_key, (EMBED_DIM, FFN_DIM)), 'up_proj': normal(u_key, (EMBED_DIM, FFN_DIM)), 'down_proj': normal(d_key, (FFN_DIM, EMBED_DIM)),},
        }
        frozen_params[f'layer_{i}'] = {'attn_norm_scale': jnp.ones(EMBED_DIM), 'ffn_norm_scale': jnp.ones(EMBED_DIM), 'attention': {}, 'feed_forward': {}}
        print(f"Quantizing layer {i} to NF4 format...")
        attn_configs, ffn_configs = {}, {}
        for name, weight in weights_to_quantize['attention'].items():
            q_w, q_c = quantize_nf4(weight)
            frozen_params[f'layer_{i}']['attention'][name] = q_w
            attn_configs[name] = q_c
        for name, weight in weights_to_quantize['feed_forward'].items():
            q_w, q_c = quantize_nf4(weight)
            frozen_params[f'layer_{i}']['feed_forward'][name] = q_w
            ffn_configs[name] = q_c
        layer_configs.append(LayerConfig(AttentionConfig(**attn_configs), FeedForwardConfig(**ffn_configs)))
        lora_layer_key, lora_ffn_key = jax.random.split(layer_keys[i])
        lora_q_key, lora_k_key, lora_v_key, lora_o_key = jax.random.split(lora_layer_key, 4)
        lora_g_key, lora_u_key, lora_d_key = jax.random.split(lora_ffn_key, 3)
        lora_params[f'layer_{i}'] = {'attention': {}, 'feed_forward': {}}
        for k, lk in zip(['q_proj', 'k_proj', 'v_proj', 'o_proj'], [lora_q_key, lora_k_key, lora_v_key, lora_o_key]):
            lora_params[f'layer_{i}']['attention'][f'{k}_A'] = normal(lk, (EMBED_DIM, LORA_RANK)) * 0.01
            lora_params[f'layer_{i}']['attention'][f'{k}_B'] = jnp.zeros((LORA_RANK, EMBED_DIM))
        for k, lk in zip(['gate_proj', 'up_proj', 'down_proj'], [lora_g_key, lora_u_key, lora_d_key]):
            in_dim, out_dim = (EMBED_DIM, FFN_DIM) if k != 'down_proj' else (FFN_DIM, EMBED_DIM)
            lora_params[f'layer_{i}']['feed_forward'][f'{k}_A'] = normal(lk, (in_dim, LORA_RANK)) * 0.01
            lora_params[f'layer_{i}']['feed_forward'][f'{k}_B'] = jnp.zeros((LORA_RANK, out_dim))

    model_config = StaticModelConfig(layers=tuple(layer_configs))
    return frozen_params, model_config, lora_params

# --- 6. LOSS, OPTIMIZER, TRAINING ---
def loss_fn(lora_params, frozen_params, model_config, x, y, start_pos):
    logits = model_forward(x, start_pos, frozen_params, model_config, lora_params)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, y[..., None], axis=-1))

def init_optimizer_state(params):
    return {'m': jax.tree_util.tree_map(jnp.zeros_like, params), 'v': jax.tree_util.tree_map(jnp.zeros_like, params), 'step': 0}

def optimizer_update(grads, state, params, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
    state['step'] += 1
    state['m'] = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state['m'], grads)
    state['v'] = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * (g ** 2), state['v'], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** state['step']), state['m'])
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** state['step']), state['v'])
    new_params = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / (jnp.sqrt(v) + eps) + wd * p), params, m_hat, v_hat)
    return new_params, state

@partial(jax.jit, static_argnums=(2,))
def train_step(lora_params, frozen_params, model_config, optimizer_state, x, y, start_pos):
    loss, grads = jax.value_and_grad(loss_fn, argnums=0)(lora_params, frozen_params, model_config, x, y, start_pos)
    new_lora_params, new_optimizer_state = optimizer_update(grads, optimizer_state, lora_params, LEARNING_RATE, WEIGHT_DECAY)
    return loss, new_lora_params, new_optimizer_state

# --- 7. INFERENCE WITH KV CACHING ---
def transformer_layer_inference(x, start_pos, rope_sin, rope_cos, kv_cache, layer_params, layer_config, lora_layer_params):
    attn_p, attn_c, attn_l = layer_params['attention'], layer_config.attention, lora_layer_params['attention']
    ffn_p, ffn_c, ffn_l = layer_params['feed_forward'], layer_config.feed_forward, lora_layer_params['feed_forward']
    x_norm = rms_norm(x, layer_params['attn_norm_scale'])
    q = qlora_linear(x_norm, attn_p['q_proj'], attn_c.q_proj, attn_l['q_proj_A'], attn_l['q_proj_B'])
    k = qlora_linear(x_norm, attn_p['k_proj'], attn_c.k_proj, attn_l['k_proj_A'], attn_l['k_proj_B'])
    v = qlora_linear(x_norm, attn_p['v_proj'], attn_c.v_proj, attn_l['v_proj_A'], attn_l['v_proj_B'])

    bsz, seq_len, _ = x.shape
    q = q.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    q = apply_rotary_embeddings(q, start_pos, rope_sin, rope_cos)
    k = apply_rotary_embeddings(k, start_pos, rope_sin, rope_cos)
    k_cache, v_cache = kv_cache
    k_cache = jax.lax.dynamic_update_slice(k_cache, k, [0, 0, start_pos, 0])
    v_cache = jax.lax.dynamic_update_slice(v_cache, v, [0, 0, start_pos, 0])

    # --- FIX STARTS HERE ---
    # The key change is to compute scores against the FULL cache and use a mask
    # that has a fixed shape based on MAX_CONTEXT_LEN.
    scores = (q @ k_cache.transpose(0, 1, 3, 2)) / jnp.sqrt(HEAD_DIM)

    # Create a mask that is independent of dynamic shapes.
    # It accounts for both causality and padding.
    query_positions = jnp.arange(seq_len) + start_pos
    key_positions = jnp.arange(MAX_CONTEXT_LEN)
    mask = query_positions[:, None] >= key_positions[None, :]
    mask = mask[None, None, :, :] # Add dimensions for batch and heads

    scores = jnp.where(mask, scores, -jnp.inf)
    scores = jax.nn.softmax(scores, axis=-1)
    attn_out = (scores @ v_cache).transpose(0, 2, 1, 3).reshape(bsz, seq_len, EMBED_DIM)
    # --- FIX ENDS HERE ---

    h = x + qlora_linear(attn_out, attn_p['o_proj'], attn_c.o_proj, attn_l['o_proj_A'], attn_l['o_proj_B'])
    ffn_out = feed_forward(rms_norm(h, layer_params['ffn_norm_scale']), ffn_p, ffn_c, ffn_l)
    return h + ffn_out, (k_cache, v_cache)

@partial(jax.jit, static_argnums=(1, 3))
def generate_impl(frozen_params, model_config, lora_params, num_tokens_to_generate, prompt_tokens):
    # Initialize a KV cache for each layer
    kv_caches = [(jnp.zeros((1, NUM_HEADS, MAX_CONTEXT_LEN, HEAD_DIM)), jnp.zeros((1, NUM_HEADS, MAX_CONTEXT_LEN, HEAD_DIM))) for _ in range(NUM_LAYERS)]
    prompt = jnp.array([prompt_tokens])
    prompt_len = prompt.shape[1]
    rope_sin, rope_cos = frozen_params['rope_freqs']

    # Process prompt
    x = frozen_params['tok_embeddings'][prompt]
    for i in range(NUM_LAYERS):
        x, kv_caches[i] = transformer_layer_inference(x, 0, rope_sin, rope_cos, kv_caches[i], frozen_params[f'layer_{i}'], model_config.layers[i], lora_params[f'layer_{i}'])
    
    logits = rms_norm(x, frozen_params['output_norm_scale']) @ frozen_params['output_proj'].T
    next_token = jnp.argmax(logits[:, -1, :], axis=-1)

    # Autoregressive generation loop
    def body_fun(i, state):
        kv_caches, current_token, generated_seq = state
        start_pos = prompt_len + i
        token_input = current_token.reshape(1, 1)
        
        x = frozen_params['tok_embeddings'][token_input]
        for j in range(NUM_LAYERS):
            x, kv_caches[j] = transformer_layer_inference(x, start_pos, rope_sin, rope_cos, kv_caches[j], frozen_params[f'layer_{j}'], model_config.layers[j], lora_params[f'layer_{j}'])
        
        logits = rms_norm(x, frozen_params['output_norm_scale']) @ frozen_params['output_proj'].T
        next_tok = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_seq = generated_seq.at[i].set(next_tok[0])
        return kv_caches, next_tok, generated_seq

    # We already have the first token, so generate N-1 more
    generated_sequence = jnp.zeros(num_tokens_to_generate - 1, dtype=jnp.int32)
    initial_state = (kv_caches, next_token, generated_sequence)
    
    _, _, final_generated_sequence = jax.lax.fori_loop(0, num_tokens_to_generate - 1, body_fun, initial_state)
    
    return jnp.concatenate([next_token, final_generated_sequence])

def generate_with_kv_cache(prompt_tokens, frozen_params, model_config, lora_params, num_new_tokens):
    print("\n--- Generating Sequence (QLoRA enabled) ---")
    start_gen = time.time()
    generated = generate_impl(frozen_params, model_config, lora_params, num_new_tokens, prompt_tokens)
    generated.block_until_ready()
    end_gen = time.time()
    print(f"Generation took {end_gen - start_gen:.4f} seconds ({num_new_tokens / (end_gen - start_gen):.2f} tokens/sec).")
    return generated.tolist()

# --- 8. MAIN SCRIPT ---
if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    data, tokenize, detokenize, vocab_size = get_dummy_data()
    dataloader = data_generator(data, BATCH_SIZE, SEQ_LEN)
    key, subkey = jax.random.split(key)
    print("--- Initializing model with QLoRA (NF4 Quantization) ---")
    frozen_params, model_config, lora_params = init_params(subkey, vocab_size)
    optimizer_state = init_optimizer_state(lora_params)
    
    print("\n--- Finetuning with QLoRA ---")
    for epoch in range(NUM_EPOCHS):
        x_batch, y_batch, start_pos_batch = next(dataloader)
        loss, lora_params, optimizer_state = train_step(lora_params, frozen_params, model_config, optimizer_state, x_batch, y_batch, start_pos_batch)
        if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch:4d}/{NUM_EPOCHS} | Loss: {loss:.4f}")
            
    print("\n--- Finetuning complete ---")
    prompt = "thequickbrownfox"
    prompt_tokenized = tokenize(prompt)
    print(f"\nGenerating from prompt: '{prompt}'...")
    
    generated_ids = generate_with_kv_cache(prompt_tokenized, frozen_params, model_config, lora_params, NUM_TOKENS_TO_GENERATE)
    
    generated_text = detokenize(generated_ids)
    print(f"Model Output: {prompt}{generated_text}")
    print("\nExpected output: ...jumpsoverthelazydogthequickbrownfoxjumpsoverthelazydog...")