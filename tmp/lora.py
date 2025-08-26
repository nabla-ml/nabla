import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time

# --- 1. CONFIGURATION ---
# Model Params
VOCAB_SIZE = 26
SEQ_LEN = 32
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = EMBED_DIM * 4
HEAD_DIM = EMBED_DIM // NUM_HEADS

# LoRA Params
LORA_RANK = 8

# Training Params
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500
WEIGHT_DECAY = 0.01

# Generation Params
NUM_TOKENS_TO_GENERATE = 64
MAX_CONTEXT_LEN = SEQ_LEN + NUM_TOKENS_TO_GENERATE

# --- 2. ROTARY POSITIONAL ENCODING (RoPE) ---
def precompute_rope_freqs(dim, max_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.sin(freqs), jnp.cos(freqs)

def apply_rotary_embeddings(x, start_pos, full_rope_sin, full_rope_cos):
    # x shape: (bsz, num_heads, seq_len, head_dim)
    # start_pos shape: (bsz,) during training, or scalar int during inference
    bsz, num_heads, seq_len, head_dim = x.shape

    # --- Handle both batched training and scalar inference ---
    is_batched = jnp.ndim(start_pos) > 0
    if is_batched:
        # Training: start_pos is a batch of starting positions
        def get_freqs(start):
            sin = jax.lax.dynamic_slice(full_rope_sin, [start, 0], [seq_len, head_dim // 2])
            cos = jax.lax.dynamic_slice(full_rope_cos, [start, 0], [seq_len, head_dim // 2])
            return sin, cos
        sin, cos = jax.vmap(get_freqs)(start_pos) # sin/cos shape: (bsz, seq_len, head_dim/2)
        sin = sin[:, None, :, :] # -> (bsz, 1, seq_len, head_dim/2)
        cos = cos[:, None, :, :] # -> (bsz, 1, seq_len, head_dim/2)
    else:
        # Inference: start_pos is a single integer
        sin = jax.lax.dynamic_slice(full_rope_sin, [start_pos, 0], [seq_len, head_dim // 2])
        cos = jax.lax.dynamic_slice(full_rope_cos, [start_pos, 0], [seq_len, head_dim // 2])
        sin = sin[None, None, :, :] # -> (1, 1, seq_len, head_dim/2)
        cos = cos[None, None, :, :] # -> (1, 1, seq_len, head_dim/2)

    x_real, x_imag = x[..., 0::2], x[..., 1::2]
    x_rotated_real = x_real * cos - x_imag * sin
    x_rotated_imag = x_real * sin + x_imag * cos
    y = jnp.empty_like(x)
    y = y.at[..., 0::2].set(x_rotated_real)
    y = y.at[..., 1::2].set(x_rotated_imag)
    return y

# --- 3. TRAINING ARCHITECTURE (Unchanged) ---
def rms_norm(x, w, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps) * w
def lora_linear(x, frozen_w, lora_a, lora_b):
    return (x @ frozen_w) + ((x @ lora_a) @ lora_b)
def attention_train(x, mask, rope_sin, rope_cos, start_pos, frozen_p, lora_p):
    bsz, seq_len, _ = x.shape
    q = lora_linear(x, frozen_p['q_proj'], lora_p['q_proj_A'], lora_p['q_proj_B'])
    k = lora_linear(x, frozen_p['k_proj'], lora_p['k_proj_A'], lora_p['k_proj_B'])
    v = lora_linear(x, frozen_p['v_proj'], lora_p['v_proj_A'], lora_p['v_proj_B'])
    q = q.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    # Apply RoPE with the correct starting position for each sequence in the batch
    q = apply_rotary_embeddings(q, start_pos, rope_sin, rope_cos)
    k = apply_rotary_embeddings(k, start_pos, rope_sin, rope_cos)

    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(HEAD_DIM)
    scores = jnp.where(mask, scores, -jnp.inf)
    scores = jax.nn.softmax(scores, axis=-1)
    attn_out = (scores @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, EMBED_DIM)
    return lora_linear(attn_out, frozen_p['o_proj'], lora_p['o_proj_A'], lora_p['o_proj_B'])

def feed_forward_train(x, frozen_p, lora_p):
    gate = lora_linear(x, frozen_p['gate_proj'], lora_p['gate_proj_A'], lora_p['gate_proj_B'])
    up = lora_linear(x, frozen_p['up_proj'], lora_p['up_proj_A'], lora_p['up_proj_B'])
    return lora_linear(jax.nn.silu(gate) * up, frozen_p['down_proj'], lora_p['down_proj_A'], lora_p['down_proj_B'])

def transformer_layer_train(x, mask, rope_sin, rope_cos, start_pos, frozen_p, lora_p):
    h = x + attention_train(rms_norm(x, frozen_p['attn_norm_scale']), mask, rope_sin, rope_cos, start_pos, frozen_p['attention'], lora_p['attention'])
    out = h + feed_forward_train(rms_norm(h, frozen_p['ffn_norm_scale']), frozen_p['feed_forward'], lora_p['feed_forward'])
    return out

def model_forward_train(tokens, start_pos, frozen_params, lora_params):
    x = frozen_params['tok_embeddings'][tokens]
    rope_sin, rope_cos = frozen_params['rope_freqs']
    mask = jnp.tril(jnp.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=bool))
    for i in range(NUM_LAYERS):
        x = transformer_layer_train(x, mask, rope_sin, rope_cos, start_pos, frozen_params[f'layer_{i}'], lora_params[f'layer_{i}'])
    x = rms_norm(x, frozen_params['output_norm_scale'])
    return x @ frozen_params['output_proj'].T

# --- 4. DATA & INIT (Unchanged) ---
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
    def normal(key, shape): return jax.random.normal(key, shape)
    key, tok_key, out_key = jax.random.split(key, 3)
    frozen_params = {
        'tok_embeddings': normal(tok_key, (vocab_size, EMBED_DIM)),
        'output_proj': normal(out_key, (vocab_size, EMBED_DIM)),
        'output_norm_scale': jnp.ones(EMBED_DIM),
        'rope_freqs': precompute_rope_freqs(HEAD_DIM, MAX_CONTEXT_LEN)
    }
    lora_params = {}
    layer_keys = jax.random.split(key, NUM_LAYERS)
    for i in range(NUM_LAYERS):
        layer_key = layer_keys[i]
        q_key, k_key, v_key, o_key, g_key, u_key, d_key = jax.random.split(layer_key, 7)
        frozen_params[f'layer_{i}'] = {
            'attention': {'q_proj': normal(q_key, (EMBED_DIM, EMBED_DIM)), 'k_proj': normal(k_key, (EMBED_DIM, EMBED_DIM)), 'v_proj': normal(v_key, (EMBED_DIM, EMBED_DIM)), 'o_proj': normal(o_key, (EMBED_DIM, EMBED_DIM)),},
            'feed_forward': {'gate_proj': normal(g_key, (EMBED_DIM, FFN_DIM)), 'up_proj': normal(u_key, (EMBED_DIM, FFN_DIM)), 'down_proj': normal(d_key, (FFN_DIM, EMBED_DIM)),},
            'attn_norm_scale': jnp.ones(EMBED_DIM), 'ffn_norm_scale': jnp.ones(EMBED_DIM),
        }
        lora_layer_key, lora_ffn_key = jax.random.split(layer_keys[i])
        lora_q_key, lora_k_key, lora_v_key, lora_o_key = jax.random.split(lora_layer_key, 4)
        lora_g_key, lora_u_key, lora_d_key = jax.random.split(lora_ffn_key, 3)
        lora_params[f'layer_{i}'] = {'attention': {}, 'feed_forward': {}}
        for k, lk in zip(['q_proj', 'k_proj', 'v_proj', 'o_proj'], [lora_q_key, lora_k_key, lora_v_key, lora_o_key]):
            lora_params[f'layer_{i}']['attention'][f'{k}_A'] = normal(lk, (EMBED_DIM, LORA_RANK)) * 0.01
            lora_params[f'layer_{i}']['attention'][f'{k}_B'] = jnp.zeros((LORA_RANK, EMBED_DIM))
        for k, lk in zip(['gate_proj', 'up_proj', 'down_proj'], [lora_g_key, lora_u_key, lora_d_key]):
            in_dim, out_dim = (FFN_DIM, EMBED_DIM) if k == 'down_proj' else (EMBED_DIM, FFN_DIM)
            lora_params[f'layer_{i}']['feed_forward'][f'{k}_A'] = normal(lk, (in_dim, LORA_RANK)) * 0.01
            lora_params[f'layer_{i}']['feed_forward'][f'{k}_B'] = jnp.zeros((LORA_RANK, out_dim))
    return frozen_params, lora_params

# --- 5. LOSS, OPTIMIZER, TRAINING (Unchanged) ---
def loss_fn(lora_params, frozen_params, x, y, start_pos):
    logits = model_forward_train(x, start_pos, frozen_params, lora_params)
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

@jax.jit
def train_step(lora_params, frozen_params, optimizer_state, x, y, start_pos):
    loss, grads = jax.value_and_grad(loss_fn)(lora_params, frozen_params, x, y, start_pos)
    new_lora_params, new_optimizer_state = optimizer_update(grads, optimizer_state, lora_params, LEARNING_RATE, WEIGHT_DECAY)
    return loss, new_lora_params, new_optimizer_state

# --- 6. INFERENCE WITH KV CACHING (REFACTORED) ---
@jax.jit
def merge_lora_params(frozen_params, lora_params):
    merged_params = jax.tree_util.tree_map(lambda x: x, frozen_params)
    for i in range(NUM_LAYERS):
        for component in ['attention', 'feed_forward']:
            for name in lora_params[f'layer_{i}'][component]:
                if name.endswith('_A'):
                    base_name, lora_A = name[:-2], lora_params[f'layer_{i}'][component][name]
                    lora_B = lora_params[f'layer_{i}'][component][f'{base_name}_B']
                    merged_params[f'layer_{i}'][component][base_name] += lora_A @ lora_B
    return merged_params

def attention_inference(x, start_pos, rope_sin, rope_cos, kv_cache, layer_params):
    bsz, seq_len, _ = x.shape
    q = (x @ layer_params['q_proj']).reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = (x @ layer_params['k_proj']).reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = (x @ layer_params['v_proj']).reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    q = apply_rotary_embeddings(q, start_pos, rope_sin, rope_cos)
    k = apply_rotary_embeddings(k, start_pos, rope_sin, rope_cos)

    k_cache, v_cache = kv_cache
    k_cache = jax.lax.dynamic_update_slice(k_cache, k, [0, 0, start_pos, 0])
    v_cache = jax.lax.dynamic_update_slice(v_cache, v, [0, 0, start_pos, 0])

    scores = (q @ k_cache.transpose(0, 1, 3, 2)) / jnp.sqrt(HEAD_DIM)

    # Create combined mask for causal attention and padding
    valid_len = start_pos + seq_len
    query_pos = jnp.arange(seq_len) + start_pos
    key_pos = jnp.arange(MAX_CONTEXT_LEN)
    
    causal_mask = query_pos[:, None] >= key_pos[None, :]
    padding_mask = key_pos < valid_len
    final_mask = jnp.logical_and(causal_mask, padding_mask[None, :])
    final_mask = final_mask[None, None, :, :]

    scores = jnp.where(final_mask, scores, -jnp.inf)
    scores = jax.nn.softmax(scores, axis=-1)
    attn_out = (scores @ v_cache).transpose(0, 2, 1, 3).reshape(bsz, seq_len, EMBED_DIM)
    return (attn_out @ layer_params['o_proj']), (k_cache, v_cache)

def transformer_layer_inference(x, start_pos, rope_sin, rope_cos, kv_cache, layer_params):
    attn_out, updated_cache = attention_inference(rms_norm(x, layer_params['attn_norm_scale']), start_pos, rope_sin, rope_cos, kv_cache, layer_params['attention'])
    h = x + attn_out
    ffn_norm_h = rms_norm(h, layer_params['ffn_norm_scale'])
    ffn_params = layer_params['feed_forward']
    ffn_out = jax.nn.silu(ffn_norm_h @ ffn_params['gate_proj']) * (ffn_norm_h @ ffn_params['up_proj'])
    out = h + (ffn_out @ ffn_params['down_proj'])
    return out, updated_cache

# NEW: JIT-compiled function specifically for the multi-token prompt
@jax.jit
def process_prompt(params, prompt_tokens, kv_caches):
    x = params['tok_embeddings'][prompt_tokens]
    rope_sin, rope_cos = params['rope_freqs']
    
    for i in range(NUM_LAYERS):
        x, new_cache = transformer_layer_inference(x, 0, rope_sin, rope_cos, kv_caches[i], params[f'layer_{i}'])
        kv_caches[i] = new_cache
        
    logits = rms_norm(x, params['output_norm_scale']) @ params['output_proj'].T
    return logits, kv_caches

# NEW: JIT-compiled function specifically for single-token generation
@jax.jit
def generate_token(params, token, start_pos, kv_caches):
    x = params['tok_embeddings'][token]
    rope_sin, rope_cos = params['rope_freqs']
    
    for i in range(NUM_LAYERS):
        x, new_cache = transformer_layer_inference(x, start_pos, rope_sin, rope_cos, kv_caches[i], params[f'layer_{i}'])
        kv_caches[i] = new_cache
        
    logits = rms_norm(x, params['output_norm_scale']) @ params['output_proj'].T
    return logits, kv_caches

# NEW: JIT-compiled function for the generation loop
@partial(jax.jit, static_argnums=(3,))
def generate_autoregressive(params, kv_caches, initial_token, num_tokens_to_generate, prompt_len):

    def body_fun(i, state):
        kv_caches, current_token, generated_seq = state
        start_pos = prompt_len + i
        token_input = current_token.reshape(1, 1)
        logits, kv_caches = generate_token(params, token_input, start_pos, kv_caches)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_seq = generated_seq.at[i].set(next_token[0])
        return kv_caches, next_token, generated_seq

    generated_sequence = jnp.zeros(num_tokens_to_generate, dtype=jnp.int32)
    initial_state = (kv_caches, initial_token, generated_sequence)
    final_kv_caches, _, final_generated_sequence = jax.lax.fori_loop(
        0,
        num_tokens_to_generate,
        body_fun,
        initial_state
    )
    return final_generated_sequence, final_kv_caches


def generate_with_kv_cache(prompt_tokens, frozen_params, lora_params, num_new_tokens):
    print("\n--- Generating Sequence ---")

    # --- 1. Merge Weights ---
    print("Merging LoRA weights for inference...")
    start_merge = time.time()
    merged_params = merge_lora_params(frozen_params, lora_params)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), merged_params)
    end_merge = time.time()
    print(f"Merging weights took {end_merge - start_merge:.4f} seconds.")

    # --- 2. Initialize KV Cache ---
    kv_caches = [
        (jnp.zeros((1, NUM_HEADS, MAX_CONTEXT_LEN, HEAD_DIM)), jnp.zeros((1, NUM_HEADS, MAX_CONTEXT_LEN, HEAD_DIM)))
        for _ in range(NUM_LAYERS)
    ]

    # --- 3. Process Prompt ---
    prompt = jnp.array([prompt_tokens])
    prompt_len = len(prompt_tokens)
    print(f"Processing prompt ({prompt_len} tokens)...")
    start_prompt = time.time()
    logits, kv_caches = process_prompt(merged_params, prompt, kv_caches)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), kv_caches)
    next_token = jnp.argmax(logits[:, -1, :], axis=-1) # Ensure shape is (1,)
    end_prompt = time.time()
    print(f"Prompt processing took {end_prompt - start_prompt:.4f} seconds.")

    # --- 4. Generate New Tokens ---
    print(f"Generating {num_new_tokens} new tokens...")
    start_gen = time.time()

    # We generate num_new_tokens total. The first one is `next_token`.
    # The autoregressive loop generates the remaining `num_new_tokens - 1`.
    if num_new_tokens > 1:
        generated_ids_rest, _ = generate_autoregressive(
            merged_params,
            kv_caches,
            next_token,
            num_new_tokens - 1,
            prompt_len
        )
        generated = jnp.concatenate([next_token, generated_ids_rest])
    else:
        generated = next_token

    generated.block_until_ready()
    end_gen = time.time()
    print(f"Token generation took {end_gen - start_gen:.4f} seconds ({num_new_tokens / (end_gen - start_gen):.2f} tokens/sec).")

    return generated.tolist()




# --- 7. MAIN SCRIPT ---
if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    data, tokenize, detokenize, vocab_size = get_dummy_data()
    dataloader = data_generator(data, BATCH_SIZE, SEQ_LEN)
    key, subkey = jax.random.split(key)
    frozen_params, lora_params = init_params(subkey, vocab_size)
    optimizer_state = init_optimizer_state(lora_params)
    
    print("--- Finetuning with LoRA ---")
    for epoch in range(NUM_EPOCHS):
        x_batch, y_batch, start_pos_batch = next(dataloader)
        loss, lora_params, optimizer_state = train_step(lora_params, frozen_params, optimizer_state, x_batch, y_batch, start_pos_batch)
        if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch:4d}/{NUM_EPOCHS} | Loss: {loss:.4f}")
            
    print("\n--- Finetuning complete ---")
    prompt = "thequickbrownfox"
    prompt_tokenized = tokenize(prompt)
    print(f"\nGenerating from prompt: '{prompt}'...")
    
    generated_ids = generate_with_kv_cache(prompt_tokenized, frozen_params, lora_params, NUM_TOKENS_TO_GENERATE)
    
    generated_text = detokenize(generated_ids)
    print(f"Model Output: {prompt}{generated_text}")
    print("\nExpected output: ...jumpsoverthelazydogthequickbrownfoxjumpsoverthelazydog...")

