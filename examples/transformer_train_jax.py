"""
🤖 EDUCATIONAL TRANSFORMER IMPLEMENTATION WITH JAX

This script demonstrates a complete transformer (encoder-decoder) implementation
from scratch using only raw JAX primitives. The goal is educational clarity
while maintaining proper performance optimizations.

LEARNING OBJECTIVES:
- Understand transformer architecture components (attention, feed-forward, layer norm)
- See how encoder-decoder models work for sequence-to-sequence tasks
- Learn JAX best practices (JIT compilation, functional programming, pytrees)
- Experience end-to-end training of a neural sequence model

TASK:     print("=" * 60)
    print("🤖 TRAINING TRANSFORMER FROM SCRATCH WITH JAX")
    print("=" * 60)
    print(f"📋 Task: Reverse sequences of {SOURCE_SEQ_LEN} integers")
    print(f"🏗️  Architecture: {NUM_LAYERS} layers, {D_MODEL} d_model, {NUM_HEADS} attention heads")
    print(f"📊 Training: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
    print(f"🎯 Vocabulary: {VOCAB_SIZE} tokens (0=PAD, 1=START, 2=END, 3-{VOCAB_SIZE-1}=content)")
    print() Reversal
- Input:  [a, b, c, d, e]
- Output: [e, d, c, b, a, <END>]

KEY FEATURES:
✅ Pure JAX implementation (no high-level libraries)
✅ Full transformer with multi-head attention
✅ Proper masking and positional encoding
✅ JIT-compiled training steps for performance
✅ Educational documentation and clear variable names
✅ Avoids .at operations in favor of concatenation/stacking

ARCHITECTURE DETAILS:
- Encoder: Self-attention + Feed-forward (with residual connections & layer norm)
- Decoder: Masked self-attention + Cross-attention + Feed-forward
- Multi-head attention with proper scaling
- Sinusoidal positional encoding
- AdamW optimizer with bias correction
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
from jax import jit, random, value_and_grad

# ============================================================================
# CONFIGURATION
# ============================================================================

# Task Configuration
VOCAB_SIZE = 20  # Total vocabulary size (0=PAD, 1=START, 2=END, 3-19=content)
SOURCE_SEQ_LEN = 9  # Length of input sequences to reverse
TARGET_SEQ_LEN = SOURCE_SEQ_LEN + 2  # +1 for END token, +1 for START token in decoder
MAX_SEQ_LEN = TARGET_SEQ_LEN

# Model Architecture
NUM_LAYERS = 2  # Number of encoder and decoder layers
D_MODEL = 64  # Model dimension (embedding size)
NUM_HEADS = 4  # Number of attention heads (must divide D_MODEL)
D_FF = 128  # Feed-forward network hidden dimension

# Training Configuration
BATCH_SIZE = 64  # Number of sequences per training batch
LEARNING_RATE = 0.0005  # AdamW learning rate (reduced for stability)
NUM_EPOCHS = 300  # Total training epochs (good balance for demonstration)
PRINT_INTERVAL = 10  # Print progress every N epochs

# ============================================================================
# TRANSFORMER BUILDING BLOCKS
# ============================================================================


def positional_encoding(max_seq_len: int, d_model: int) -> jnp.ndarray:
    """
    Create sinusoidal positional encodings for transformer inputs.

    The encoding alternates between sine and cosine for each dimension:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimension (must be even)

    Returns:
        Positional encoding matrix of shape (1, max_seq_len, d_model)
    """
    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    position = jnp.arange(max_seq_len).reshape(
        (max_seq_len, 1)
    )  # Shape: (max_seq_len, 1)

    # Create dimension indices for pairs: [0, 1, 2, ..., d_model//2-1]
    half_d_model = d_model // 2
    dim_indices = jnp.arange(half_d_model).reshape(
        (1, half_d_model)
    )  # Shape: (1, d_model//2)

    # Calculate scaling factors: 10000^(2i/d_model) for each dimension pair
    scaling_factors = 10000.0 ** (2.0 * dim_indices / d_model)

    # Calculate the angle for each position-dimension pair
    angles = position / scaling_factors  # Shape: (max_seq_len, d_model//2)

    # Calculate sine and cosine
    sin_vals = jnp.sin(angles)  # Shape: (max_seq_len, d_model//2)
    cos_vals = jnp.cos(angles)  # Shape: (max_seq_len, d_model//2)

    # Interleave sine and cosine: [sin, cos, sin, cos, ...]
    # Stack and reshape to create the correct pattern
    stacked = jnp.stack(
        [sin_vals, cos_vals], axis=2
    )  # Shape: (max_seq_len, d_model//2, 2)
    pe = stacked.reshape((max_seq_len, d_model))  # Shape: (max_seq_len, d_model)

    # Add batch dimension
    return pe.reshape((1, max_seq_len, d_model))


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V

    Args:
        q: Query matrix (batch, heads, seq_len_q, d_k)
        k: Key matrix (batch, heads, seq_len_k, d_k)
        v: Value matrix (batch, heads, seq_len_v, d_v)
        mask: Optional attention mask to prevent attention to certain positions

    Returns:
        Attention output (batch, heads, seq_len_q, d_v)
    """
    d_k = q.shape[-1]  # Dimension of key vectors

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = jnp.matmul(q, k.transpose((0, 1, 3, 2))) / jnp.sqrt(d_k)

    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)

    # Apply softmax to get attention weights
    attention_weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention weights to values
    output = jnp.matmul(attention_weights, v)
    return output


def multi_head_attention(x, xa, params, mask=None):
    """
    Multi-head attention mechanism.

    Args:
        x: Query input (batch, seq_len, d_model)
        xa: Key/Value input (batch, seq_len, d_model) - same as x for self-attention
        params: Dictionary containing weight matrices w_q, w_k, w_v, w_o
        mask: Optional attention mask

    Returns:
        Multi-head attention output (batch, seq_len, d_model)
    """
    batch_size, seq_len, d_model = x.shape
    num_heads = NUM_HEADS

    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0, (
        f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
    )
    d_head = d_model // num_heads

    # Linear projections: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    q_linear = jnp.matmul(x, params["w_q"])
    k_linear = jnp.matmul(xa, params["w_k"])
    v_linear = jnp.matmul(xa, params["w_v"])

    # Reshape and transpose for multi-head attention
    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_head)
    q = q_linear.reshape(batch_size, seq_len, num_heads, d_head).transpose((0, 2, 1, 3))
    k = k_linear.reshape(batch_size, -1, num_heads, d_head).transpose((0, 2, 1, 3))
    v = v_linear.reshape(batch_size, -1, num_heads, d_head).transpose((0, 2, 1, 3))

    # Apply scaled dot-product attention
    attention_output = scaled_dot_product_attention(q, k, v, mask)

    # Concatenate heads: (batch, num_heads, seq_len, d_head) -> (batch, seq_len, d_model)
    attention_output = attention_output.transpose((0, 2, 1, 3)).reshape(
        batch_size, seq_len, d_model
    )

    # Final linear projection
    return jnp.matmul(attention_output, params["w_o"])


def feed_forward(x, params):
    """
    Position-wise feed-forward network: FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        x: Input tensor (batch, seq_len, d_model)
        params: Dictionary containing w1, b1, w2, b2

    Returns:
        Feed-forward output (batch, seq_len, d_model)
    """
    # First linear transformation with ReLU activation
    hidden = jax.nn.relu(jnp.matmul(x, params["w1"]) + params["b1"])

    # Second linear transformation (output layer)
    output = jnp.matmul(hidden, params["w2"]) + params["b2"]

    return output


def layer_norm(x, params, eps=1e-6):
    """
    Layer normalization: LayerNorm(x) = γ * (x - μ) / σ + β

    Args:
        x: Input tensor (batch, seq_len, d_model)
        params: Dictionary containing gamma (scale) and beta (shift)
        eps: Small constant for numerical stability

    Returns:
        Layer normalized output (batch, seq_len, d_model)
    """
    # Compute mean and standard deviation across the last dimension (d_model)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)

    # Normalize and apply learnable scale/shift parameters
    normalized = (x - mean) / (std + eps)
    return params["gamma"] * normalized + params["beta"]


# ============================================================================
# ENCODER AND DECODER LAYERS
# ============================================================================


def encoder_layer(x, params, mask):
    """
    Single transformer encoder layer with Pre-Norm architecture.

    Architecture: x = x + MultiHeadAttention(LayerNorm(x))
                  x = x + FFN(LayerNorm(x))

    Args:
        x: Input embeddings (batch, seq_len, d_model)
        params: Layer parameters (mha, ffn, norm1, norm2)
        mask: Attention mask (unused in basic encoder)

    Returns:
        Encoder layer output (batch, seq_len, d_model)
    """
    # Pre-norm: normalize then apply attention, then residual
    norm_x = layer_norm(x, params["norm1"])
    attention_output = multi_head_attention(norm_x, norm_x, params["mha"], mask)
    x = x + attention_output

    # Pre-norm: normalize then apply FFN, then residual
    norm_x = layer_norm(x, params["norm2"])
    ffn_output = feed_forward(norm_x, params["ffn"])
    x = x + ffn_output

    return x


def decoder_layer(x, encoder_output, params, look_ahead_mask, padding_mask):
    """
    Single transformer decoder layer with Pre-Norm architecture.

    Architecture:
    1. x = x + MaskedMultiHeadAttention(LayerNorm(x))
    2. x = x + CrossMultiHeadAttention(LayerNorm(x), encoder_output)
    3. x = x + FFN(LayerNorm(x))

    Args:
        x: Decoder input embeddings (batch, target_seq_len, d_model)
        encoder_output: Encoder output (batch, source_seq_len, d_model)
        params: Layer parameters (masked_mha, cross_mha, ffn, norm1, norm2, norm3)
        look_ahead_mask: Causal mask for self-attention
        padding_mask: Mask for encoder-decoder attention (unused here)

    Returns:
        Decoder layer output (batch, target_seq_len, d_model)
    """
    # 1. Pre-norm masked self-attention
    norm_x = layer_norm(x, params["norm1"])
    masked_attention_output = multi_head_attention(
        norm_x, norm_x, params["masked_mha"], look_ahead_mask
    )
    x = x + masked_attention_output

    # 2. Pre-norm cross-attention
    norm_x = layer_norm(x, params["norm2"])
    cross_attention_output = multi_head_attention(
        norm_x, encoder_output, params["cross_mha"], padding_mask
    )
    x = x + cross_attention_output

    # 3. Pre-norm feed-forward
    norm_x = layer_norm(x, params["norm3"])
    ffn_output = feed_forward(norm_x, params["ffn"])
    x = x + ffn_output

    return x


# ============================================================================
# FULL TRANSFORMER MODEL
# ============================================================================


def transformer_forward(encoder_inputs, decoder_inputs, params):
    """
    Full transformer forward pass (encoder-decoder architecture).

    Args:
        encoder_inputs: Source sequence token ids (batch, source_seq_len)
        decoder_inputs: Target sequence token ids (batch, target_seq_len)
        params: All model parameters

    Returns:
        logits: Output predictions (batch, target_seq_len, vocab_size)
    """
    # Create causal mask for decoder self-attention (prevents looking at future tokens)
    # Using the same logic as our Nabla implementation for exact comparison
    target_seq_len = decoder_inputs.shape[1]
    pos_i = jnp.arange(target_seq_len).reshape((target_seq_len, 1))  # Column positions
    pos_j = jnp.arange(target_seq_len).reshape((1, target_seq_len))  # Row positions
    mask = (pos_i >= pos_j).astype(jnp.float32)  # Lower triangular mask
    look_ahead_mask = mask.reshape((1, 1, target_seq_len, target_seq_len))

    # Get sequence lengths
    encoder_seq_len = encoder_inputs.shape[1]
    decoder_seq_len = decoder_inputs.shape[1]

    # --- ENCODER PROCESSING ---
    # Convert token ids to embeddings and add positional encoding
    encoder_embeddings = params["encoder"]["embedding"][
        encoder_inputs
    ]  # (batch, enc_seq_len, d_model)
    encoder_pos_enc = params["pos_encoding"][
        :, :encoder_seq_len, :
    ]  # (1, enc_seq_len, d_model)
    encoder_x = encoder_embeddings + encoder_pos_enc

    # Pass through encoder layers
    encoder_output = encoder_x
    for layer_idx in range(NUM_LAYERS):
        layer_params = params["encoder"][f"layer_{layer_idx}"]
        encoder_output = encoder_layer(encoder_output, layer_params, mask=None)

    # Final encoder layer norm (important for Pre-Norm architecture)
    encoder_output = layer_norm(encoder_output, params["encoder"]["final_norm"])

    # --- DECODER PROCESSING ---
    # Convert token ids to embeddings and add positional encoding
    decoder_embeddings = params["decoder"]["embedding"][
        decoder_inputs
    ]  # (batch, dec_seq_len, d_model)
    decoder_pos_enc = params["pos_encoding"][
        :, :decoder_seq_len, :
    ]  # (1, dec_seq_len, d_model)
    decoder_x = decoder_embeddings + decoder_pos_enc

    # Pass through decoder layers
    decoder_output = decoder_x
    for layer_idx in range(NUM_LAYERS):
        layer_params = params["decoder"][f"layer_{layer_idx}"]
        decoder_output = decoder_layer(
            decoder_output,
            encoder_output,
            layer_params,
            look_ahead_mask=look_ahead_mask,
            padding_mask=None,
        )

    # Final decoder layer norm (important for Pre-Norm architecture)
    decoder_output = layer_norm(decoder_output, params["decoder"]["final_norm"])

    # Final linear projection to vocabulary size
    logits = jnp.matmul(
        decoder_output, params["output_linear"]
    )  # (batch, dec_seq_len, vocab_size)

    return logits


def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss for sequence-to-sequence training.

    Args:
        logits: Model predictions (batch, seq_len, vocab_size)
        targets: True token indices (batch, seq_len)

    Returns:
        Average cross-entropy loss across batch
    """
    # Convert targets to one-hot encoding
    one_hot_targets = jax.nn.one_hot(targets, num_classes=VOCAB_SIZE)

    # Compute log probabilities and cross-entropy
    log_probs = jax.nn.log_softmax(logits)
    cross_entropy = -jnp.sum(one_hot_targets * log_probs)

    # Average over batch size
    batch_size = logits.shape[0]
    return cross_entropy / batch_size


# --- 4. Initialization ---


def init_transformer_params(key: jax.Array) -> dict[str, Any]:
    """
    Initialize all transformer parameters using Xavier/Glorot initialization.

    Args:
        key: JAX random key for parameter initialization

    Returns:
        Dictionary containing all model parameters organized by component
    """
    # Split key for different parameter groups
    keys = random.split(key, 100)  # Generate enough keys for all parameters
    key_idx = 0

    def glorot_init(key, shape):
        """Xavier/Glorot uniform initialization for better training stability."""
        return jax.nn.initializers.glorot_uniform()(key, shape, dtype=jnp.float32)

    def get_next_key():
        """Helper to get next key and increment counter."""
        nonlocal key_idx
        k = keys[key_idx]
        key_idx += 1
        return k

    # Initialize main parameter structure with flexible typing
    params: dict[str, Any] = {"encoder": {}, "decoder": {}}

    # --- Embedding Layers ---
    print("Initializing embedding layers...")
    params["encoder"]["embedding"] = random.normal(
        get_next_key(), (VOCAB_SIZE, D_MODEL)
    )
    params["decoder"]["embedding"] = random.normal(
        get_next_key(), (VOCAB_SIZE, D_MODEL)
    )

    # --- Positional Encoding (Fixed, not trainable) ---
    params["pos_encoding"] = positional_encoding(MAX_SEQ_LEN, D_MODEL)

    # --- Initialize Encoder Layers ---
    print(f"Initializing {NUM_LAYERS} encoder layers...")
    for layer_idx in range(NUM_LAYERS):
        params["encoder"][f"layer_{layer_idx}"] = _init_encoder_layer_params(
            get_next_key, glorot_init
        )

    # Final encoder layer norm (for Pre-Norm architecture)
    params["encoder"]["final_norm"] = {
        "gamma": jnp.ones(D_MODEL),
        "beta": jnp.zeros(D_MODEL),
    }

    # --- Initialize Decoder Layers ---
    print(f"Initializing {NUM_LAYERS} decoder layers...")
    for layer_idx in range(NUM_LAYERS):
        params["decoder"][f"layer_{layer_idx}"] = _init_decoder_layer_params(
            get_next_key, glorot_init
        )

    # Final decoder layer norm (for Pre-Norm architecture)
    params["decoder"]["final_norm"] = {
        "gamma": jnp.ones(D_MODEL),
        "beta": jnp.zeros(D_MODEL),
    }

    # --- Output Projection Layer ---
    print("Initializing output projection layer...")
    params["output_linear"] = glorot_init(get_next_key(), (D_MODEL, VOCAB_SIZE))

    return params


def _init_encoder_layer_params(get_next_key, glorot_init) -> dict[str, Any]:
    """Initialize parameters for a single encoder layer."""
    return {
        # Multi-head self-attention
        "mha": {
            "w_q": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),  # Query projection
            "w_k": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),  # Key projection
            "w_v": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),  # Value projection
            "w_o": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),  # Output projection
        },
        # Feed-forward network
        "ffn": {
            "w1": glorot_init(get_next_key(), (D_MODEL, D_FF)),  # First linear layer
            "b1": jnp.zeros(D_FF),  # First bias
            "w2": glorot_init(get_next_key(), (D_FF, D_MODEL)),  # Second linear layer
            "b2": jnp.zeros(D_MODEL),  # Second bias
        },
        # Layer normalization parameters
        "norm1": {
            "gamma": jnp.ones(D_MODEL),
            "beta": jnp.zeros(D_MODEL),
        },  # After attention
        "norm2": {"gamma": jnp.ones(D_MODEL), "beta": jnp.zeros(D_MODEL)},  # After FFN
    }


def _init_decoder_layer_params(get_next_key, glorot_init) -> dict[str, Any]:
    """Initialize parameters for a single decoder layer."""
    return {
        # Masked multi-head self-attention
        "masked_mha": {
            "w_q": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),
            "w_k": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),
            "w_v": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),
            "w_o": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),
        },
        # Cross-attention (encoder-decoder attention)
        "cross_mha": {
            "w_q": glorot_init(
                get_next_key(), (D_MODEL, D_MODEL)
            ),  # Query from decoder
            "w_k": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),  # Key from encoder
            "w_v": glorot_init(
                get_next_key(), (D_MODEL, D_MODEL)
            ),  # Value from encoder
            "w_o": glorot_init(get_next_key(), (D_MODEL, D_MODEL)),
        },
        # Feed-forward network
        "ffn": {
            "w1": glorot_init(get_next_key(), (D_MODEL, D_FF)),
            "b1": jnp.zeros(D_FF),
            "w2": glorot_init(get_next_key(), (D_FF, D_MODEL)),
            "b2": jnp.zeros(D_MODEL),
        },
        # Layer normalization parameters (decoder has 3 layer norms)
        "norm1": {
            "gamma": jnp.ones(D_MODEL),
            "beta": jnp.zeros(D_MODEL),
        },  # After masked attention
        "norm2": {
            "gamma": jnp.ones(D_MODEL),
            "beta": jnp.zeros(D_MODEL),
        },  # After cross attention
        "norm3": {"gamma": jnp.ones(D_MODEL), "beta": jnp.zeros(D_MODEL)},  # After FFN
    }


# --- 5. Data Generation ---


def create_reverse_dataset(key, batch_size):
    """
    Create a dataset for the sequence reversal task.

    Task: Given a sequence [a, b, c, d], learn to output [d, c, b, a, <END>]

    Args:
        key: JAX random key
        batch_size: Number of sequences to generate

    Returns:
        encoder_input: Source sequences (batch_size, SOURCE_SEQ_LEN)
        decoder_input: Target sequences with <START> token (batch_size, TARGET_SEQ_LEN)
        target: Expected output sequences with <END> token (batch_size, TARGET_SEQ_LEN)
    """
    # Generate random sequences (avoiding tokens 0, 1, 2 which are special)
    base_sequences = random.randint(key, (batch_size, SOURCE_SEQ_LEN), 3, VOCAB_SIZE)

    # Create reversed sequences
    reversed_sequences = jnp.fliplr(base_sequences)

    # Prepare encoder input (original sequences)
    encoder_input = base_sequences

    # Prepare decoder input: <START> + reversed_sequence
    # Token 1 = <START>, Token 2 = <END>, Token 0 = <PAD>
    start_tokens = jnp.full((batch_size, 1), 1)  # <START> token
    decoder_input = jnp.concatenate([start_tokens, reversed_sequences], axis=1)

    # Prepare target: reversed_sequence + <END>
    end_tokens = jnp.full((batch_size, 1), 2)  # <END> token
    target = jnp.concatenate([reversed_sequences, end_tokens], axis=1)

    return encoder_input, decoder_input, target


# --- 6. Optimizer and Training Step ---


def init_adamw_state(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Initialize AdamW optimizer state (momentum and velocity)."""
    m_states = jax.tree_util.tree_map(jnp.zeros_like, params)
    v_states = jax.tree_util.tree_map(jnp.zeros_like, params)
    return m_states, v_states


# CHANGED: This is a clean, explicit AdamW implementation using JAX pytrees.
# @jit
def adamw_step(params, gradients, m_states, v_states, step, learning_rate):
    """
    AdamW optimizer step with bias correction and gradient clipping.

    AdamW = Adam with decoupled weight decay:
    1. Clip gradients to prevent explosion
    2. Update momentum estimates (m, v)
    3. Apply bias correction
    4. Update parameters with weight decay
    """
    beta1, beta2, eps, weight_decay = 0.9, 0.999, 1e-8, 0.01
    max_grad_norm = 1.0  # Gradient clipping threshold

    # Clip gradients to prevent explosion (common cause of loss spikes)
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(gradients))
    )
    clip_factor = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
    clipped_gradients = jax.tree_util.tree_map(lambda g: g * clip_factor, gradients)

    # Update moments (using clipped gradients)
    updated_m = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g, m_states, clipped_gradients
    )
    updated_v = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g),
        v_states,
        clipped_gradients,
    )

    # Bias correction
    m_corrected = jax.tree_util.tree_map(lambda m: m / (1.0 - beta1**step), updated_m)
    v_corrected = jax.tree_util.tree_map(lambda v: v / (1.0 - beta2**step), updated_v)

    # Parameter update
    updated_params = jax.tree_util.tree_map(
        lambda p, m, v: p
        - learning_rate * (m / (jnp.sqrt(v) + eps) + weight_decay * p),
        params,
        m_corrected,
        v_corrected,
    )

    return updated_params, updated_m, updated_v


# @jit
def complete_training_step(
    encoder_in, decoder_in, targets, params, m_states, v_states, step
):
    """Complete JIT-compiled training step."""

    def loss_fn(params_inner):
        logits = transformer_forward(encoder_in, decoder_in, params_inner)
        return cross_entropy_loss(logits, targets)

    # Compute loss and gradients
    loss_value, param_gradients = value_and_grad(loss_fn)(params)

    # Update parameters using AdamW
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, LEARNING_RATE
    )

    return updated_params, updated_m, updated_v, loss_value


# --- 7. Inference and Main Training Loop ---


def predict_sequence(encoder_input, params):
    """
    Generate a sequence using the trained transformer (autoregressive inference).

    Args:
        encoder_input: Input sequence to reverse (seq_len,) or (1, seq_len)
        params: Trained model parameters

    Returns:
        Generated sequence including start token (TARGET_SEQ_LEN,)
    """
    # Ensure batch dimension exists
    if encoder_input.ndim == 1:
        encoder_input = encoder_input[jnp.newaxis, :]

    batch_size = encoder_input.shape[0]

    # Initialize with start token
    decoder_tokens = [jnp.full((batch_size,), 1)]  # Start with <START> token (1)

    # Generate tokens one by one (autoregressive generation)
    for position in range(1, TARGET_SEQ_LEN):
        # Build current decoder input by concatenating all generated tokens so far
        current_decoder_input = jnp.stack(decoder_tokens, axis=1)  # (batch, position)

        # Pad with zeros to reach TARGET_SEQ_LEN
        padding_length = TARGET_SEQ_LEN - current_decoder_input.shape[1]
        if padding_length > 0:
            padding = jnp.zeros(
                (batch_size, padding_length), dtype=current_decoder_input.dtype
            )
            padded_decoder_input = jnp.concatenate(
                [current_decoder_input, padding], axis=1
            )
        else:
            padded_decoder_input = current_decoder_input

        # Get model predictions
        logits = transformer_forward(encoder_input, padded_decoder_input, params)

        # Get logits for the current position we're predicting
        next_token_logits = logits[:, position - 1, :]  # (batch, vocab_size)

        # Select the most likely next token
        predicted_token = jnp.argmax(next_token_logits, axis=-1)  # (batch,)

        # Add predicted token to our sequence
        decoder_tokens.append(predicted_token)

        # Don't stop early - let the model generate the full sequence
        # The END token should naturally appear at the end if trained properly

    # Convert list of tokens to final sequence
    final_sequence = jnp.stack(decoder_tokens, axis=1)  # (batch, seq_len)

    return final_sequence[0]  # Return first (and only) sequence


def train_transformer():
    """
    Main training loop for the transformer sequence reversal task.

    This function demonstrates:
    1. Parameter initialization
    2. Training loop with JIT-compiled steps
    3. Loss monitoring
    4. Final evaluation
    """
    print("=" * 60)
    print("🤖 TRAINING TRANSFORMER FROM SCRATCH WITH JAX")
    print("=" * 60)
    print(f"📋 Task: Reverse sequences of {SOURCE_SEQ_LEN} integers")
    print(
        f"🏗️  Architecture: {NUM_LAYERS} layers, {D_MODEL} d_model, {NUM_HEADS} attention heads"
    )
    print(f"📊 Training: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
    print(
        f"🎯 Vocabulary size: {VOCAB_SIZE} (tokens 0=PAD, 1=START, 2=END, 3-{VOCAB_SIZE - 1}=content)"
    )
    print()

    # Initialize random keys
    key = random.PRNGKey(42)
    key, init_key, data_key = random.split(key, 3)

    # Initialize model parameters
    print("🔧 Initializing transformer parameters...")
    params = init_transformer_params(init_key)

    # Initialize optimizer state
    print("📈 Initializing AdamW optimizer...")
    m_states, v_states = init_adamw_state(params)

    print("✅ Setup complete! Starting training...\n")

    # Training loop
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        # Generate training batch
        data_key, subkey = random.split(data_key)
        encoder_input, decoder_input, targets = create_reverse_dataset(
            subkey, BATCH_SIZE
        )

        # Perform one training step (JIT-compiled for speed)
        params, m_states, v_states, loss = complete_training_step(
            encoder_input, decoder_input, targets, params, m_states, v_states, epoch
        )

        # Print progress (similar to Nabla style)
        if epoch % PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d} | Loss: {float(loss):.4f} | Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n✅ Training complete! Total time: {total_time:.1f}s")

    # Final evaluation
    print("\n" + "=" * 60)
    print("🧪 FINAL EVALUATION")
    print("=" * 60)

    # Test on a few examples
    test_key, subkey = random.split(key)
    num_test_examples = 3

    print("Testing on random sequences:")
    print("-" * 40)

    correct_predictions = 0
    for i in range(num_test_examples):
        subkey, test_subkey = random.split(subkey)
        test_encoder_input, _, test_target = create_reverse_dataset(test_subkey, 1)
        test_encoder_input, test_target = test_encoder_input[0], test_target[0]

        # Generate prediction
        prediction = predict_sequence(test_encoder_input, params)

        # Extract just the content part (skip start token, compare sequence + end)
        predicted_content = prediction[1:]  # Remove start token
        expected_content = test_target  # Already has content + end token

        # Check if correct
        is_correct = jnp.array_equal(predicted_content, expected_content)
        if is_correct:
            correct_predictions += 1

        print(f"Test {i + 1}:")
        print(f"  Input:           {test_encoder_input}")
        print(f"  Expected output: {expected_content}")
        print(f"  Predicted:       {predicted_content}")
        print(f"  Full prediction: {prediction}")  # Show full sequence for clarity
        print(f"  Correct:         {'✅ YES' if is_correct else '❌ NO'}")
        print()

    # Summary
    accuracy = correct_predictions / num_test_examples
    print(
        f"📊 Test Accuracy: {correct_predictions}/{num_test_examples} ({accuracy:.1%})"
    )

    if accuracy >= 0.8:
        print("🎉 SUCCESS: The transformer learned to reverse sequences!")
    else:
        print("🤔 The model needs more training or tuning.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    train_transformer()
