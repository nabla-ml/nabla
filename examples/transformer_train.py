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

"""
🤖 EDUCATIONAL TRANSFORMER IMPLEMENTATION WITH NABLA

This script demonstrates a complete transformer (encoder-decoder) implementation
from scratch using only raw Nabla primitives. The goal is educational clarity
while maintaining proper performance optimizations.

LEARNING OBJECTIVES:
- Understand transformer architecture components (attention, feed-forward, layer norm)
- See how encoder-decoder models work with one-hot encoding
    # Create one-hot by comparing target indices with vocabulary indices
    one_hot_targets = nb.equal(targets_expanded, vocab_indices).astype(nb.DType.float32) for sequence-to-sequence tasks
- Learn Nabla best practices (JIT compilation, functional programming, pytrees)
- Experience end-to-end training of a neural sequence model

TASK:     print("=" * 60)
    print("🤖 TRAINING TRANSFORMER FROM SCRATCH WITH Nabla")
    print("=" * 60)
    print(f"📋 Task: Reverse sequences of {SOURCE_SEQ_LEN} integers")
    print(f"🏗️  Architecture: {NUM_LAYERS} layers, {D_MODEL} d_model, {NUM_HEADS} attention heads")
    print(f"📊 Training: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
    print(f"🎯 Vocabulary: {VOCAB_SIZE} tokens (0=PAD, 1=START, 2=END, 3-{VOCAB_SIZE-1}=content)")
    print() Reversal
- Input:  [a, b, c, d, e]
- Output: [e, d, c, b, a, <END>]

KEY FEATURES:
✅ Pure Nabla implementation (no high-level libraries)
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

import numpy as np

import nabla as nb
from nabla.nn.layers.activations import log_softmax

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
NUM_EPOCHS = 500  # Total training epochs (good balance for demonstration)
PRINT_INTERVAL = 10  # Print progress every N epochs

# ============================================================================
# TRANSFORMER BUILDING BLOCKS
# ============================================================================


def positional_encoding(max_seq_len: int, d_model: int) -> nb.Array:
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
    position = nb.ndarange((max_seq_len,)).reshape(
        (max_seq_len, 1)
    )  # Shape: (max_seq_len, 1)

    # Create dimension indices for pairs: [0, 1, 2, ..., d_model//2-1]
    half_d_model = d_model // 2
    dim_indices = nb.ndarange((half_d_model,)).reshape(
        (1, half_d_model)
    )  # Shape: (1, d_model//2)

    # Calculate scaling factors: 10000^(2i/d_model) for each dimension pair
    scaling_factors = 10000.0 ** (2.0 * dim_indices / d_model)

    # Calculate the angle for each position-dimension pair
    angles = position / scaling_factors  # Shape: (max_seq_len, d_model//2)

    # Calculate sine and cosine
    sin_vals = nb.sin(angles)  # Shape: (max_seq_len, d_model//2)
    cos_vals = nb.cos(angles)  # Shape: (max_seq_len, d_model//2)

    # Interleave sine and cosine: [sin, cos, sin, cos, ...]
    # Stack and reshape to create the correct pattern
    stacked = nb.stack(
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
    scores = nb.matmul(q, k.permute((0, 1, 3, 2))) / nb.sqrt(
        nb.array([d_k], dtype=nb.DType.float32)
    )

    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        # mask is boolean, so we need to check where mask is False (0)
        scores = nb.where(mask, scores, nb.full_like(scores, -1e9))

    # Apply softmax to get attention weights
    attention_weights = nb.softmax(scores, axis=-1)

    # Apply attention weights to values
    output = nb.matmul(attention_weights, v)
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
    q_linear = nb.matmul(x, params["w_q"])
    k_linear = nb.matmul(xa, params["w_k"])
    v_linear = nb.matmul(xa, params["w_v"])

    # Reshape and transpose for multi-head attention
    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_head)
    q = q_linear.reshape((batch_size, seq_len, num_heads, d_head)).permute((0, 2, 1, 3))
    k = k_linear.reshape((batch_size, -1, num_heads, d_head)).permute((0, 2, 1, 3))
    v = v_linear.reshape((batch_size, -1, num_heads, d_head)).permute((0, 2, 1, 3))

    # Apply scaled dot-product attention
    attention_output = scaled_dot_product_attention(q, k, v, mask)

    # Concatenate heads: (batch, num_heads, seq_len, d_head) -> (batch, seq_len, d_model)
    attention_output = attention_output.permute((0, 2, 1, 3)).reshape(
        (batch_size, seq_len, d_model)
    )

    # Final linear projection
    return nb.matmul(attention_output, params["w_o"])


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
    hidden = nb.relu(nb.matmul(x, params["w1"]) + params["b1"])

    # Second linear transformation (output layer)
    output = nb.matmul(hidden, params["w2"]) + params["b2"]

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
    mean = nb.mean(x, axes=[-1], keep_dims=True)
    variance = nb.mean((x - mean) * (x - mean), axes=[-1], keep_dims=True)

    # Normalize and apply learnable scale/shift parameters
    normalized = (x - mean) / nb.sqrt(variance + eps)
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
    normed_x = layer_norm(x, params["norm1"])
    attention_output = multi_head_attention(normed_x, normed_x, params["mha"], mask)
    x = x + attention_output

    # Pre-norm: normalize then apply FFN, then residual
    normed_x = layer_norm(x, params["norm2"])
    ffn_output = feed_forward(normed_x, params["ffn"])
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
    normed_x = layer_norm(x, params["norm1"])
    masked_attention_output = multi_head_attention(
        normed_x, normed_x, params["masked_mha"], look_ahead_mask
    )
    x = x + masked_attention_output

    # 2. Pre-norm cross-attention
    normed_x = layer_norm(x, params["norm2"])
    cross_attention_output = multi_head_attention(
        normed_x, encoder_output, params["cross_mha"], padding_mask
    )
    x = x + cross_attention_output

    # 3. Pre-norm feed-forward
    normed_x = layer_norm(x, params["norm3"])
    ffn_output = feed_forward(normed_x, params["ffn"])
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
    target_seq_len = decoder_inputs.shape[1]

    # Create proper causal mask using available Nabla operations
    # We need a lower triangular matrix where mask[i,j] = 1 if j <= i, else 0

    # Create position indices
    positions = nb.ndarange((target_seq_len,))  # [0, 1, 2, ..., seq_len-1]
    pos_i = nb.reshape(positions, (target_seq_len, 1))  # Column vector
    pos_j = nb.reshape(positions, (1, target_seq_len))  # Row vector

    # Create causal mask: allow attention to position j if j <= i
    # This is equivalent to pos_i >= pos_j
    causal_mask = nb.greater_equal(pos_i, pos_j)  # Shape: (seq_len, seq_len)

    # Add batch and head dimensions: (1, 1, seq_len, seq_len)
    look_ahead_mask = nb.reshape(causal_mask, (1, 1, target_seq_len, target_seq_len))

    # Get sequence lengths
    encoder_seq_len = encoder_inputs.shape[1]
    decoder_seq_len = decoder_inputs.shape[1]

    # --- ENCODER PROCESSING ---
    # Convert token ids to embeddings and add positional encoding
    batch_size = encoder_inputs.shape[0]
    encoder_embeddings = embedding_lookup(
        encoder_inputs, params["encoder"]["embedding"]
    )

    encoder_pos_enc = params["pos_encoding"][
        :, :encoder_seq_len, :
    ]  # (1, enc_seq_len, d_model)
    encoder_x = encoder_embeddings + encoder_pos_enc

    # Pass through encoder layers
    encoder_output = encoder_x
    for layer_idx in range(NUM_LAYERS):
        layer_params = params["encoder"][f"layer_{layer_idx}"]
        encoder_output = encoder_layer(encoder_output, layer_params, mask=None)

    # Final layer norm for encoder (Pre-Norm architecture)
    encoder_output = layer_norm(encoder_output, params["encoder"]["final_norm"])

    # --- DECODER PROCESSING ---
    # Convert token ids to embeddings and add positional encoding
    decoder_embeddings = embedding_lookup(
        decoder_inputs, params["decoder"]["embedding"]
    )
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

    # Final layer norm for decoder (Pre-Norm architecture)
    decoder_output = layer_norm(decoder_output, params["decoder"]["final_norm"])

    # Final linear projection to vocabulary size
    logits = nb.matmul(
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
    # Create one-hot encoding using basic primitives without .at/.set
    batch_size, seq_len = targets.shape
    vocab_size = logits.shape[-1]

    # Create target indices expanded to match one-hot shape
    targets_expanded = targets.reshape(
        (batch_size, seq_len, 1)
    )  # (batch_size, seq_len, 1)

    # Create vocabulary indices for comparison
    vocab_indices = nb.ndarange((vocab_size,), dtype=nb.DType.int32).reshape(
        (1, 1, vocab_size)
    )  # (1, 1, vocab_size)

    # Create one-hot by comparing target indices with vocabulary indices
    one_hot_targets = nb.equal(targets_expanded, vocab_indices).astype(nb.DType.float32)

    # Compute log probabilities and cross-entropy
    log_probs = log_softmax(logits)
    cross_entropy = -nb.sum(one_hot_targets * log_probs)

    # Average over batch size
    batch_size = logits.shape[0]
    return cross_entropy / batch_size


# --- 4. Initialization ---


def init_transformer_params() -> dict[str, Any]:
    """
    Initialize all transformer parameters using Xavier/Glorot initialization.

    Args:
        key: Nabla random key for parameter initialization

    Returns:
        Dictionary containing all model parameters organized by component
    """
    # Initialize main parameter structure with flexible typing
    params: dict[str, Any] = {"encoder": {}, "decoder": {}}

    # --- Embedding Layers ---
    print("Initializing embedding layers...")
    params["encoder"]["embedding"] = nb.randn((VOCAB_SIZE, D_MODEL))
    params["decoder"]["embedding"] = nb.randn((VOCAB_SIZE, D_MODEL))

    # --- Positional Encoding (Fixed, not trainable) ---
    params["pos_encoding"] = positional_encoding(MAX_SEQ_LEN, D_MODEL)

    # --- Initialize Encoder Layers ---
    print(f"Initializing {NUM_LAYERS} encoder layers...")
    for layer_idx in range(NUM_LAYERS):
        params["encoder"][f"layer_{layer_idx}"] = _init_encoder_layer_params()

    # Final encoder layer norm (important for Pre-Norm architecture)
    params["encoder"]["final_norm"] = {
        "gamma": nb.ones((D_MODEL,)),
        "beta": nb.zeros((D_MODEL,)),
    }

    # --- Initialize Decoder Layers ---
    print(f"Initializing {NUM_LAYERS} decoder layers...")
    for layer_idx in range(NUM_LAYERS):
        params["decoder"][f"layer_{layer_idx}"] = _init_decoder_layer_params()

    # Final decoder layer norm (important for Pre-Norm architecture)
    params["decoder"]["final_norm"] = {
        "gamma": nb.ones((D_MODEL,)),
        "beta": nb.zeros((D_MODEL,)),
    }

    # --- Output Projection Layer ---
    print("Initializing output projection layer...")
    params["output_linear"] = nb.glorot_uniform((D_MODEL, VOCAB_SIZE))

    return params


def _init_encoder_layer_params() -> dict[str, Any]:
    """Initialize parameters for a single encoder layer."""
    return {
        # Multi-head self-attention
        "mha": {
            "w_q": nb.glorot_uniform((D_MODEL, D_MODEL)),  # Query projection
            "w_k": nb.glorot_uniform((D_MODEL, D_MODEL)),  # Key projection
            "w_v": nb.glorot_uniform((D_MODEL, D_MODEL)),  # Value projection
            "w_o": nb.glorot_uniform((D_MODEL, D_MODEL)),  # Output projection
        },
        # Feed-forward network
        "ffn": {
            "w1": nb.glorot_uniform((D_MODEL, D_FF)),  # First linear layer
            "b1": nb.zeros((D_FF,)),  # First bias
            "w2": nb.glorot_uniform((D_FF, D_MODEL)),  # Second linear layer
            "b2": nb.zeros((D_MODEL,)),  # Second bias
        },
        # Layer normalization parameters
        "norm1": {
            "gamma": nb.ones((D_MODEL,)),
            "beta": nb.zeros((D_MODEL,)),
        },  # After attention
        "norm2": {
            "gamma": nb.ones((D_MODEL,)),
            "beta": nb.zeros((D_MODEL,)),
        },  # After FFN
    }


def _init_decoder_layer_params() -> dict[str, Any]:
    """Initialize parameters for a single decoder layer."""
    return {
        # Masked multi-head self-attention
        "masked_mha": {
            "w_q": nb.glorot_uniform((D_MODEL, D_MODEL)),
            "w_k": nb.glorot_uniform((D_MODEL, D_MODEL)),
            "w_v": nb.glorot_uniform((D_MODEL, D_MODEL)),
            "w_o": nb.glorot_uniform((D_MODEL, D_MODEL)),
        },
        # Cross-attention (encoder-decoder attention)
        "cross_mha": {
            "w_q": nb.glorot_uniform((D_MODEL, D_MODEL)),  # Query from decoder
            "w_k": nb.glorot_uniform((D_MODEL, D_MODEL)),  # Key from encoder
            "w_v": nb.glorot_uniform((D_MODEL, D_MODEL)),  # Value from encoder
            "w_o": nb.glorot_uniform((D_MODEL, D_MODEL)),
        },
        # Feed-forward network
        "ffn": {
            "w1": nb.glorot_uniform((D_MODEL, D_FF)),
            "b1": nb.zeros((D_FF,)),
            "w2": nb.glorot_uniform((D_FF, D_MODEL)),
            "b2": nb.zeros((D_MODEL,)),
        },
        # Layer normalization parameters (decoder has 3 layer norms)
        "norm1": {
            "gamma": nb.ones((D_MODEL,)),
            "beta": nb.zeros((D_MODEL,)),
        },  # After masked attention
        "norm2": {
            "gamma": nb.ones((D_MODEL,)),
            "beta": nb.zeros((D_MODEL,)),
        },  # After cross attention
        "norm3": {
            "gamma": nb.ones((D_MODEL,)),
            "beta": nb.zeros((D_MODEL,)),
        },  # After FFN
    }


# --- 5. Data Generation ---


def create_reverse_dataset(batch_size):
    """
    Create a dataset for the sequence reversal task using numpy then converting to nabla.

    Task: Given a sequence [a, b, c, d], learn to output [d, c, b, a, <END>]

    Args:
        batch_size: Number of sequences to generate

    Returns:
        encoder_input: Source sequences (batch_size, SOURCE_SEQ_LEN)
        decoder_input: Target sequences with <START> token (batch_size, TARGET_SEQ_LEN)
        target: Expected output sequences with <END> token (batch_size, TARGET_SEQ_LEN)
    """
    # Generate random sequences using numpy (avoiding tokens 0, 1, 2 which are special)
    base_sequences_np = np.random.randint(
        3, VOCAB_SIZE, size=(batch_size, SOURCE_SEQ_LEN), dtype=np.int32
    )

    # Create reversed sequences by flipping along axis 1
    reversed_sequences_np = np.flip(base_sequences_np, axis=1)

    # Prepare encoder input (original sequences)
    encoder_input_np = base_sequences_np

    # Prepare decoder input: <START> + reversed_sequence
    # Token 1 = <START>, Token 2 = <END>, Token 0 = <PAD>
    start_tokens_np = np.ones((batch_size, 1), dtype=np.int32)  # <START> token (1)
    decoder_input_np = np.concatenate([start_tokens_np, reversed_sequences_np], axis=1)

    # Prepare target: reversed_sequence + <END>
    end_tokens_np = np.full((batch_size, 1), 2, dtype=np.int32)  # <END> token (2)
    target_np = np.concatenate([reversed_sequences_np, end_tokens_np], axis=1)

    # Convert numpy arrays to nabla arrays
    encoder_input = nb.Array.from_numpy(encoder_input_np)
    decoder_input = nb.Array.from_numpy(decoder_input_np)
    target = nb.Array.from_numpy(target_np)

    return encoder_input, decoder_input, target


# --- 6. Optimizer and Training Step ---


def init_adamw_state(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Initialize AdamW optimizer state (momentum and velocity)."""
    m_states = {}
    v_states = {}

    # Initialize momentum and velocity states for each parameter
    def init_zeros_like(params_dict, m_dict, v_dict):
        for key, value in params_dict.items():
            if isinstance(value, dict):
                m_dict[key] = {}
                v_dict[key] = {}
                init_zeros_like(value, m_dict[key], v_dict[key])
            else:
                m_dict[key] = nb.zeros_like(value)
                v_dict[key] = nb.zeros_like(value)

    init_zeros_like(params, m_states, v_states)
    return m_states, v_states


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

    # Calculate gradient norm for clipping
    total_grad_norm_sq = 0.0

    def calculate_grad_norm(grad_dict):
        nonlocal total_grad_norm_sq
        for key, value in grad_dict.items():
            if isinstance(value, dict):
                calculate_grad_norm(value)
            else:
                total_grad_norm_sq += nb.sum(value * value)

    calculate_grad_norm(gradients)
    grad_norm = nb.sqrt(total_grad_norm_sq)
    clip_factor = nb.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))

    # Initialize output dictionaries
    updated_params = {}
    updated_m = {}
    updated_v = {}

    def update_recursive(
        params_dict,
        grad_dict,
        m_dict,
        v_dict,
        updated_params_dict,
        updated_m_dict,
        updated_v_dict,
    ):
        for key in params_dict:
            if isinstance(params_dict[key], dict):
                # Recursive case: nested dictionary
                updated_params_dict[key] = {}
                updated_m_dict[key] = {}
                updated_v_dict[key] = {}
                update_recursive(
                    params_dict[key],
                    grad_dict[key],
                    m_dict[key],
                    v_dict[key],
                    updated_params_dict[key],
                    updated_m_dict[key],
                    updated_v_dict[key],
                )
            else:
                # Base case: actual parameter tensor
                p = params_dict[key]
                g = grad_dict[key] * clip_factor  # Apply gradient clipping
                m = m_dict[key]
                v = v_dict[key]

                # Update momentum estimates
                updated_m_dict[key] = beta1 * m + (1.0 - beta1) * g
                updated_v_dict[key] = beta2 * v + (1.0 - beta2) * (g * g)

                # Bias correction
                m_corrected = updated_m_dict[key] / (1.0 - beta1**step)
                v_corrected = updated_v_dict[key] / (1.0 - beta2**step)

                # Parameter update with AdamW weight decay
                updated_params_dict[key] = p - learning_rate * (
                    m_corrected / (nb.sqrt(v_corrected) + eps) + weight_decay * p
                )

    # Perform the recursive update
    update_recursive(
        params, gradients, m_states, v_states, updated_params, updated_m, updated_v
    )

    return updated_params, updated_m, updated_v


def complete_training_step(
    encoder_in, decoder_in, targets, params, m_states, v_states, step
):
    """Complete training step."""

    def loss_fn(params_inner):
        logits = transformer_forward(encoder_in, decoder_in, params_inner)
        return cross_entropy_loss(logits, targets)

    # Compute loss and gradients
    loss_value, param_gradients = nb.value_and_grad(loss_fn)(params)

    # Update parameters using AdamW
    updated_params, updated_m, updated_v = adamw_step(
        params, param_gradients, m_states, v_states, step, LEARNING_RATE
    )

    return updated_params, updated_m, updated_v, loss_value
    # return params, m_states, v_states, loss_fn(params)


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
    if len(encoder_input.shape) == 1:
        encoder_input = encoder_input.reshape((1, encoder_input.shape[0]))

    batch_size = encoder_input.shape[0]

    # Initialize with start token
    decoder_tokens = [
        nb.ones((batch_size,), dtype=nb.DType.int32)
    ]  # Start with <START> token (1)

    # Generate tokens one by one (autoregressive generation)
    for position in range(1, TARGET_SEQ_LEN):
        # Build current decoder input by concatenating all generated tokens so far
        current_decoder_input = nb.stack(decoder_tokens, axis=1)  # (batch, position)

        # Pad with zeros to reach TARGET_SEQ_LEN
        padding_length = TARGET_SEQ_LEN - current_decoder_input.shape[1]
        if padding_length > 0:
            padding = nb.zeros(
                (batch_size, padding_length), dtype=current_decoder_input.dtype
            )
            padded_decoder_input = nb.concatenate(
                [current_decoder_input, padding], axis=1
            )
        else:
            padded_decoder_input = current_decoder_input

        # Get model predictions
        logits = transformer_forward(encoder_input, padded_decoder_input, params)

        # Get logits for the current position we're predicting
        # logits shape: (batch, seq_len, vocab_size)
        # We want the logits for the last valid position (position - 1)
        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, position - 1, :]  # (batch, vocab_size)

        # Select the most likely next token
        predicted_token = nb.argmax(next_token_logits, axes=-1)  # (batch,)

        # Cast to same dtype as start tokens (int32)
        predicted_token = predicted_token.astype(nb.DType.int32)

        # Add predicted token to our sequence
        decoder_tokens.append(predicted_token)

        # Don't stop early - let the model generate the full sequence
        # The END token should naturally appear at the end if trained properly

    # Convert list of tokens to final sequence
    final_sequence = nb.stack(decoder_tokens, axis=1)  # (batch, seq_len)

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
    print("🤖 TRAINING TRANSFORMER FROM SCRATCH WITH Nabla")
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
    # key = nb.PRNGKey(42)
    # key, init_key, data_key = nb.split(key, 3)

    # Initialize model parameters
    print("🔧 Initializing transformer parameters...")
    params = init_transformer_params()

    # Initialize optimizer state
    print("📈 Initializing AdamW optimizer...")
    m_states, v_states = init_adamw_state(params)

    print("✅ Setup complete! Starting training...\n")

    # Training loop
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        # Generate training batch
        encoder_input, decoder_input, targets = create_reverse_dataset(BATCH_SIZE)

        # Perform one training step (JIT-compiled for speed)
        params, m_states, v_states, loss = complete_training_step(
            encoder_input, decoder_input, targets, params, m_states, v_states, epoch
        )

        # Print progress
        if epoch % PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:5d} | Loss: {loss.to_numpy():.4f} | Time: {elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\n✅ Training complete! Total time: {total_time:.1f}s")

    # Final evaluation
    print("\n" + "=" * 60)
    print("🧪 FINAL EVALUATION")
    print("=" * 60)

    num_test_examples = 3

    print("Testing on random sequences:")
    print("-" * 40)

    correct_predictions = 0
    for i in range(num_test_examples):
        # subkey, test_subkey = nb.split(subkey)
        test_encoder_input, _, test_target = create_reverse_dataset(1)
        test_encoder_input, test_target = test_encoder_input[0], test_target[0]

        # Generate prediction
        prediction = predict_sequence(test_encoder_input, params)

        # Extract just the content part (skip start token, compare sequence + end)
        predicted_content = prediction[1:]  # Remove start token
        expected_content = test_target  # Already has content + end token

        # Check if correct
        is_correct = np.array_equal(
            predicted_content.to_numpy(), expected_content.to_numpy()
        )
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


def embedding_lookup(token_ids, embedding_matrix):
    """
    Perform embedding lookup using nb.where for each token.

    Args:
        token_ids: (batch_size, seq_len) - token indices
        embedding_matrix: (vocab_size, d_model) - embedding vectors

    Returns:
        embeddings: (batch_size, seq_len, d_model)
    """
    batch_size, seq_len = token_ids.shape
    vocab_size, d_model = embedding_matrix.shape

    # Initialize output with zeros
    output = nb.zeros((batch_size, seq_len, d_model))

    # For each token in vocabulary, use where to select appropriate embeddings
    for token_idx in range(vocab_size):
        # Create condition: token_ids == token_idx
        token_idx_array = nb.array([token_idx], dtype=nb.DType.int32)
        token_idx_broadcast = nb.broadcast_to(token_idx_array, token_ids.shape)
        condition = nb.equal(token_ids, token_idx_broadcast)

        # Expand condition to match embedding dimensions (batch, seq_len, d_model)
        condition_expanded = nb.broadcast_to(
            condition.reshape((batch_size, seq_len, 1)), (batch_size, seq_len, d_model)
        )

        # Get the embedding vector for this token (1, 1, d_model)
        token_embedding = embedding_matrix[token_idx : token_idx + 1, :].reshape(
            (1, 1, d_model)
        )
        token_embedding_expanded = nb.broadcast_to(
            token_embedding, (batch_size, seq_len, d_model)
        )

        # Use where to select this embedding where condition is true
        output = nb.where(condition_expanded, token_embedding_expanded, output)

    return output


if __name__ == "__main__":
    train_transformer()
