# ===----------------------------------------------------------------------=== #
# Nabla Tutorials - 05b: Transformer Training (JAX-Style)
# ===----------------------------------------------------------------------=== #
"""Transformer Training with Nabla's Functional (JAX-Style) API.

This tutorial builds the same Transformer encoder as 05a, but in a purely
functional style — no nn.Module, just dicts of parameters and pure functions:
- Explicit parameter initialization (pytree dicts)
- Pure-function transformer layers
- Functional scaled dot-product attention
- Training with value_and_grad + adamw_update
"""

# %% [markdown]
# # Tutorial 5b: Transformer Training (JAX-Style / Functional)
#
# This tutorial builds the same sequence classification Transformer as 05a,
# but without nn.Module — everything is **pure functions** operating on
# **parameter dicts** (pytrees).
#
# This style is closer to JAX/Flax and shows Nabla's functional flexibility.

# %%
import numpy as np

import nabla as nb
import nabla.nn.functional as F

print("Nabla Transformer Training — JAX-style (functional)")

# %% [markdown]
# ## 1. Parameter Initialization
#
# We initialize parameters as nested dicts. Each tensor that needs gradients
# has `requires_grad = True`.

# %%
def init_linear(in_dim: int, out_dim: int) -> dict:
    """Initialize a linear layer: {weight, bias}."""
    params = {
        "weight": F.xavier_normal((in_dim, out_dim)),
        "bias": nb.zeros((1, out_dim)),
    }
    for p in params.values():
        p.requires_grad = True
    return params


def init_layer_norm(dim: int) -> dict:
    """Initialize layer norm: {weight, bias}."""
    params = {
        "weight": nb.ones((dim,)),
        "bias": nb.zeros((dim,)),
    }
    for p in params.values():
        p.requires_grad = True
    return params


def init_mha(d_model: int) -> dict:
    """Initialize multi-head attention projections."""
    return {
        "q_proj": init_linear(d_model, d_model),
        "k_proj": init_linear(d_model, d_model),
        "v_proj": init_linear(d_model, d_model),
        "out_proj": init_linear(d_model, d_model),
    }


def init_encoder_layer(d_model: int, dim_ff: int) -> dict:
    """Initialize one Transformer encoder layer."""
    return {
        "attn": init_mha(d_model),
        "norm1": init_layer_norm(d_model),
        "norm2": init_layer_norm(d_model),
        "ff1": init_linear(d_model, dim_ff),
        "ff2": init_linear(dim_ff, d_model),
    }


def init_transformer(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    num_classes: int,
    max_len: int,
    dim_ff: int,
) -> dict:
    """Initialize all transformer parameters."""
    # Embedding
    emb_weight = F.xavier_normal((vocab_size, d_model))
    emb_weight.requires_grad = True

    # Positional encoding (fixed, not learned)
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)

    params = {
        "embedding": emb_weight,
        "layers": [init_encoder_layer(d_model, dim_ff) for _ in range(num_layers)],
        "classifier": init_linear(d_model, num_classes),
        "_num_heads": num_heads,
    }
    # Store PE as a non-differentiable constant
    params["pe"] = nb.Tensor.from_dlpack(pe)
    params["pe"].requires_grad = False

    return params

# %% [markdown]
# ## 2. Pure Function Layers
#
# Each layer is a pure function: `output = layer(params, input)`.

# %%
def linear(params: dict, x):
    """Functional linear layer."""
    return x @ params["weight"] + params["bias"]


def layer_norm(params: dict, x, eps: float = 1e-5):
    """Functional layer normalization."""
    return F.layer_norm(x, weight=params["weight"], bias=params["bias"], eps=eps)


def multi_head_attention(params: dict, x, num_heads: int):
    """Functional multi-head self-attention.

    Args:
        params: Dict with q_proj, k_proj, v_proj, out_proj.
        x: Input tensor (batch, seq_len, d_model).
        num_heads: Number of attention heads.
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    d_model = x.shape[2]
    head_dim = d_model // num_heads

    # Project to Q, K, V
    q = linear(params["q_proj"], x)  # (batch, seq, d_model)
    k = linear(params["k_proj"], x)
    v = linear(params["v_proj"], x)

    # Reshape to (batch, num_heads, seq, head_dim)
    q = nb.reshape(q, (batch_size, seq_len, num_heads, head_dim))
    q = nb.permute(q, (0, 2, 1, 3))
    k = nb.reshape(k, (batch_size, seq_len, num_heads, head_dim))
    k = nb.permute(k, (0, 2, 1, 3))
    v = nb.reshape(v, (batch_size, seq_len, num_heads, head_dim))
    v = nb.permute(v, (0, 2, 1, 3))

    # Scaled dot-product attention
    attn_out = F.scaled_dot_product_attention(q, k, v, training=False)

    # Reshape back: (batch, seq, d_model)
    attn_out = nb.permute(attn_out, (0, 2, 1, 3))
    attn_out = nb.reshape(attn_out, (batch_size, seq_len, d_model))

    # Output projection
    return linear(params["out_proj"], attn_out)


def encoder_layer(params: dict, x, num_heads: int):
    """Functional Transformer encoder layer (pre-norm)."""
    # Self-attention with residual
    normed = layer_norm(params["norm1"], x)
    attn_out = multi_head_attention(params["attn"], normed, num_heads)
    x = x + attn_out

    # FFN with residual
    normed = layer_norm(params["norm2"], x)
    ff_out = linear(params["ff2"], nb.gelu(linear(params["ff1"], normed)))
    x = x + ff_out

    return x


def transformer_forward(params: dict, token_ids):
    """Full transformer forward pass.

    Args:
        params: Nested parameter dict from init_transformer.
        token_ids: Integer tensor (batch, seq_len).

    Returns:
        Logits of shape (batch, num_classes).
    """
    num_heads = params["_num_heads"]

    # Token embedding + positional encoding
    x = F.embedding(token_ids, params["embedding"])
    seq_len = token_ids.shape[-1]
    d_model = int(x.shape[-1])
    pe = nb.slice_tensor(params["pe"], start=(0, 0), size=(seq_len, d_model))
    x = x + pe

    # Encoder layers
    for layer_params in params["layers"]:
        x = encoder_layer(layer_params, x, num_heads)

    # Mean pooling + classify
    x = nb.mean(x, axis=-2)
    return linear(params["classifier"], x)

# %% [markdown]
# ## 3. Create Data

# %%
np.random.seed(42)

vocab_size = 20
seq_len = 8
num_classes = 3
n_samples = 150
d_model = 32
num_heads = 4
num_layers = 2
dim_ff = 64

# Random token sequences, labels = (sum of tokens) mod num_classes
token_ids_np = np.random.randint(0, vocab_size, (n_samples, seq_len)).astype(np.int64)
labels_np = (token_ids_np.sum(axis=1) % num_classes).astype(np.int64)
labels_onehot_np = np.zeros((n_samples, num_classes), dtype=np.float32)
labels_onehot_np[np.arange(n_samples), labels_np] = 1.0

token_ids = nb.Tensor.from_dlpack(token_ids_np)
labels = nb.Tensor.from_dlpack(labels_onehot_np)

print(f"Dataset: {n_samples} sequences of length {seq_len}")
print(f"Vocab: {vocab_size}, Classes: {num_classes}")

# %% [markdown]
# ## 4. Initialize Model and Optimizer

# %%
params = init_transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    num_classes=num_classes,
    max_len=seq_len,
    dim_ff=dim_ff,
)

opt_state = nb.nn.optim.adamw_init(params)

# Count parameters
from nabla import tree_leaves, Tensor
n_params = sum(p.numel() for p in tree_leaves(params) if isinstance(p, Tensor) and p.requires_grad)
print(f"Model: {num_layers} layers, d_model={d_model}, heads={num_heads}")
print(f"Total trainable parameters: {n_params}")

# %% [markdown]
# ## 5. Training Loop

# %%
def loss_fn(params, tokens, targets):
    logits = transformer_forward(params, tokens)
    return nb.nn.functional.cross_entropy_loss(logits, targets)


lr = 1e-3
num_epochs = 60

print(f"\n{'Epoch':<8} {'Loss':<12} {'Accuracy':<10}")
print("-" * 32)

for epoch in range(num_epochs):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(params, token_ids, labels)
    params, opt_state = nb.nn.optim.adamw_update(
        params, grads, opt_state, lr=lr
    )

    if (epoch + 1) % 10 == 0:
        logits = transformer_forward(params, token_ids)
        pred_classes = nb.argmax(logits, axis=-1)
        target_classes = nb.Tensor.from_dlpack(labels_np.astype(np.int64))
        correct = nb.equal(pred_classes, target_classes)
        accuracy = nb.mean(nb.cast(correct, nb.DType.float32)).item()
        print(f"{epoch + 1:<8} {loss.item():<12.4f} {accuracy:<10.2%}")

# %% [markdown]
# ## 6. Compiled Training (Bonus)

# %%
params2 = init_transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    num_classes=num_classes,
    max_len=seq_len,
    dim_ff=dim_ff,
)
opt_state2 = nb.nn.optim.adamw_init(params2)


@nb.compile
def compiled_step(params, opt_state, tokens, targets):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(params, tokens, targets)
    params, opt_state = nb.nn.optim.adamw_update(
        params, grads, opt_state, lr=1e-3
    )
    return params, opt_state, loss


print(f"\nCompiled training:")
print(f"{'Step':<8} {'Loss':<12}")
print("-" * 22)

for step in range(30):
    params2, opt_state2, loss = compiled_step(
        params2, opt_state2, token_ids, labels
    )

    if (step + 1) % 10 == 0:
        print(f"{step + 1:<8} {loss.item():<12.4f}")

# %% [markdown]
# ## Summary
#
# The functional style decomposes the Transformer into pure functions:
# - `init_transformer(...)` → parameter pytree
# - `transformer_forward(params, input)` → logits
# - `loss_fn(params, ...)` → scalar loss
# - `value_and_grad(loss_fn)` → (loss, gradient pytree)
# - `adamw_update(params, grads, ...)` → (new_params, new_opt_state)
#
# No mutation, no hidden state — everything flows through function arguments.
#
# **Congratulations!** You've completed all the Nabla tutorials. You're now
# equipped to build, train, and optimize ML models with Nabla's dual API.

# %%
print("\n✅ Tutorial 05b completed!")
print("🎉 All tutorials complete!")
