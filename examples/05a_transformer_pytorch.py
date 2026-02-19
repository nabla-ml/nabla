# ===----------------------------------------------------------------------=== #
# Nabla Tutorials - 05a: Transformer Training (PyTorch-Style)
# ===----------------------------------------------------------------------=== #
"""Transformer Training with Nabla's PyTorch-Style API.

This tutorial builds and trains a Transformer encoder for sequence
classification using Nabla's nn.Module system:
- Custom Transformer model with nn.MultiHeadAttention
- Positional encoding
- Training with cross-entropy loss
- Using @nb.compile for speed
"""

# %% [markdown]
# # Tutorial 5a: Transformer Training (PyTorch-Style)
#
# We'll build a small Transformer encoder for a synthetic **sequence
# classification** task: given a sequence of token embeddings, predict which
# class it belongs to.
#
# The model uses Nabla's built-in `TransformerEncoderLayer`, `Embedding`,
# and `MultiHeadAttention` modules.

# %%
import numpy as np

import nabla as nb

print("Nabla Transformer Training — PyTorch-style")

# %% [markdown]
# ## 1. Positional Encoding
#
# We'll use sinusoidal positional encoding, computed as a fixed buffer.

# %%
def make_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe  # (max_len, d_model)

# %% [markdown]
# ## 2. Define the Model
#
# A small Transformer encoder with:
# - Learned token embeddings
# - Sinusoidal positional encoding (fixed buffer)
# - N Transformer encoder layers
# - Mean pooling + classification head

# %%
class TransformerClassifier(nb.nn.Module):
    """Transformer encoder for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        max_len: int = 128,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.embedding = nb.nn.Embedding(vocab_size, d_model)

        # Positional encoding (fixed, not learned)
        pe_np = make_positional_encoding(max_len, d_model)
        self.pe = nb.Tensor.from_dlpack(pe_np)
        self.pe.requires_grad = False

        # Transformer encoder layers
        self.layers = []
        for i in range(num_layers):
            layer = nb.nn.TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=0.0,  # No dropout for deterministic training
            )
            setattr(self, f"encoder_{i}", layer)
            self.layers.append(layer)

        # Classification head
        self.classifier = nb.nn.Linear(d_model, num_classes)

    def forward(self, token_ids):
        """Forward pass.

        Args:
            token_ids: Integer tensor of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Embed tokens
        x = self.embedding(token_ids)  # (batch, seq_len, d_model)

        # Add positional encoding (sliced to sequence length)
        seq_len = token_ids.shape[-1]
        pe_slice = nb.slice_tensor(
            self.pe, start=(0, 0), size=(seq_len, self.d_model)
        )
        x = x + pe_slice

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x)

        # Mean pooling over sequence dimension
        x = nb.mean(x, axis=-2)  # (batch, d_model)

        # Classify
        return self.classifier(x)  # (batch, num_classes)

# %% [markdown]
# ## 3. Create Synthetic Data
#
# Generate a simple classification task:
# - Sequences of random token IDs
# - Labels based on a rule (e.g., majority token determines class)

# %%
np.random.seed(42)

vocab_size = 20
seq_len = 8
num_classes = 3
n_samples = 150
d_model = 32
num_heads = 4
num_layers = 2

# Generate random token sequences
token_ids_np = np.random.randint(0, vocab_size, (n_samples, seq_len)).astype(np.int64)

# Labels: class = (sum of tokens) mod num_classes
labels_np = (token_ids_np.sum(axis=1) % num_classes).astype(np.int64)

# One-hot encode labels
labels_onehot_np = np.zeros((n_samples, num_classes), dtype=np.float32)
labels_onehot_np[np.arange(n_samples), labels_np] = 1.0

token_ids = nb.Tensor.from_dlpack(token_ids_np)
labels = nb.Tensor.from_dlpack(labels_onehot_np)

print(f"Dataset: {n_samples} sequences of length {seq_len}")
print(f"Vocab size: {vocab_size}, Classes: {num_classes}")
print(f"Sample tokens: {token_ids_np[0]}")
print(f"Sample label:  {labels_np[0]}")

# %% [markdown]
# ## 4. Build Model and Optimizer

# %%
model = TransformerClassifier(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    max_len=seq_len,
    dim_feedforward=64,
)
model.eval()  # Disable dropout

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {num_layers} encoder layers, d_model={d_model}, heads={num_heads}")
print(f"Total parameters: {n_params}")

opt_state = nb.nn.optim.adamw_init(model)

# %% [markdown]
# ## 5. Training Loop

# %%
def loss_fn(model, tokens, targets):
    logits = model(tokens)
    return nb.nn.functional.cross_entropy_loss(logits, targets)


num_epochs = 60
lr = 1e-3

print(f"\n{'Epoch':<8} {'Loss':<12} {'Accuracy':<10}")
print("-" * 32)

for epoch in range(num_epochs):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(model, token_ids, labels)
    model, opt_state = nb.nn.optim.adamw_update(
        model, grads, opt_state, lr=lr
    )

    if (epoch + 1) % 10 == 0:
        # Compute accuracy
        logits = model(token_ids)
        pred_classes = nb.argmax(logits, axis=-1)
        target_classes = nb.Tensor.from_dlpack(labels_np.astype(np.int64))
        correct = nb.equal(pred_classes, target_classes)
        accuracy = nb.mean(nb.cast(correct, nb.DType.float32)).item()
        print(f"{epoch + 1:<8} {loss.item():<12.4f} {accuracy:<10.2%}")

# %% [markdown]
# ## 6. Compiled Training (Bonus)
#
# For maximum performance, wrap the training step in `@nb.compile`.

# %%
model2 = TransformerClassifier(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    max_len=seq_len,
    dim_feedforward=64,
)
model2.eval()
opt_state2 = nb.nn.optim.adamw_init(model2)


@nb.compile
def compiled_step(model, opt_state, tokens, targets):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(model, tokens, targets)
    model, opt_state = nb.nn.optim.adamw_update(
        model, grads, opt_state, lr=1e-3
    )
    return model, opt_state, loss


print(f"\nCompiled training:")
print(f"{'Step':<8} {'Loss':<12}")
print("-" * 22)

for step in range(30):
    model2, opt_state2, loss = compiled_step(model2, opt_state2, token_ids, labels)

    if (step + 1) % 10 == 0:
        print(f"{step + 1:<8} {loss.item():<12.4f}")

print("\nCompiled step executes forward + backward + optimizer in a single MAX graph!")

# %% [markdown]
# ## Summary
#
# | Component | API |
# |-----------|-----|
# | Token embedding | `nb.nn.Embedding(vocab_size, d_model)` |
# | Transformer layer | `nb.nn.TransformerEncoderLayer(d_model, heads, ff_dim)` |
# | Multi-head attention | `nb.nn.MultiHeadAttention(d_model, heads)` |
# | Cross-entropy | `nb.nn.functional.cross_entropy_loss(logits, targets)` |
# | Compiled training | `@nb.compile` on the full train step |
#
# **Next:** [05b_transformer_jax](05b_transformer_jax)
# — The same Transformer, built in a purely functional style.

# %%
print("\n✅ Tutorial 05a completed!")
