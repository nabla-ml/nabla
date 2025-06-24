"""
🚀 CPU-OPTIMIZED TRANSFORMER (PYTORCH 2.0) - JAX COMPARISON EDITION

This version is optimized for CPU-only performance to fairly compare with JAX:
- CPU-only execution (no GPU dependencies)
- torch.compile with max-autotune mode for peak CPU performance
- Shared embedding weights between encoder/decoder
- Weight tying between embedding and output projection
- Bias-free linear layers for reduced computation
- SiLU activation (faster than ReLU on modern hardware)
- On-demand data generation (no pre-allocation)
- Optimized MultiheadAttention (need_weights=False)
- Gradient clipping and proper regularization
- Multi-threaded CPU utilization

This represents the peak CPU performance achievable with PyTorch 2.0.
"""

import math
import time

import torch
import torch.nn as nn

# ============================================================================
# CONFIGURATION
# ============================================================================
VOCAB_SIZE, SOURCE_SEQ_LEN, TARGET_SEQ_LEN, MAX_SEQ_LEN = 20, 9, 10, 11
NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT = 2, 64, 4, 128, 0.1
BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PRINT_INTERVAL = 64, 0.0005, 500, 10

# Force CPU usage for fair comparison with JAX
device = torch.device("cpu")
print("🚀 Using CPU for fair comparison with JAX")

# Optimize for CPU performance
torch.set_num_threads(torch.get_num_threads())  # Use all available CPU cores
torch.set_float32_matmul_precision("high")  # Use optimized matmul precision
torch.set_float32_matmul_precision("high")  # Use TensorFloat-32 on Ampere GPUs


# ============================================================================
# BUILDING BLOCKS and MODEL (with the crucial fix in `forward`)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_slice = self.pe[:, : x.size(1), :]  # type: ignore
        x = x + pe_slice
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Use SiLU (Swish) activation which is faster than ReLU on modern hardware
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),  # Remove bias for efficiency
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True, bias=False
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Pre-norm architecture for better gradient flow
        norm_src = self.norm1(src)
        attn_out, _ = self.self_attn(norm_src, norm_src, norm_src, need_weights=False)
        src = src + self.dropout(attn_out)

        norm_src = self.norm2(src)
        ff_out = self.feed_forward(norm_src)
        src = src + self.dropout(ff_out)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True, bias=False
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True, bias=False
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask):
        # Pre-norm architecture with optimized attention calls
        norm_tgt = self.norm1(tgt)
        self_attn_out, _ = self.self_attn(
            norm_tgt, norm_tgt, norm_tgt, attn_mask=tgt_mask, need_weights=False
        )
        tgt = tgt + self.dropout(self_attn_out)

        norm_tgt = self.norm2(tgt)
        cross_attn_out, _ = self.cross_attn(
            norm_tgt, memory, memory, need_weights=False
        )
        tgt = tgt + self.dropout(cross_attn_out)

        norm_tgt = self.norm3(tgt)
        ff_out = self.feed_forward(norm_tgt)
        tgt = tgt + self.dropout(ff_out)
        return tgt


class ManualPreNormTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout):
        super().__init__()
        # Shared embedding for encoder and decoder (common optimization)
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_SEQ_LEN)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.encoder_norm, self.decoder_norm = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )

        # Tie output projection with embedding weights (standard optimization)
        self.output_linear = nn.Linear(d_model, vocab_size, bias=False)
        self.output_linear.weight = self.shared_embedding.weight

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1).to(device)

    def forward(self, src, tgt):
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))

        # Use shared embedding
        src_emb = self.pos_encoder(self.shared_embedding(src))
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        encoder_output = self.encoder_norm(encoder_output)

        tgt_emb = self.pos_encoder(self.shared_embedding(tgt))
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask)
        decoder_output = self.decoder_norm(decoder_output)

        return self.output_linear(decoder_output)


# Data, inference, and compiled training step setup optimized for performance
def create_reverse_dataset(batch_size, device):
    """Optimized data generation matching JAX approach."""
    # Use torch.randint directly on device for efficiency
    base_seq = torch.randint(
        3, VOCAB_SIZE, (batch_size, SOURCE_SEQ_LEN), dtype=torch.long, device=device
    )
    encoder_input = base_seq

    # Use torch.flip instead of numpy
    reversed_seq = torch.flip(base_seq, dims=[1])

    # Use torch.ones/torch.full directly on device
    start_token = torch.ones((batch_size, 1), dtype=torch.long, device=device)
    end_token = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)

    target = torch.cat([reversed_seq, end_token], dim=1)
    decoder_input = torch.cat([start_token, target[:, :-1]], dim=1)

    return encoder_input, decoder_input, target


def predict_sequence(model, encoder_input, max_len=TARGET_SEQ_LEN):
    model.eval()
    if encoder_input.dim() == 1:
        encoder_input = encoder_input.unsqueeze(0)
    decoder_tokens = torch.ones((1, 1), dtype=torch.long, device=device) * 1
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(encoder_input, decoder_tokens)
            predicted_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            decoder_tokens = torch.cat([decoder_tokens, predicted_token], dim=1)
            if predicted_token.item() == 2:
                break
    return decoder_tokens.squeeze(0)


def get_training_step_fn(model, optimizer, criterion):
    def training_step():
        # Generate data on-demand like JAX version for better memory efficiency
        encoder_input, decoder_input, targets = create_reverse_dataset(
            BATCH_SIZE, device
        )

        optimizer.zero_grad()
        logits = model(encoder_input, decoder_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss

    return training_step


def train_transformer():
    print("=" * 60)
    print("🚀 CPU-OPTIMIZED TRANSFORMER (PYTORCH 2.0) - JAX COMPARISON")
    print("=" * 60)

    model = ManualPreNormTransformer(
        VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        reduction="mean"
    )  # Use mean instead of sum for consistency

    # Use AdamW with optimized settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,  # Add weight decay for regularization
        eps=1e-8,
        betas=(0.9, 0.999),
    )

    print("🔥 Compiling the entire training step function with torch.compile...")
    training_step_fn = get_training_step_fn(model, optimizer, criterion)

    try:
        # Use default mode - no autotuning bullshit like JAX
        compiled_training_step = torch.compile(training_step_fn, mode="default")
        print(
            "✅ Training step compiled successfully with default mode (no autotuning)!"
        )
    except Exception as e:
        print(f"⚠️ Could not compile training step: {e}. Running in eager mode.")
        compiled_training_step = training_step_fn

    print("🔥 Warming up the JIT compiler...")
    model.train()
    for i in range(5):
        _ = compiled_training_step()
    print("✅ Warmup complete! Starting timed training loop.\n")

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        loss = compiled_training_step()
        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"Epoch {epoch + 1:5d} | Loss: {loss.item():.4f}")

    total_time = time.time() - start_time
    print(
        f"\n✅ Training complete! Total time for {NUM_EPOCHS} epochs: {total_time:.1f}s"
    )

    # Evaluation...
    # (The rest of the script is unchanged)
    print("\n" + "=" * 60)
    print("🧪 FINAL EVALUATION")
    print("=" * 60)
    correct_predictions = 0
    num_test_examples = 3
    for i in range(num_test_examples):
        test_encoder_input, _, test_target = create_reverse_dataset(1, device)
        prediction = predict_sequence(model, test_encoder_input)
        expected_output = test_target.squeeze(0)
        predicted_content = prediction[1:]
        len_exp, len_pred = len(expected_output), len(predicted_content)
        if len_exp > len_pred:
            predicted_content = torch.cat(
                [
                    predicted_content,
                    torch.zeros(len_exp - len_pred, dtype=torch.long, device=device),
                ]
            )
        elif len_pred > len_exp:
            predicted_content = predicted_content[:len_exp]
        is_correct = torch.equal(expected_output, predicted_content)
        if is_correct:
            correct_predictions += 1
        print(f"Test {i + 1}:")
        print(f"  Input:           {test_encoder_input.squeeze(0).tolist()}")
        print(f"  Expected output: {expected_output.tolist()}")
        print(f"  Predicted output:{predicted_content.tolist()}")
        print(f"  Correct:         {'✅ YES' if is_correct else '❌ NO'}")
        print()
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
