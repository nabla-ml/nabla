# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""PyTorch-style transformer training on a toy copy task.

Demonstrates the imperative training loop:
  model.forward() → loss.backward() → optimizer.step()

Uses nabla's stateful AdamW optimizer with automatic realization of
updated parameters after each step.
"""

from __future__ import annotations

import math

import numpy as np

import nabla as nb

from .conftest import make_rng

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOCAB = 12
SEQ = 6
BATCH = 16
D_MODEL = 32
HEADS = 4
DFF = 64
LR = 3e-3
STEPS = 10


# ---------------------------------------------------------------------------
# Positional encoding (computed once as a nabla buffer)
# ---------------------------------------------------------------------------


def sinusoidal_pe(seq_len: int, d_model: int) -> nb.Tensor:
    """Sinusoidal positional encoding returned as a nabla Tensor."""
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    div = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return nb.Tensor.from_dlpack(pe)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Transformer(nb.nn.Module):
    """Encoder-decoder transformer for the copy task."""

    def __init__(self) -> None:
        super().__init__()
        self.src_embed = nb.nn.Embedding(VOCAB, D_MODEL)
        self.tgt_embed = nb.nn.Embedding(VOCAB, D_MODEL)
        self.register_buffer("pe", sinusoidal_pe(SEQ, D_MODEL))

        self.enc1 = nb.nn.TransformerEncoderLayer(D_MODEL, HEADS, DFF, dropout=0.0)
        self.enc2 = nb.nn.TransformerEncoderLayer(D_MODEL, HEADS, DFF, dropout=0.0)
        self.dec1 = nb.nn.TransformerDecoderLayer(D_MODEL, HEADS, DFF, dropout=0.0)
        self.dec2 = nb.nn.TransformerDecoderLayer(D_MODEL, HEADS, DFF, dropout=0.0)
        self.head = nb.nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, src: nb.Tensor, tgt: nb.Tensor) -> nb.Tensor:
        s = self.src_embed(src) + self.pe
        t = self.tgt_embed(tgt) + self.pe
        mem = self.enc2(self.enc1(s))
        out = self.dec2(self.dec1(t, mem, is_causal=True), mem, is_causal=True)
        return self.head(out)


class GPT(nb.nn.Module):
    """Decoder-only causal LM for next-token prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nb.nn.Embedding(VOCAB, D_MODEL)
        self.register_buffer("pe", sinusoidal_pe(SEQ, D_MODEL))
        self.layer1 = nb.nn.TransformerEncoderLayer(D_MODEL, HEADS, DFF, dropout=0.0)
        self.layer2 = nb.nn.TransformerEncoderLayer(D_MODEL, HEADS, DFF, dropout=0.0)
        self.head = nb.nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x: nb.Tensor) -> nb.Tensor:
        h = self.embed(x) + self.pe
        h = self.layer2(self.layer1(h, is_causal=True), is_causal=True)
        return self.head(h)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _random_tokens(rng: np.random.Generator) -> nb.Tensor:
    """Random token ids in [1, VOCAB) — nabla has no randint yet."""
    return nb.Tensor.from_dlpack(
        rng.integers(1, VOCAB, size=(BATCH, SEQ)).astype(np.int64)
    )


class TestEncoderDecoderCopyTask:
    """Train an encoder-decoder transformer using .backward() + AdamW."""

    def test_loss_decreases(self):
        rng = make_rng(1000)
        model = Transformer()

        params = dict(model.named_parameters())
        optimizer = nb.nn.optim.AdamW(params, lr=LR)

        losses: list[float] = []
        for i in range(STEPS):
            print(f"[PT copy] step {i + 1}/{STEPS} start", flush=True)
            src = _random_tokens(rng)
            tgt = _random_tokens(rng)
            # Copy task: target == source
            tgt = src

            logits = model(src, tgt)
            loss = nb.nn.functional.cross_entropy_loss(logits, tgt, axis=-1)
            loss_value = float(loss.to_numpy())
            print(f"[PT copy] step {i + 1}/{STEPS} loss={loss_value:.6f}", flush=True)
            losses.append(loss_value)

            model.zero_grad()
            loss.backward()

            # Collect gradients into a matching dict
            grads = {
                name: p.grad if p.grad is not None else nb.zeros_like(p)
                for name, p in model.named_parameters()
            }
            updated = optimizer.step(grads)

            # Write back — requires_grad_ so Module stores them as _parameters
            for v in updated.values():
                v.requires_grad_(True)
            model.load_state_dict(updated)
            optimizer.params = dict(model.named_parameters())

        assert np.mean(losses[-5:]) < np.mean(losses[:5])


class TestDecoderOnlyNextToken:
    """Train a GPT model using .backward() + AdamW."""

    def test_loss_decreases(self):
        rng = make_rng(2000)
        model = GPT()

        params = dict(model.named_parameters())
        optimizer = nb.nn.optim.AdamW(params, lr=LR)

        losses: list[float] = []
        for i in range(STEPS):
            print(f"[PT gpt] step {i + 1}/{STEPS} start", flush=True)
            tokens = _random_tokens(rng)
            logits = model(tokens)

            # Predict next token: logits[:, :-1] → tokens[:, 1:]
            pred = nb.slice_tensor(
                logits, start=(0, 0, 0), size=(BATCH, SEQ - 1, VOCAB)
            )
            tgt = nb.slice_tensor(tokens, start=(0, 1), size=(BATCH, SEQ - 1))
            loss = nb.nn.functional.cross_entropy_loss(pred, tgt, axis=-1)
            loss_value = float(loss.to_numpy())
            print(f"[PT gpt] step {i + 1}/{STEPS} loss={loss_value:.6f}", flush=True)
            losses.append(loss_value)

            model.zero_grad()
            loss.backward()

            grads = {
                name: p.grad if p.grad is not None else nb.zeros_like(p)
                for name, p in model.named_parameters()
            }
            updated = optimizer.step(grads)
            for v in updated.values():
                v.requires_grad_(True)
            model.load_state_dict(updated)
            optimizer.params = dict(model.named_parameters())

        assert np.mean(losses[-5:]) < np.mean(losses[:5])


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)

    t1 = TestEncoderDecoderCopyTask()
    print("Running TestEncoderDecoderCopyTask (PyTorch)...")
    t1.test_loss_decreases()

    t2 = TestDecoderOnlyNextToken()
    print("\nRunning TestDecoderOnlyNextToken (PyTorch)...")
    t2.test_loss_decreases()
    print("\nAll PyTorch tests passed!")

    faulthandler.cancel_dump_traceback_later()
