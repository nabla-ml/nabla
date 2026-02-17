# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Transformer encoder / decoder layer modules."""

from __future__ import annotations

from max.dtype import DType

from ...core import Tensor
from .. import functional as F
from .attention import MultiHeadAttention
from .base import Module
from .dropout import Dropout
from .layernorm import LayerNorm
from .linear import Linear


class TransformerEncoderLayer(Module):
    """A single Transformer encoder layer (pre-norm variant).

    Structure::

        x ─→ LayerNorm ─→ MultiHeadAttention ─→ Dropout ─→ + ─→
        │                                                    ↑
        └────────────────────────────────────────────────────┘
        x ─→ LayerNorm ─→ FFN ─→ Dropout ─→ + ─→
        │                                      ↑
        └──────────────────────────────────────┘

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    num_heads : int
        Number of attention heads.
    dim_feedforward : int
        Hidden size of the position-wise feed-forward network.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        *,
        dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout=dropout, dtype=dtype
        )
        self.linear1 = Linear(d_model, dim_feedforward, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, dtype=dtype)
        self.norm1 = LayerNorm(d_model, dtype=dtype)
        self.norm2 = LayerNorm(d_model, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        # Self-attention sub-layer with residual
        normed = self.norm1(src)
        attn_out = self.self_attn(
            normed, normed, normed, attn_mask=src_mask, is_causal=is_causal
        )
        src = src + self.dropout1(attn_out)

        # Feed-forward sub-layer with residual
        normed = self.norm2(src)
        ff_out = self.linear2(F.gelu(self.linear1(normed)))
        src = src + self.dropout2(ff_out)

        return src


class TransformerDecoderLayer(Module):
    """A single Transformer decoder layer (pre-norm variant).

    Structure::

        tgt ─→ LayerNorm ─→ Masked-Self-Attention ─→ Dropout ─→ + ─→
        tgt ─→ LayerNorm ─→ Cross-Attention(tgt, memory) ─→ Dropout ─→ + ─→
        tgt ─→ LayerNorm ─→ FFN ─→ Dropout ─→ + ─→

    Parameters
    ----------
    d_model, num_heads, dim_feedforward, dropout : same as encoder layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        *,
        dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout=dropout, dtype=dtype
        )
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout=dropout, dtype=dtype
        )
        self.linear1 = Linear(d_model, dim_feedforward, dtype=dtype)
        self.linear2 = Linear(dim_feedforward, d_model, dtype=dtype)
        self.norm1 = LayerNorm(d_model, dtype=dtype)
        self.norm2 = LayerNorm(d_model, dtype=dtype)
        self.norm3 = LayerNorm(d_model, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        # Masked self-attention
        normed = self.norm1(tgt)
        self_attn_out = self.self_attn(
            normed, normed, normed, attn_mask=tgt_mask, is_causal=is_causal
        )
        tgt = tgt + self.dropout1(self_attn_out)

        # Cross-attention over encoder memory
        normed = self.norm2(tgt)
        cross_attn_out = self.cross_attn(normed, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(cross_attn_out)

        # Feed-forward
        normed = self.norm3(tgt)
        ff_out = self.linear2(F.gelu(self.linear1(normed)))
        tgt = tgt + self.dropout3(ff_out)

        return tgt
