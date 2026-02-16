# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Multi-head attention module."""

from __future__ import annotations

from max.dtype import DType

from ...core import Tensor
from ...ops.view import reshape, swap_axes
from .. import functional as F
from .base import Module
from .linear import Linear


class MultiHeadAttention(Module):
    """Multi-head attention as described in *Attention Is All You Need*.

    Parameters
    ----------
    d_model : int
        Total model dimensionality.
    num_heads : int
        Number of parallel attention heads.  ``d_model`` must be divisible
        by ``num_heads``.
    dropout : float
        Dropout probability on attention weights (applied during training).
    bias : bool
        Whether the linear projections include bias terms.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        *,
        dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = float(dropout)

        self.q_proj = Linear(d_model, d_model, bias=bias, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, bias=bias, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, bias=bias, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, bias=bias, dtype=dtype)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Run multi-head attention.

        Parameters
        ----------
        query, key, value : Tensor ``(batch, seq_*, d_model)``
        attn_mask : optional additive mask ``(..., seq_q, seq_k)``
        is_causal : apply a causal mask

        Returns
        -------
        Tensor ``(batch, seq_q, d_model)``
        """
        batch = int(query.shape[0])
        seq_q = int(query.shape[1])
        seq_k = int(key.shape[1])

        # Project and reshape to (batch, num_heads, seq, head_dim)
        q = self._reshape_to_heads(self.q_proj(query), batch, seq_q)
        k = self._reshape_to_heads(self.k_proj(key), batch, seq_k)
        v = self._reshape_to_heads(self.v_proj(value), batch, seq_k)

        # Scaled dot-product attention per head
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            is_causal=is_causal,
            training=self._training,
        )

        # Concatenate heads: (batch, num_heads, seq_q, head_dim) -> (batch, seq_q, d_model)
        attn_out = swap_axes(attn_out, 1, 2)  # (batch, seq_q, num_heads, head_dim)
        attn_out = reshape(attn_out, (batch, seq_q, self.d_model))

        return self.out_proj(attn_out)

    def _reshape_to_heads(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        """(batch, seq, d_model) -> (batch, num_heads, seq, head_dim)"""
        x = reshape(x, (batch, seq_len, self.num_heads, self.head_dim))
        return swap_axes(x, 1, 2)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"dropout={self.dropout_p}"
        )
