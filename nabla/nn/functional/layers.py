# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Pure functional neural-network building blocks.

Every function here is stateless and built from nabla.ops primitives so that
it is automatically differentiable.  The Module wrappers in nabla.nn.modules
delegate to these functions to avoid duplicating logic.
"""

from __future__ import annotations

import math

from ...core import Tensor
from ...ops.binary import matmul
from ...ops.comparison import greater
from ...ops.control_flow import where
from ...ops.creation import full, ones, tril, uniform, zeros_like
from ...ops.reduction import mean, reduce_sum
from ...ops.unary import rsqrt, softmax
from ...ops.view import gather, reshape, squeeze, swap_axes, unsqueeze


# ===----------------------------------------------------------------------=== #
# Core layers
# ===----------------------------------------------------------------------=== #


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a linear projection: y = x @ weight + bias."""
    out = matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def layer_norm(
    x: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    axis: int | tuple[int, ...] = -1,
) -> Tensor:
    """Apply layer normalization over one or more axes."""
    mu = mean(x, axis=axis, keepdims=True)
    centered = x - mu
    var = mean(centered * centered, axis=axis, keepdims=True)
    normalized = centered * rsqrt(var + eps)

    out = normalized
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


# ===----------------------------------------------------------------------=== #
# Dropout
# ===----------------------------------------------------------------------=== #


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Apply dropout: randomly zero elements with probability *p*.

    During evaluation (``training=False``) the input is returned unchanged.
    Uses inverted-dropout scaling so no adjustment is needed at test time.
    """
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return zeros_like(x)
    keep_prob = 1.0 - p
    mask = greater(uniform(tuple(int(d) for d in x.shape), dtype=x.dtype), p)
    return x * mask / keep_prob


# ===----------------------------------------------------------------------=== #
# Embedding
# ===----------------------------------------------------------------------=== #


def embedding(indices: Tensor, weight: Tensor) -> Tensor:
    """Look up rows of *weight* by integer *indices*.

    Parameters
    ----------
    indices : Tensor
        Integer tensor of arbitrary shape ``(*)``.
    weight : Tensor
        Embedding matrix of shape ``(num_embeddings, embedding_dim)``.

    Returns
    -------
    Tensor of shape ``(*, embedding_dim)``.
    """
    return gather(weight, indices, axis=0)


# ===----------------------------------------------------------------------=== #
# Scaled dot-product attention
# ===----------------------------------------------------------------------=== #


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
) -> Tensor:
    """Scaled dot-product attention (functional).

    Parameters
    ----------
    query : Tensor  ``(..., seq_q, d_k)``
    key   : Tensor  ``(..., seq_k, d_k)``
    value : Tensor  ``(..., seq_k, d_v)``
    attn_mask : optional additive mask broadcastable to ``(..., seq_q, seq_k)``
    dropout_p : dropout probability on attention weights (training only)
    is_causal : if True, apply a causal (lower-triangular) mask
    training  : whether we are in training mode (affects dropout)

    Returns
    -------
    Tensor of shape ``(..., seq_q, d_v)``
    """
    d_k = int(query.shape[-1])
    scale = 1.0 / math.sqrt(d_k)

    # (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k)
    scores = matmul(query, swap_axes(key, -2, -1)) * scale

    if is_causal:
        seq_q = int(query.shape[-2])
        seq_k = int(key.shape[-2])
        causal = tril(ones((seq_q, seq_k), dtype=query.dtype))
        # Where causal==0, fill with large negative
        neg_inf = full((seq_q, seq_k), -1e9, dtype=query.dtype)
        scores = where(greater(causal, 0.0), scores, neg_inf)

    if attn_mask is not None:
        scores = scores + attn_mask

    weights = softmax(scores, axis=-1)

    if dropout_p > 0.0 and training:
        weights = dropout(weights, p=dropout_p, training=True)

    return matmul(weights, value)
