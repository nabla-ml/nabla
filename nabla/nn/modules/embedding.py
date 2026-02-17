# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.dtype import DType

from ...core import Tensor
from .. import functional as F
from .base import Module


class Embedding(Module):
    """A learnable lookup table mapping integer indices to dense vectors.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary (number of rows).
    embedding_dim : int
        Dimensionality of each embedding vector.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: DType = DType.float32,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight = F.xavier_normal((num_embeddings, embedding_dim), dtype=dtype)
        weight.requires_grad_(True)
        self.weight = weight

    def forward(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.weight)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        )
