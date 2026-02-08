# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Any

from max.graph import TensorValue, ops

from ..base import Operation


class AsInterleavedComplexOp(Operation):
    """View a real tensor as interleaved complex."""

    @property
    def name(self) -> str:
        return "as_interleaved_complex"

    def kernel(self, args: list, kwargs: dict) -> list:
        return [ops.as_interleaved_complex(args[0])]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        # Input shape (..., 2) -> Output shape (...)
        shapes = []
        for i in range(x.num_shards):
            s = x.physical_local_shape_ints(i)
            shapes.append(s[:-1])
        
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(self, primals: list, cotangents: list, outputs: list, kwargs: dict) -> list:
        """VJP for as_interleaved_complex: view as real."""
        return [view_as_real_interleaved(cotangents[0])]


class ViewAsRealInterleavedOp(Operation):
    """View a complex tensor as real interleaved."""

    @property
    def name(self) -> str:
        return "view_as_real_interleaved"

    def kernel(self, args: list, kwargs: dict) -> list:
        # Assuming ops.view_as_real exists or similar
        return [ops.view_as_real(args[0])]

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        x = args[0]
        # Input shape (...) -> Output shape (..., 2)
        shapes = []
        for i in range(x.num_shards):
            s = x.physical_local_shape_ints(i)
            shapes.append(s + (2,))
            
        return shapes, [x.dtype] * x.num_shards, [x.device] * x.num_shards

    def vjp_rule(self, primals: list, cotangents: list, outputs: list, kwargs: dict) -> list:
        return [as_interleaved_complex(cotangents[0])]


_as_interleaved_complex_op = AsInterleavedComplexOp()
_view_as_real_interleaved_op = ViewAsRealInterleavedOp()


def as_interleaved_complex(x):
    return _as_interleaved_complex_op([x], {})[0]


def view_as_real_interleaved(x):
    return _view_as_real_interleaved_op([x], {})[0]
