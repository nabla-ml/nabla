# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .base import Module


class Sequential(Module):
    """A sequential container of Modules."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                setattr(self, key, module)
        else:
            for i, module in enumerate(args):
                setattr(self, str(i), module)

    def forward(self, x: Any) -> Any:
        out = x
        for module in self._modules.values():
            out = module(out)
        return out
