# ===----------------------------------------------------------------------=== #
# Nabla 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tracing infrastructure for VJP and other autodiff transformations."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tensor_impl import TensorImpl
    from .pytree import PyTreeDef
    from .ops import Operation


@dataclass(frozen=True)
class OutputRefs:
    """Lightweight container for multi-output operation siblings.
    
    This struct is shared among all TensorImpls produced by the same operation call.
    It serves as the single source of truth for operation metadata, eliminating
    duplication across sibling outputs.
    
    Attributes:
        _refs: Tuple of weak references to output TensorImpls.
        tree_def: PyTreeDef describing the output structure.
        op: The Operation object that produced these outputs.
        op_args: Original positional arguments (Pytrees of inputs/static values).
        op_kwargs: Original keyword arguments.
    """
    _refs: tuple[weakref.ref, ...]
    tree_def: PyTreeDef
    op: Operation
    op_args: tuple[Any, ...]
    op_kwargs: dict[str, Any] | None
    
    def __post_init__(self):
        """Validate that refs and tree_def are consistent."""
        if len(self._refs) != self.tree_def.num_leaves:
            raise ValueError(
                f"OutputRefs: ref count {len(self._refs)} doesn't match "
                f"tree_def leaves {self.tree_def.num_leaves}"
            )
    
    def get_alive_outputs(self) -> list[TensorImpl | None]:
        """Get list of output TensorImpls, with None for dead/GC'd outputs.
        
        Returns:
            List matching the flattened output structure. Contains None for
            any outputs that have been garbage collected.
        """
        return [ref() for ref in self._refs]
    
    @property
    def num_outputs(self) -> int:
        """Number of outputs (including potentially dead ones)."""
        return len(self._refs)
    
    def __repr__(self) -> str:
        alive = sum(1 for ref in self._refs if ref() is not None)
        op_name = self.op.name if hasattr(self.op, 'name') else str(self.op)
        return f"OutputRefs(op={op_name}, outputs={self.num_outputs}, alive={alive})"
