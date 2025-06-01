# ===----------------------------------------------------------------------=== #
# Nabla 2025
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


"""Utility functions for the Nabla framework.

Utilities are imported manually in the main package since they're stable and few.
"""

# Export pytree utilities for working with nested structures
# These are now defined in core.trafos but re-exported here for backward compatibility
from ..core.trafos import (
    tree_flatten,
    tree_unflatten,
    tree_map,
)

__all__ = [
    "tree_flatten", 
    "tree_unflatten",
    "tree_map",
]
