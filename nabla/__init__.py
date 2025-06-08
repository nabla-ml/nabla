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

"""
Nabla: Dynamic Neural Networks and Function Transformations in Python 🐍 + Mojo 🔥
"""

# Imports used within this file
import sys
from typing import Any

# Re-exports: imported here to make them available when importing from nabla
from max.dtype import DType

from .core.array import Array
from .core.trafos import (
    grad,
    jacfwd,
    jacrev,
    jit,
    jvp,
    vjp,
    vmap,
    xpr,
)
from .ops.operation import (
    BinaryOperation,
    Operation,
    ReductionOperation,
    UnaryOperation,
    ViewOperation,
)
from .utils.max_interop import accelerator, cpu, device, device_ref


# Lazy loading for operations (imported on first access)
def _build_ops_registry():
    """Build the operations registry from __all__ definitions in modules."""
    import importlib  # Import locally where it's used

    registry = {}

    # Define the ops modules to scan
    ops_modules = [
        "nabla.ops.binary",
        "nabla.ops.unary",
        "nabla.ops.creation",
        "nabla.ops.view",
        "nabla.ops.linalg",
        "nabla.ops.reduce",
    ]

    for module_name in ops_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "__all__"):
                for func_name in module.__all__:
                    registry[func_name] = (module_name, func_name)
        except ImportError:
            # Skip modules that can't be imported
            continue

    return registry


_ops_registry = _build_ops_registry()

# Cache for lazily loaded operations
_ops_cache = {}


def __getattr__(name: str) -> Any:
    """
    Lazy loading of operations using __getattr__.

    This is called when an attribute is not found in the module.
    It allows us to import operations only when they're first accessed.
    """
    if name in _ops_registry:
        if name not in _ops_cache:
            module_name, attr_name = _ops_registry[name]
            try:
                import importlib  # Import locally where it's used

                module = importlib.import_module(module_name)
                _ops_cache[name] = getattr(module, attr_name)
            except (ImportError, AttributeError) as e:
                raise AttributeError(
                    f"Cannot import {name} from {module_name}: {e}"
                ) from e
        return _ops_cache[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Build the __all__ list
__all__ = [
    # Core framework
    "Array",
    # Operations
    "Operation",
    "UnaryOperation",
    "BinaryOperation",
    "ReductionOperation",
    "ViewOperation",
    # Transformations
    "xpr",
    "vjp",
    "jvp",
    "vmap",
    "jit",
    "jacrev",
    "jacfwd",
    "grad",
    # Utilities
    "device",
    "cpu",
    "accelerator",
    "device_ref",
    # Types
    "DType",
] + list(_ops_registry.keys())  # Add all operation names


# For test compatibility - provide a reference to this module
graph_improved = sys.modules[__name__]
