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

"""Enhanced type safety with proper protocols and enums."""

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, Protocol, Union


class ExecutionMode(Enum):
    """Execution modes for the framework."""

    EAGER = "eager"
    LAZY = "lazy"


class OperationType(Enum):
    """Types of operations."""

    UNARY = "unary"
    BINARY = "binary"
    REDUCTION = "reduction"
    VIEW = "view"
    CREATION = "creation"


# Shape type alias
Shape = tuple[int, ...]

# Better type aliases
AxisSpec = Union[int, list[int], None]
DeviceType = Literal["cpu", "gpu", "accelerator"]

# Function type aliases for operations
MaxprCallable = Callable[..., None]
VJPRule = Callable[..., list]
JVPRule = Callable[..., Any]


class Differentiable(Protocol):
    """Protocol for differentiable operations."""

    def vjp_rule(self, primals, cotangent, output): ...
    def jvp_rule(self, primals, tangents, output): ...
