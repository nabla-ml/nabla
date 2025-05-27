"""Enhanced type safety with proper protocols and enums."""

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, Protocol, Union

# Re-export MAX types for convenience


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
