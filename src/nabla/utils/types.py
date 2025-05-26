"""Enhanced type safety with proper protocols and enums."""

from typing import Protocol, Union, Literal, Tuple, List, Callable, Any
from enum import Enum

# Re-export MAX types for convenience
from max.dtype import DType
from max.driver import Device, CPU, Accelerator


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
Shape = Tuple[int, ...]

# Better type aliases
AxisSpec = Union[int, List[int], None]
DeviceType = Literal["cpu", "gpu", "accelerator"]

# Function type aliases for operations
MaxprCallable = Callable[..., None]
VJPRule = Callable[..., List]
JVPRule = Callable[..., Any]


class Differentiable(Protocol):
    """Protocol for differentiable operations."""

    def vjp_rule(self, primals, cotangent, output): ...
    def jvp_rule(self, primals, tangents, output): ...
