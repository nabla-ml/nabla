"""Base operation classes for a clean OOP design."""

from __future__ import annotations

from abc import ABC, abstractmethod

from max.dtype import DType  # Added DType import
from max.graph import Value

from ..core.array import Array


class Operation(ABC):
    """Abstract base class for all operations."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forward(self, *args: Array) -> Array:  # Standardized signature
        """Forward pass - creates the result Array."""
        pass

    @abstractmethod
    def compute_output_shape(
        self, *input_shapes: tuple
    ) -> tuple:  # Standardized signature
        """Compute the output shape given input shapes."""
        pass

    @abstractmethod
    def maxpr(self, args: list[Value], output: Array) -> None:
        """MAX graph computation."""
        pass

    @abstractmethod
    def eagerxpr(self, args: list[Array], output: Array) -> None:
        """Eager computation using NumPy."""
        pass

    @abstractmethod
    def vjp_rule(
        self, primals: list[Array], cotangent: Array, output: Array
    ) -> list[Array]:
        """Vector-Jacobian product rule for reverse-mode autodiff."""
        pass

    @abstractmethod
    def jvp_rule(
        self, primals: list[Array], tangents: list[Array], output: Array
    ) -> Array:
        """Jacobian-vector product rule for forward-mode autodiff."""
        pass


class UnaryOperation(Operation):
    """Base class for unary operations."""

    def forward(self, *args: Array) -> Array:  # Compatible signature
        """Forward pass for unary operations."""
        if len(args) != 1:
            raise ValueError(f"Unary operation requires 1 argument, got {len(args)}")
        arg = args[0]

        output_shape = self.compute_output_shape(arg.shape)
        output_dtype = self.compute_output_dtype(arg)  # Use compute_output_dtype

        res = Array(
            shape=output_shape,
            dtype=output_dtype,  # Use determined output_dtype
            device=arg.device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        res.add_argument(arg)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        from .base import EAGERMODE

        if EAGERMODE:
            self.eagerxpr([arg], res)

        return res

    def compute_output_shape(
        self, *input_shapes: tuple
    ) -> tuple:  # Compatible signature
        """Default: output shape same as input shape."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Unary operation requires 1 input shape, got {len(input_shapes)}"
            )
        return input_shapes[0]

    def compute_output_dtype(self, arg: Array) -> DType:  # Added method
        """Default: output dtype same as input dtype."""
        return arg.dtype


class BinaryOperation(Operation):
    """Base class for binary operations."""

    def forward(self, *args: Array) -> Array:  # Compatible signature
        """Forward pass for binary operations."""
        if len(args) != 2:
            raise ValueError(f"Binary operation requires 2 arguments, got {len(args)}")
        arg1, arg2 = args[0], args[1]

        # Import here to avoid circular imports
        from ..ops.view import broadcast_to

        # Validate inputs
        self._validate_inputs(arg1, arg2)

        # Compute output shape and broadcast inputs
        output_shape = self.compute_output_shape(arg1.shape, arg2.shape)
        output_dtype = self.compute_output_dtype(arg1, arg2)  # Use compute_output_dtype
        arg1_broadcasted = broadcast_to(arg1, output_shape)
        arg2_broadcasted = broadcast_to(arg2, output_shape)

        res = Array(
            shape=output_shape,
            dtype=output_dtype,  # Use determined output_dtype
            device=arg1.device,
            materialize=False,
            name=self.name,
        )

        res.set_maxpr(self.maxpr)
        res.add_argument(arg1_broadcasted)
        res.add_argument(arg2_broadcasted)
        res.vjp_rule = self.vjp_rule
        res.jvp_rule = self.jvp_rule

        from .base import EAGERMODE

        if EAGERMODE:
            self.eagerxpr([arg1_broadcasted, arg2_broadcasted], res)

        return res

    def compute_output_shape(
        self, *input_shapes: tuple
    ) -> tuple:  # Compatible signature
        """Compute broadcasted output shape."""
        if len(input_shapes) != 2:
            raise ValueError(
                f"Binary operation requires 2 input shapes, got {len(input_shapes)}"
            )
        shape1, shape2 = input_shapes[0], input_shapes[1]

        from ..utils.broadcasting import get_broadcasted_shape

        return get_broadcasted_shape(shape1, shape2)

    def compute_output_dtype(self, arg1: Array, arg2: Array) -> DType:  # Added method
        """Default: output dtype same as first input dtype."""
        # Asreduce_sumes dtypes are validated to be compatible by _validate_inputs
        return arg1.dtype

    def _validate_inputs(self, arg1: Array, arg2: Array) -> None:
        """Validate binary operation inputs."""
        if not isinstance(arg1, Array) or not isinstance(arg2, Array):
            raise TypeError("Both arguments must be Array instances")
        if arg1.dtype != arg2.dtype:
            raise ValueError(f"Dtypes {arg1.dtype} and {arg2.dtype} are incompatible")
        if arg1.device != arg2.device:
            raise ValueError(
                f"Devices {arg1.device} and {arg2.device} are incompatible"
            )


class ReductionOperation(UnaryOperation):
    """Base class for reduction operations."""

    def __init__(
        self,
        name: str,
        axes: int | list[int] | None = None,
        keep_dims: bool = False,
    ):
        super().__init__(name)
        self.axes = axes
        self.keep_dims = keep_dims

    def compute_output_shape(
        self, *input_shapes: tuple
    ) -> tuple:  # Compatible signature
        """Compute output shape for reduction."""
        if len(input_shapes) != 1:
            raise ValueError(
                f"Reduction operation requires 1 input shape, got {len(input_shapes)}"
            )
        input_shape = input_shapes[0]
        return self._compute_reduction_shape(input_shape, self.axes, self.keep_dims)

    # compute_output_dtype will be inherited from UnaryOperation (i.e., same as input)

    @staticmethod
    def _compute_reduction_shape(
        input_shape: tuple, axes: int | list[int] | None, keep_dims: bool = False
    ) -> tuple:
        """Compute the output shape for a reduction operation."""
        if axes is None:
            return () if not keep_dims else (1,) * len(input_shape)

        if isinstance(axes, int):
            axes = [axes]

        # Normalize negative axes
        normalized_axes = []
        for axis in axes:
            if axis < 0:
                axis += len(input_shape)
            if axis < 0 or axis >= len(input_shape):
                raise ValueError(
                    f"Axis {axis} is out of bounds for shape {input_shape}"
                )
            normalized_axes.append(axis)

        output_shape = []
        for i, dim in enumerate(input_shape):
            if i in normalized_axes:
                if keep_dims:
                    output_shape.append(1)
            else:
                output_shape.append(dim)

        return tuple(output_shape)


class ViewOperation(UnaryOperation):
    """Base class for view operations (reshape, transpose, etc.)."""

    def __init__(self, name: str):  # Constructor takes only name
        super().__init__(name)
