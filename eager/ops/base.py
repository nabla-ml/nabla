"""Base class for all operations in the computation graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from max import graph
    from ..tensor import TensorImpl


class Operation(ABC):
    """Base class for all operations in the computation graph.
    
    Each operation defines:
    - name: Unique identifier (e.g., 'add', 'matmul', 'sum')
    - maxpr(): Symbolic lowering to MAX graph (the ONLY execution path)
    - vjp_rule(): Reverse-mode autodiff (optional)
    - jvp_rule(): Forward-mode autodiff (optional)
    - sharding_rule(): For distributed execution (optional)
    
    Batch dimensions are always the prefix of the physical shape.
    In maxpr(), translate logical axes to physical by adding batch_dims:
        physical_axis = logical_axis + batch_dims (for positive axes)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique operation name."""
        ...
    
    @abstractmethod
    def maxpr(self, *args: TensorImpl, **kwargs: Any) -> graph.TensorValue:
        """Symbolic lowering to MAX graph operations.
        
        This is the ONLY execution path - we are always lazy.
        
        When handling axes parameters, translate from logical to physical:
            physical_axis = logical_axis + tensor.batch_dims
        
        Args:
            *args: Input TensorImpls
            **kwargs: Operation-specific parameters
            
        Returns:
            The resulting TensorValue in the MAX graph
        """
        ...
    
    # ===== Autodiff Rules (Optional) =====
    
    def vjp_rule(
        self, 
        primals: list[TensorImpl], 
        cotangent: TensorImpl,
    ) -> list[TensorImpl | None]:
        """Vector-Jacobian product for reverse-mode autodiff."""
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement vjp_rule"
        )
    
    def jvp_rule(
        self,
        primals: list[TensorImpl],
        tangents: list[TensorImpl | None],
    ) -> tuple[TensorImpl, TensorImpl | None]:
        """Jacobian-vector product for forward-mode autodiff."""
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement jvp_rule"
        )
    
    # ===== Sharding (Optional) =====
    
    def sharding_rule(
        self,
        input_shardings: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Determine output sharding from input shardings.
        
        For distributed execution across multiple devices.
        """
        raise NotImplementedError(
            f"Operation '{self.name}' does not implement sharding_rule"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
