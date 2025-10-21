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

"""Base Module class for imperative neural network programming (PyTorch-like API).

This module provides the foundational Module class that enables:
- Automatic parameter registration
- Recursive parameter collection
- Submodule management
- Gradient handling
- Callable models

Examples
--------
>>> import nabla as nb
>>> from nabla.nn import Module
>>> class Linear(Module):
...     def __init__(self, in_features, out_features):
...         super().__init__()
...         weight = nb.glorot_uniform((in_features, out_features))
...         weight.requires_grad_(True)
...         self.weight = weight  # Auto-registered!
...     def forward(self, x):
...         return nb.matmul(x, self.weight)
"""

from __future__ import annotations

from typing import Iterator

from ..core.tensor import Tensor

__all__ = ["Module"]


class Module:
    """Base class for all neural network modules (PyTorch-like nn.Module).
    
    Your models should subclass this class and implement the forward() method.
    
    Automatically tracks:
    - Parameters (Tensors with requires_grad=True)
    - Submodules (nested Module instances)
    
    Provides:
    - Recursive parameter access via .parameters()
    - Named parameter iteration via .named_parameters()
    - Module tree iteration via .modules()
    - Gradient zeroing via .zero_grad()
    - Callable interface: model(x) calls model.forward(x)
    
    Examples
    --------
    >>> from nabla.nn import Module, Linear
    >>> class MLP(Module):
    ...     def __init__(self, layer_sizes):
    ...         super().__init__()
    ...         self.layers = [Linear(layer_sizes[i], layer_sizes[i+1])
    ...                       for i in range(len(layer_sizes)-1)]
    ...     def forward(self, x):
    ...         for layer in self.layers:
    ...             x = layer(x)
    ...         return x
    >>> model = MLP([10, 20, 10])
    >>> params = list(model.parameters())  # Gets all params recursively
    """
    
    def __init__(self):
        """Initialize the Module with empty parameter and submodule registries."""
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})
    
    def __setattr__(self, name: str, value) -> None:
        """
        Intercept attribute setting to auto-register parameters and submodules.
        
        - If value is a Tensor with requires_grad=True, register as parameter
        - If value is a Module, register as submodule  
        - If value is a list of Modules, auto-wrap in ModuleList (if imported)
        
        Args:
            name: Attribute name
            value: Attribute value
        """
        # Check if it's a parameter (Tensor with requires_grad)
        if isinstance(value, Tensor):
            if getattr(value, 'requires_grad', None) is True:
                self._parameters[name] = value
        # Check if it's a submodule
        elif isinstance(value, Module):
            self._modules[name] = value
        # Check if it's a list of modules - auto-wrap in ModuleList
        elif isinstance(value, list) and len(value) > 0:
            # Import here to avoid circular dependency
            if all(isinstance(item, Module) for item in value):
                from .containers import ModuleList
                module_list = ModuleList(*value)
                self._modules[name] = module_list
                value = module_list
        
        # Always set as normal attribute
        object.__setattr__(self, name, value)
    
    def parameters(self) -> Iterator[Tensor]:
        """Recursively yield all parameters from this module and submodules.
        
        This is the primary way to get all trainable parameters for optimization.
        
        Returns
        -------
        Iterator[Tensor]
            Iterator over all parameters
            
        Examples
        --------
        >>> from nabla.nn import SGD
        >>> model = MyModel()
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        """
        # Yield own parameters
        for param in self._parameters.values():
            yield param
        
        # Recursively yield submodule parameters
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self, prefix: str = '') -> Iterator[tuple[str, Tensor]]:
        """Recursively yield (name, parameter) pairs with hierarchical names.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix to prepend to parameter names (used internally for recursion)
            
        Returns
        -------
        Iterator[tuple[str, Tensor]]
            Iterator over (name, parameter) tuples
            
        Examples
        --------
        >>> for name, param in model.named_parameters():
        ...     print(f"{name}: shape {param.shape}")
        layer1.weight: shape (10, 20)
        layer1.bias: shape (1, 20)
        layer2.weight: shape (20, 10)
        """
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield (full_name, param)
        
        for name, module in self._modules.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(full_name)
    
    def modules(self) -> Iterator[Module]:
        """Recursively yield all modules including self.
        
        Returns
        -------
        Iterator[Module]
            Iterator over all modules in the tree
            
        Examples
        --------
        >>> for module in model.modules():
        ...     print(module.__class__.__name__)
        """
        yield self
        for module in self._modules.values():
            yield from module.modules()
    
    def forward(self, *args, **kwargs):
        """
        Define the forward pass computation.
        
        Must be overridden by subclasses.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Output of the forward pass
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward() method"
        )
    
    def __call__(self, *args, **kwargs):
        """
        Make the module callable. Calls forward() method.
        
        This allows: output = model(input) instead of model.forward(input)
        
        Args:
            *args: Positional arguments passed to forward()
            **kwargs: Keyword arguments passed to forward()
            
        Returns:
            Output of forward()
        """
        return self.forward(*args, **kwargs)
    
    def zero_grad(self) -> None:
        """Zero gradients for all parameters in this module and submodules.
        
        Should be called before each backward pass during training.
        
        Examples
        --------
        >>> model.zero_grad()  # Clear all gradients
        >>> loss.backward()    # Compute new gradients
        >>> optimizer.step()   # Update parameters
        """
        for param in self.parameters():
            param.grad = None
    
    def __repr__(self) -> str:
        """
        String representation showing module structure.
        
        Returns:
            String representation of the module
        """
        lines = [f"{self.__class__.__name__}("]
        
        # Show submodules
        for name, module in self._modules.items():
            module_str = repr(module)
            # Indent nested modules
            module_str = '\n'.join('  ' + line for line in module_str.split('\n'))
            lines.append(f"  ({name}): {module_str}")
        
        # Show parameters
        for name in self._parameters.keys():
            lines.append(f"  ({name}): Parameter")
        
        lines.append(")")
        return '\n'.join(lines)
