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
Container modules for organizing neural network components.

Provides PyTorch-like container classes:
- Sequential: Apply modules in sequence
- ModuleList: Hold modules in a list with proper registration
- ModuleDict: Hold modules in a dict with proper registration
"""

from __future__ import annotations

from .module import Module

__all__ = ["Sequential", "ModuleList", "ModuleDict"]


class Sequential(Module):
    """
    Sequential container that applies modules in order.
    
    Modules will be added to the container in the order they are passed
    in the constructor. The forward() method automatically chains them.
    
    Args:
        *modules: Variable number of modules to add sequentially
        
    Example:
        >>> model = Sequential(
        ...     Linear(10, 20),
        ...     ReLU(),
        ...     Linear(20, 10)
        ... )
        >>> output = model(input)  # Automatically applies layers in order
    """
    
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[f'layer_{i}'] = module
            # Also set as attribute for easy access
            object.__setattr__(self, f'layer_{i}', module)
    
    def forward(self, x):
        """
        Apply all modules sequentially.
        
        Args:
            x: Input tensor
            
        Returns:
            Output after applying all modules
        """
        for module in self._modules.values():
            x = module(x)
        return x
    
    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, module in enumerate(self._modules.values()):
            lines.append(f"  ({i}): {repr(module)}")
        lines.append(")")
        return '\n'.join(lines)


class ModuleList(Module):
    """
    Container that holds modules in a list.
    
    Like PyTorch's nn.ModuleList - modules are properly registered and can
    be indexed, iterated, and appended to. The modules are registered as
    submodules so their parameters are collected.
    
    Note: ModuleList does not define forward() - it's a container that you
    use within your own modules.
    
    Args:
        *modules: Variable number of modules to add to the list
        
    Example:
        >>> layers = ModuleList([
        ...     Linear(10, 20),
        ...     Linear(20, 10)
        ... ])
        >>> for layer in layers:
        ...     x = layer(x)
    """
    
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
    
    def __getitem__(self, idx: int) -> Module:
        """
        Access module by index.
        
        Args:
            idx: Index of the module
            
        Returns:
            Module at the given index
        """
        return self._modules[str(idx)]
    
    def __len__(self) -> int:
        """
        Return number of modules.
        
        Returns:
            Number of modules in the list
        """
        return len(self._modules)
    
    def __iter__(self):
        """
        Iterate over modules.
        
        Yields:
            Modules in order
        """
        return iter(self._modules.values())
    
    def append(self, module: Module) -> None:
        """
        Append a module to the end of the list.
        
        Args:
            module: Module to append
        """
        idx = len(self._modules)
        self._modules[str(idx)] = module
    
    def forward(self, *args, **kwargs):
        """
        ModuleList doesn't define forward - use it as a container.
        
        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "ModuleList has no forward() - use it as a container in your module"
        )
    
    def __repr__(self) -> str:
        lines = ["ModuleList("]
        for i, module in enumerate(self._modules.values()):
            lines.append(f"  ({i}): {repr(module)}")
        lines.append(")")
        return '\n'.join(lines)


class ModuleDict(Module):
    """
    Container that holds modules in a dictionary.
    
    Like PyTorch's nn.ModuleDict - modules are properly registered with
    string keys. Can be accessed, iterated, and modified like a dict.
    
    Note: ModuleDict does not define forward() - it's a container that you
    use within your own modules.
    
    Args:
        modules: Optional dict of modules to initialize with
        
    Example:
        >>> components = ModuleDict({
        ...     'encoder': Linear(10, 5),
        ...     'decoder': Linear(5, 10)
        ... })
        >>> encoded = components['encoder'](x)
        >>> decoded = components['decoder'](encoded)
    """
    
    def __init__(self, modules: dict[str, Module] | None = None):
        super().__init__()
        if modules:
            for key, module in modules.items():
                self._modules[key] = module
    
    def __getitem__(self, key: str) -> Module:
        """
        Access module by key.
        
        Args:
            key: Key of the module
            
        Returns:
            Module associated with the key
        """
        return self._modules[key]
    
    def __setitem__(self, key: str, module: Module) -> None:
        """
        Set module by key.
        
        Args:
            key: Key to associate with the module
            module: Module to add
        """
        self._modules[key] = module
    
    def __len__(self) -> int:
        """
        Return number of modules.
        
        Returns:
            Number of modules in the dict
        """
        return len(self._modules)
    
    def __iter__(self):
        """
        Iterate over keys.
        
        Yields:
            Keys in the dict
        """
        return iter(self._modules.keys())
    
    def keys(self):
        """
        Return module keys.
        
        Returns:
            Keys view
        """
        return self._modules.keys()
    
    def values(self):
        """
        Return modules.
        
        Returns:
            Modules view
        """
        return self._modules.values()
    
    def items(self):
        """
        Return (key, module) pairs.
        
        Returns:
            Items view
        """
        return self._modules.items()
    
    def forward(self, *args, **kwargs):
        """
        ModuleDict doesn't define forward - use it as a container.
        
        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "ModuleDict has no forward() - use it as a container in your module"
        )
    
    def __repr__(self) -> str:
        lines = ["ModuleDict("]
        for key, module in self._modules.items():
            lines.append(f"  ({key}): {repr(module)}")
        lines.append(")")
        return '\n'.join(lines)
