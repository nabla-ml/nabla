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

from __future__ import annotations
from collections import OrderedDict

from .base import Module

__all__ = ["Sequential", "ModuleList", "ModuleDict"]


class Sequential(Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._modules[key] = module
        else:
            for i, module in enumerate(args):
                self._modules[str(i)] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class ModuleList(Module):
    """Holds submodules in a list.
    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.
    """

    def __init__(self, modules: list[Module] | None = None):
        super().__init__()
        if modules is not None:
            for i, module in enumerate(modules):
                self._modules[str(i)] = module

    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)

    def append(self, module: Module) -> None:
        idx = len(self._modules)
        self._modules[str(idx)] = module

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ModuleList has no forward() - use it as a container in your module"
        )


class ModuleDict(Module):
    """Holds submodules in a dictionary.
    ModuleDict can be indexed like a regular Python dictionary, but modules it
    contains are properly registered, and will be visible by all Module methods.
    """

    def __init__(self, modules: dict[str, Module] | None = None):
        super().__init__()
        if modules is not None:
            for key, module in modules.items():
                self._modules[key] = module

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self._modules[key] = module

    def __len__(self) -> int:
        return len(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ModuleDict has no forward() - use it as a container in your module"
        )
