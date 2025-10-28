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
from typing import Iterator, OrderedDict

from ...transforms import jit
from collections import OrderedDict

from ...core.tensor import Tensor

__all__ = ["Module"]


class Module:
    """Base class for all neural network modules, inspired by PyTorch's nn.Module."""
    
    def __init__(self):
        object.__setattr__(self, '_parameters', OrderedDict())
        object.__setattr__(self, '_buffers', OrderedDict())
        object.__setattr__(self, '_modules', OrderedDict())
        object.__setattr__(self, '_training', True)
        object.__setattr__(self, '_compiled_forward', None)

    def __setattr__(self, name: str, value) -> None:
        """Intercept attribute setting to auto-register parameters and submodules."""
        if hasattr(self, '_parameters') and name in self._parameters:
            del self._parameters[name]
        elif hasattr(self, '_buffers') and name in self._buffers:
            del self._buffers[name]
        elif hasattr(self, '_modules') and name in self._modules:
            del self._modules[name]

        object.__setattr__(self, name, value)

        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Tensor):
            self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

    def register_buffer(self, name: str, tensor: Tensor | None):
        """Adds a persistent buffer to the module."""
        if 'tensor' not in self.__dict__:
             object.__setattr__(self, 'tensor', None)
        setattr(self, name, tensor)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward() method")

    def compile(self, **jit_kwargs):
        import nabla as nb

        def functional_forward(params_and_buffers, *args, **kwargs):
            return nb.nn.functional_call(self, params_and_buffers, args, kwargs)

        jitted_functional = nb.jit(functional_forward, **jit_kwargs)

        def wrapper(*args, **kwargs):
            params_and_buffers = {
                **dict(self.named_parameters()),
                **dict(self.named_buffers())
            }
            return jitted_functional(params_and_buffers, *args, **kwargs)

        return wrapper

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Tensor]:
        for _, param in self.named_parameters():
            yield param

    def named_parameters(self, prefix: str = '') -> Iterator[tuple[str, Tensor]]:
        for name, param in self._parameters.items():
            yield f"{prefix}.{name}" if prefix else name, param
        for name, module in self._modules.items():
            yield from module.named_parameters(f"{prefix}.{name}" if prefix else name)

    def buffers(self) -> Iterator[Tensor]:
        for _, buf in self.named_buffers():
            yield buf

    def named_buffers(self, prefix: str = '') -> Iterator[tuple[str, Tensor]]:
        for name, buf in self._buffers.items():
            yield f"{prefix}.{name}" if prefix else name, buf
        for name, module in self._modules.items():
            yield from module.named_buffers(f"{prefix}.{name}" if prefix else name)

    def modules(self) -> Iterator[Module]:
        yield self
        for _, module in self._modules.items():
            yield from module.modules()

    def train(self):
        self._training = True
        for module in self.modules():
            module._training = True

    def eval(self):
        self._training = False
        for module in self.modules():
            module._training = False

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def state_dict(self) -> OrderedDict[str, Tensor]:
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param
        for name, buf in self.named_buffers():
            state[name] = buf
        return state

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor]):
        for name, tensor in state_dict.items():
            path = name.split('.')
            module_to_set_on = self
            try:
                for attr in path[:-1]:
                    module_to_set_on = getattr(module_to_set_on, attr)
                setattr(module_to_set_on, path[-1], tensor)
            except AttributeError:
                # This can happen if a key in state_dict doesn't exist in the model,
                # which we allow for now. A `strict=True` mode would raise here.
                pass

    def extra_repr(self) -> str:
        return ''

    def __repr__(self) -> str:
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '\n'.join('  ' + line for line in mod_str.split('\n'))
            child_lines.append(f'({name}): {mod_str}')
        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str