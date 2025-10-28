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
from typing import Iterator
from collections import defaultdict

from ..core.tensor import Tensor
from ..core.graph_execution import realize_

__all__ = ["Optimizer"]


class Optimizer:
    """Base class for all optimizers, inspired by PyTorch's optim.Optimizer.

    Args:
        params (iterable): An iterable of parameters to optimize or dicts defining
            parameter groups.
        defaults (dict): A dict containing default values of optimization
            options (e.g. learning rate, momentum).
    """

    def __init__(self, params, defaults: dict):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups_input = list(params)
        if len(param_groups_input) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups_input[0], dict):
            param_groups_input = [{'params': param_groups_input}]

        # Directly process initial param_groups
        for param_group in param_groups_input:
            # Add default values to the param_group
            for k, v in self.defaults.items():
                param_group.setdefault(k, v)
            
            # Ensure params are in a list
            param_group['params'] = list(param_group['params'])

            # Basic checks
            if not param_group['params']:
                raise ValueError("optimizer got an empty parameter list in a group")

            self.param_groups.append(param_group)

    def step(self) -> None:
        """Performs a single optimization step."""
        # This method will be implemented by subclasses.
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None