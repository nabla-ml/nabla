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

from collections import defaultdict
import nabla as nb
from .optimizer import Optimizer
from ..nn import functional as F

__all__ = ["SGD"]


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum)."""

    def __init__(self, params, lr: float, momentum: float = 0, weight_decay: float = 0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self) -> None:
        """Performs a single optimization step."""
        original_params_and_buffers: list[nb.Tensor] = []
        symbolic_updated_params_and_buffers: list[nb.Tensor] = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                momentum_buffer = param_state.get('momentum_buffer')

                # Symbolically compute new parameter and momentum buffer
                new_param, new_buf = F.sgd_step(
                    p,
                    p.grad,
                    momentum_buffer,
                    lr=group['lr'],
                    momentum=group['momentum'],
                    weight_decay=group['weight_decay'],
                )

                # Collect original and symbolic updated tensors
                original_params_and_buffers.append(p)
                symbolic_updated_params_and_buffers.append(new_param)

                if new_buf is not None:
                    # If momentum buffer exists, add it to the lists
                    # and update the state to hold the symbolic new_buf
                    original_params_and_buffers.append(momentum_buffer)
                    symbolic_updated_params_and_buffers.append(new_buf)
                    param_state['momentum_buffer'] = new_buf

        # Realize all symbolic tensors in a single batched operation
        nb.core.graph_execution.realize_(symbolic_updated_params_and_buffers)
        print("SGD step completed: parameters and momentum buffers updated.")
        print("Original params and buffers:", original_params_and_buffers)
        print("Symbolic updated params and buffers:", symbolic_updated_params_and_buffers)
        
        # Perform in-place updates on the original parameter and momentum buffer tensors
        self._update_params_inplace(original_params_and_buffers, symbolic_updated_params_and_buffers)
