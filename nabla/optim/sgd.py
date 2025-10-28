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
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # print(f"Parameter: {p.name}, shape: {p.shape}")
                # print(f"Gradient: {p.grad.to_numpy()}")

                param_state = self.state[p]
                momentum_buffer = param_state.get('momentum_buffer')

                new_param, new_buf = F.sgd_step(
                    p,
                    p.grad,
                    momentum_buffer,
                    lr=group['lr'],
                    momentum=group['momentum'],
                    weight_decay=group['weight_decay'],
                )

                # print(f"Old impl: {p.to_numpy()}")
                p._impl = new_param._impl
                # new_param.realize()
                # print(f"New param value: {new_param.to_numpy()}")

                
                # p.copy_from(new_param)
                # print(f"New impl: {p.to_numpy()}")
                # print(p.requires_grad, p.traced)

                if new_buf is not None:
                    param_state['momentum_buffer'] = new_buf
