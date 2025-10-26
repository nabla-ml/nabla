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

import nabla as nb
from .optimizer import Optimizer
from ..nn import functional as F

__all__ = ["Adam"]


class Adam(Optimizer):
    """Implements Adam algorithm."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self) -> None:
        """Performs a single optimization step."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = nb.zeros_like(p)
                    state['exp_avg_sq'] = nb.zeros_like(p)

                state['step'] += 1

                new_param, new_exp_avg, new_exp_avg_sq = F.adam_step(
                    p,
                    p.grad,
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state['step'],
                    lr=group['lr'],
                    beta1=group['betas'][0],
                    beta2=group['betas'][1],
                    eps=group['eps'],
                    weight_decay=group['weight_decay'],
                )

                p._impl = new_param._impl
                state['exp_avg'] = new_exp_avg
                state['exp_avg_sq'] = new_exp_avg_sq
