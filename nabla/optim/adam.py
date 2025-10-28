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

    def _step_internal(self, param_groups: list[dict], state: dict) -> tuple[list[dict], dict]:
        updated_param_groups = []
        updated_state = defaultdict(dict)

        for group in param_groups:
            updated_group = group.copy()
            updated_params_in_group = []

            for p in group['params']:
                if p.grad is None:
                    updated_params_in_group.append(p)
                    if p in state:
                        updated_state[p] = state[p]
                    continue
                
                p_state = state.get(p, {})

                # State initialization (if not already present in the input state)
                if 'step' not in p_state:
                    p_state['step'] = nb.tensor(0, dtype=nb.DType.int32) # Use nb.tensor for JIT compatibility
                    p_state['exp_avg'] = nb.zeros_like(p)
                    p_state['exp_avg_sq'] = nb.zeros_like(p)

                # Increment step for the current parameter
                current_step = p_state['step'] + 1 # This will create a new tensor for step

                new_param, new_exp_avg, new_exp_avg_sq = F.adam_step(
                    p,
                    p.grad,
                    p_state['exp_avg'],
                    p_state['exp_avg_sq'],
                    current_step, # Pass the incremented step
                    lr=group['lr'],
                    beta1=group['betas'][0],
                    beta2=group['betas'][1],
                    eps=group['eps'],
                    weight_decay=group['weight_decay'],
                )

                updated_params_in_group.append(new_param)

                updated_p_state = {} # Create a new state dict for the updated parameter
                updated_p_state['step'] = current_step
                updated_p_state['exp_avg'] = new_exp_avg
                updated_p_state['exp_avg_sq'] = new_exp_avg_sq
                updated_state[p] = updated_p_state # Associate state with the original parameter object

            updated_group['params'] = updated_params_in_group
            updated_param_groups.append(updated_group)
        
        return updated_param_groups, updated_state
