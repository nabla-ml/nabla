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
from .activations import relu

__all__ = ["linear_forward", "mlp_forward"]

def linear_forward(
    x: nb.Tensor, weight: nb.Tensor, bias: nb.Tensor | None = None
) -> nb.Tensor:
    """Forward pass through a linear layer."""
    output = nb.matmul(x, weight)
    if bias is not None:
        output = output + bias
    return output


def mlp_forward(x: nb.Tensor, params: list[nb.Tensor]) -> nb.Tensor:
    """MLP forward pass through all layers."""
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nb.matmul(output, w) + b
        if i < len(params) - 2:
            output = relu(output)
    return output
