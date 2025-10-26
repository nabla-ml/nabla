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

__all__ = [
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "gelu",
]

def relu(x: nb.Tensor) -> nb.Tensor:
    return nb.maximum(x, 0)

def leaky_relu(x: nb.Tensor, negative_slope: float = 0.01) -> nb.Tensor:
    return nb.maximum(negative_slope * x, x)

def sigmoid(x: nb.Tensor) -> nb.Tensor:
    return 1 / (1 + nb.exp(-x))

def tanh(x: nb.Tensor) -> nb.Tensor:
    return nb.tanh(x)

def softmax(x: nb.Tensor, axis: int = -1) -> nb.Tensor:
    exp_x = nb.exp(x - nb.max(x, axes=axis, keep_dims=True))
    return exp_x / nb.sum(exp_x, axes=axis, keep_dims=True)

def log_softmax(x: nb.Tensor, axis: int = -1) -> nb.Tensor:
    return x - nb.log(nb.sum(nb.exp(x), axes=axis, keep_dims=True))

def gelu(x: nb.Tensor) -> nb.Tensor:
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1 + nb.tanh((2 / 3.1415926535) ** 0.5 * (x + 0.044715 * x ** 3)))
