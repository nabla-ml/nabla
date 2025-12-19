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

from .base import Module
from .. import functional as F

__all__ = ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "GELU"]


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)

class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)

class Tanh(Module):
    def forward(self, x):
        return F.tanh(x)

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)

class GELU(Module):
    def forward(self, x):
        return F.gelu(x)
