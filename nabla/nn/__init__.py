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

"""Neural Network module for Nabla.

This module provides two APIs:

1. **Imperative API (PyTorch-like)**: For imperative, object-oriented programming.
   - `nn.Module`, `nn.Linear`, `nn.Sequential`

2. **Functional API (JAX-like)**: For functional programming with pure functions.
   - `nn.functional.relu`, `nn.functional.linear_forward`
   - `nn.functional.cross_entropy_loss`
"""

# Imperative API (Modules)
from . import modules

from .modules import (
    Module,
    Sequential,
    ModuleList,
    ModuleDict,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    GELU,
)

# Functional API
from . import functional

__all__ = [
    # Imperative Modules
    "Module",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "GELU",
    # Sub-packages
    "modules",
    "functional",
]
