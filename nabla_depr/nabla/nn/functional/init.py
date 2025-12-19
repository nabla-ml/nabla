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

import numpy as np
import nabla as nb

__all__ = [
    "he_normal",
    "xavier_normal",
    "lecun_normal",
]

def he_normal(shape: tuple[int, ...], seed: int | None = None) -> nb.Tensor:
    if seed is not None:
        np.random.seed(seed)
    fan_in = shape[0]
    std = (2.0 / fan_in) ** 0.5
    return nb.Tensor.from_numpy(np.random.normal(0.0, std, shape).astype(np.float32))

def xavier_normal(shape: tuple[int, ...], seed: int | None = None) -> nb.Tensor:
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape[0], shape[1]
    std = (2.0 / (fan_in + fan_out)) ** 0.5
    return nb.Tensor.from_numpy(np.random.normal(0.0, std, shape).astype(np.float32))

def lecun_normal(shape: tuple[int, ...], seed: int | None = None) -> nb.Tensor:
    if seed is not None:
        np.random.seed(seed)
    fan_in = shape[0]
    std = (1.0 / fan_in) ** 0.5
    return nb.Tensor.from_numpy(np.random.normal(0.0, std, shape).astype(np.float32))
