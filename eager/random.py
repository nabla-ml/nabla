# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Provides experimental random tensor generation utilities."""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, ShapeLike, TensorType, ops

from .context import defaults
from .tensor import Tensor
from .compute_graph import GRAPH


def uniform(
    shape: ShapeLike = (),
    range: tuple[float, float] = (0, 1),
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Creates a tensor filled with random values from a uniform distribution."""
    dtype, device = defaults(dtype, device)
    tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
    with GRAPH.graph:
        result = ops.random.uniform(tensor_type, range=range)
    return Tensor(value=result)


def gaussian(
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Creates a tensor filled with random values from a Gaussian distribution."""
    dtype, device = defaults(dtype, device)
    tensor_type = TensorType(dtype, shape, device=DeviceRef.from_device(device))
    with GRAPH.graph:
        result = ops.random.gaussian(tensor_type, mean=mean, std=std)
    return Tensor(value=result)


# Alias for gaussian
normal = gaussian
