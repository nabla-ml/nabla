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

"""Context managers and default settings for eager execution."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator
from contextvars import ContextVar
from typing import TYPE_CHECKING, TypeVar

from max import driver, engine, graph
from max.driver import (
    CPU,
    Accelerator,
    Device,
    accelerator_count,
)
from max.dtype import DType
from max.graph import TensorType

if TYPE_CHECKING:
    from .tensor import Tensor

_SESSION: ContextVar[engine.api.InferenceSession] = ContextVar("_SESSION")
_DEFAULT_DEVICE: ContextVar[Device] = ContextVar("_DEFAULT_DEVICE")
_DEFAULT_DTYPE: ContextVar[DType] = ContextVar("_DEFAULT_DTYPE")

T = TypeVar("T")


@contextlib.contextmanager
def contextvar_context(var: ContextVar[T], value: T):  # noqa: ANN201
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def _default_dtype(device: Device) -> DType:
    if dtype := _DEFAULT_DTYPE.get(None):
        return dtype
    return DType.float32 if isinstance(device, CPU) else DType.bfloat16


def _default_device() -> Device:
    if device := _DEFAULT_DEVICE.get(None):
        return device
    return Accelerator() if accelerator_count() else CPU()


def defaults(
    dtype: DType | None = None, device: Device | None = None
) -> tuple[DType, Device]:
    """Gets the default dtype and device for tensor creation."""
    device = device or _default_device()
    return (dtype or _default_dtype(device)), device


def default_device(device: Device | graph.DeviceRef):  # noqa: ANN201
    """Context manager for setting the default device for tensor creation."""
    if isinstance(device, graph.DeviceRef):
        device = device.to_device()
    return contextvar_context(_DEFAULT_DEVICE, device)


def default_dtype(dtype: DType):  # noqa: ANN201
    """Context manager for setting the default dtype for tensor creation."""
    return contextvar_context(_DEFAULT_DTYPE, dtype)


@contextlib.contextmanager
def defaults_like(like: Tensor | TensorType) -> Generator[None]:
    """Context manager setting the default dtype and device for tensor creation."""
    with default_dtype(like.dtype), default_device(like.device):
        yield


def _session() -> engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels."""
    device_specs = driver.scan_available_devices()
    if (cpu := driver.DeviceSpec.cpu()) not in device_specs:
        device_specs.append(cpu)
    devices = driver.load_devices(device_specs)
    if not (session := _SESSION.get(None)):
        _SESSION.set(session := engine.api.InferenceSession(devices=devices))
    return session


def _in_running_loop() -> bool:
    """Check whether the caller is inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
