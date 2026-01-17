# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
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
    from ..tensor.api import Tensor

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


_DEVICE_SPECS_CACHE: list[driver.DeviceSpec] | None = None
_HAS_ACCELERATOR_CACHE: bool | None = None


def _get_device_specs() -> list[driver.DeviceSpec]:
    """Cached wrapper around driver.scan_available_devices."""
    global _DEVICE_SPECS_CACHE
    if _DEVICE_SPECS_CACHE is None:
        _DEVICE_SPECS_CACHE = driver.scan_available_devices()
        # Ensure CPU is always available
        if (cpu := driver.DeviceSpec.cpu()) not in _DEVICE_SPECS_CACHE:
            _DEVICE_SPECS_CACHE.append(cpu)
    return _DEVICE_SPECS_CACHE


def _has_accelerator() -> bool:
    """Cached wrapper around accelerator_count."""
    global _HAS_ACCELERATOR_CACHE
    if _HAS_ACCELERATOR_CACHE is None:
        _HAS_ACCELERATOR_CACHE = accelerator_count() > 0
    return _HAS_ACCELERATOR_CACHE


def _default_device() -> Device:
    if device := _DEFAULT_DEVICE.get(None):
        return device
    return Accelerator() if _has_accelerator() else CPU()


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


_GLOBAL_SESSION: engine.api.InferenceSession | None = None

def _session() -> engine.api.InferenceSession:
    """A single global inference session for compiling and running kernels."""
    if session := _SESSION.get(None):
        return session

    global _GLOBAL_SESSION
    if _GLOBAL_SESSION is not None:
        _SESSION.set(_GLOBAL_SESSION)
        return _GLOBAL_SESSION

    device_specs = _get_device_specs()
    devices = driver.load_devices(device_specs)
    
    session = engine.api.InferenceSession(devices=devices)
    _GLOBAL_SESSION = session
    _SESSION.set(session)
    return session


def _in_running_loop() -> bool:
    """Check whether the caller is inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
