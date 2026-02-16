# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import inspect
import types
from collections.abc import Mapping
from types import FunctionType
from typing import Any

from ... import ops
from ...core import Tensor
from ...ops.creation import gaussian, uniform
from .base import Module


def _make_cell(value: Any):
    return (lambda x: lambda: x)(value).__closure__[0]


def _rebind_class_closure(fn: FunctionType, new_cls: type[Any]) -> FunctionType:
    if "__class__" not in fn.__code__.co_freevars or fn.__closure__ is None:
        return fn

    new_cells = list(fn.__closure__)
    class_index = fn.__code__.co_freevars.index("__class__")
    new_cells[class_index] = _make_cell(new_cls)

    rebound = types.FunctionType(
        fn.__code__,
        fn.__globals__,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=tuple(new_cells),
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__doc__ = fn.__doc__
    rebound.__qualname__ = fn.__qualname__
    rebound.__module__ = fn.__module__
    rebound.__annotations__ = dict(fn.__annotations__)
    if hasattr(fn, "__signature__"):
        rebound.__signature__ = fn.__signature__
    return rebound


def _list_named_parameters(self: Any, prefix: str = ""):
    for index, child in enumerate(self):
        if isinstance(child, Module):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from child.named_parameters(child_prefix)


def _list_parameters(self: Any):
    for _, param in _list_named_parameters(self):
        yield param


def _list_named_buffers(self: Any, prefix: str = ""):
    for index, child in enumerate(self):
        if isinstance(child, Module):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from child.named_buffers(child_prefix)


def _list_buffers(self: Any):
    for _, buf in _list_named_buffers(self):
        yield buf


def _list_modules(self: Any):
    yield self
    for child in self:
        if isinstance(child, Module):
            yield from child.modules()


class NablaFunctionalShim:
    """Small shim mimicking the subset of `max.functional` used by adapted modules."""

    @staticmethod
    def functional(fn: Any) -> Any:
        return fn

    @staticmethod
    def add(lhs: Any, rhs: Any) -> Tensor:
        return ops.add(lhs, rhs)

    @staticmethod
    def sub(lhs: Any, rhs: Any) -> Tensor:
        return ops.sub(lhs, rhs)

    @staticmethod
    def mul(lhs: Any, rhs: Any) -> Tensor:
        return ops.mul(lhs, rhs)

    @staticmethod
    def div(lhs: Any, rhs: Any) -> Tensor:
        return ops.div(lhs, rhs)

    @staticmethod
    def matmul(lhs: Any, rhs: Any) -> Tensor:
        return ops.matmul(lhs, rhs)

    @staticmethod
    def outer(lhs: Any, rhs: Any) -> Tensor:
        return ops.outer(lhs, rhs)

    @staticmethod
    def cos(x: Any) -> Tensor:
        return ops.cos(x)

    @staticmethod
    def sin(x: Any) -> Tensor:
        return ops.sin(x)

    @staticmethod
    def relu(x: Any) -> Tensor:
        return ops.relu(x)

    @staticmethod
    def sigmoid(x: Any) -> Tensor:
        return ops.sigmoid(x)

    @staticmethod
    def tanh(x: Any) -> Tensor:
        return ops.tanh(x)

    @staticmethod
    def gelu(x: Any) -> Tensor:
        return ops.gelu(x)

    @staticmethod
    def silu(x: Any) -> Tensor:
        return ops.silu(x)

    @staticmethod
    def softmax(x: Any, axis: int = -1) -> Tensor:
        return ops.softmax(x, axis=axis)

    @staticmethod
    def logsoftmax(x: Any, axis: int = -1) -> Tensor:
        return ops.logsoftmax(x, axis=axis)

    @staticmethod
    def reshape(x: Any, shape: Any) -> Tensor:
        return ops.reshape(x, shape)

    @staticmethod
    def permute(x: Any, dims: list[int]) -> Tensor:
        return ops.permute(x, dims)

    @staticmethod
    def stack(values: Any, axis: int = 0) -> Tensor:
        return ops.stack(values, axis=axis)

    @staticmethod
    def broadcast_to(x: Any, shape: Any) -> Tensor:
        return ops.broadcast_to(x, shape)

    @staticmethod
    def constant(value: Any, dtype: Any = None, device: Any = None) -> Tensor:
        return ops.constant(value, dtype=dtype, device=device)

    @staticmethod
    def gather(input_tensor: Any, indices: Any, axis: int) -> Tensor:
        return ops.gather(input_tensor, indices, axis=axis)

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(
            f"NablaFunctionalShim does not implement `F.{name}` yet. "
            "Add this mapping before adapting this MAX module."
        )


class NablaRandomShim:
    """Small shim mimicking the subset of `max.random` used by adapted modules."""

    @staticmethod
    def normal(shape: Any, dtype: Any = None, device: Any = None) -> Tensor:
        t = gaussian(shape, dtype=dtype, device=device)
        t.requires_grad_(True)
        return t

    @staticmethod
    def uniform(shape: Any, dtype: Any = None, device: Any = None) -> Tensor:
        t = uniform(shape, dtype=dtype, device=device)
        t.requires_grad_(True)
        return t


def _remap_annotations(annotations: dict[str, Any], global_map: Mapping[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in annotations.items():
        if value is global_map.get("_SRC_TENSOR"):
            remapped[key] = Tensor
        elif value is global_map.get("_SRC_MODULE"):
            remapped[key] = Module
        else:
            remapped[key] = value
    return remapped


def _adapt_function(fn: FunctionType, global_overrides: Mapping[str, Any]) -> FunctionType:
    source_fn = inspect.unwrap(fn)
    new_globals = dict(source_fn.__globals__)
    new_globals.update(global_overrides)
    adapted = types.FunctionType(
        source_fn.__code__,
        new_globals,
        name=fn.__name__,
        argdefs=source_fn.__defaults__,
        closure=source_fn.__closure__,
    )
    adapted.__kwdefaults__ = source_fn.__kwdefaults__
    adapted.__doc__ = fn.__doc__
    adapted.__qualname__ = fn.__qualname__
    adapted.__module__ = fn.__module__
    adapted.__annotations__ = _remap_annotations(dict(fn.__annotations__), global_overrides)
    return adapted


def _wrap_init_with_module_bootstrap(init_fn: FunctionType) -> FunctionType:
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self, "_parameters"):
            Module.__init__(self)
        return init_fn(self, *args, **kwargs)

    wrapped.__name__ = init_fn.__name__
    wrapped.__qualname__ = init_fn.__qualname__
    wrapped.__module__ = init_fn.__module__
    wrapped.__doc__ = init_fn.__doc__
    wrapped.__annotations__ = dict(init_fn.__annotations__)
    wrapped.__defaults__ = init_fn.__defaults__
    wrapped.__kwdefaults__ = init_fn.__kwdefaults__
    wrapped.__signature__ = inspect.signature(init_fn)
    return wrapped


def adapt_max_module_class(
    cls: type[Any],
    *,
    source_module_base: type[Any] | None = None,
    base_overrides: Mapping[type[Any], type[Any]] | None = None,
    class_name: str | None = None,
    global_overrides: Mapping[str, Any] | None = None,
) -> type[Module]:
    """Adapt a MAX-style module class into a Nabla module class.

    This function clones methods while remapping key globals (`Tensor`, `Module`,
    `F`, `random`) so existing module code can run with Nabla tensors/ops.
    """
    if not inspect.isclass(cls):
        raise TypeError("`cls` must be a class.")

    overrides: dict[str, Any] = {
        "Tensor": Tensor,
        "Module": Module,
        "F": NablaFunctionalShim(),
        "random": NablaRandomShim(),
    }
    if global_overrides:
        overrides.update(dict(global_overrides))

    source_tensor = cls.__dict__.get("__annotations__", {}).get("weight", None)
    if source_tensor is not None:
        overrides.setdefault("_SRC_TENSOR", source_tensor)
    if source_module_base is not None:
        overrides.setdefault("_SRC_MODULE", source_module_base)

    resolved_base_overrides: dict[type[Any], type[Any]] = {}
    if base_overrides:
        resolved_base_overrides.update(dict(base_overrides))
    if source_module_base is not None and source_module_base not in resolved_base_overrides:
        resolved_base_overrides[source_module_base] = Module

    bases: list[type[Any]] = []
    for base in cls.__bases__:
        if base in resolved_base_overrides:
            bases.append(resolved_base_overrides[base])
        else:
            bases.append(base)

    namespace: dict[str, Any] = {
        "__module__": cls.__module__,
        "__doc__": cls.__doc__,
    }

    for name, value in cls.__dict__.items():
        if name in {"__dict__", "__weakref__", "__module__", "__doc__"}:
            continue
        if isinstance(value, FunctionType):
            namespace[name] = _adapt_function(value, overrides)
        elif isinstance(value, staticmethod):
            namespace[name] = staticmethod(_adapt_function(value.__func__, overrides))
        elif isinstance(value, classmethod):
            namespace[name] = classmethod(_adapt_function(value.__func__, overrides))
        elif isinstance(value, property):
            fget = _adapt_function(value.fget, overrides) if value.fget else None
            fset = _adapt_function(value.fset, overrides) if value.fset else None
            fdel = _adapt_function(value.fdel, overrides) if value.fdel else None
            namespace[name] = property(fget, fset, fdel, value.__doc__)
        else:
            namespace[name] = value

    new_name = class_name or f"Nabla{cls.__name__}"
    adapted_cls = type(new_name, tuple(bases), namespace)

    for attr_name, attr_value in list(vars(adapted_cls).items()):
        if isinstance(attr_value, FunctionType):
            setattr(adapted_cls, attr_name, _rebind_class_closure(attr_value, adapted_cls))
        elif isinstance(attr_value, staticmethod):
            rebound = _rebind_class_closure(attr_value.__func__, adapted_cls)
            setattr(adapted_cls, attr_name, staticmethod(rebound))
        elif isinstance(attr_value, classmethod):
            rebound = _rebind_class_closure(attr_value.__func__, adapted_cls)
            setattr(adapted_cls, attr_name, classmethod(rebound))
        elif isinstance(attr_value, property):
            fget = _rebind_class_closure(attr_value.fget, adapted_cls) if attr_value.fget else None
            fset = _rebind_class_closure(attr_value.fset, adapted_cls) if attr_value.fset else None
            fdel = _rebind_class_closure(attr_value.fdel, adapted_cls) if attr_value.fdel else None
            setattr(adapted_cls, attr_name, property(fget, fset, fdel, attr_value.__doc__))

    init_attr = vars(adapted_cls).get("__init__")
    if isinstance(init_attr, FunctionType):
        setattr(adapted_cls, "__init__", _wrap_init_with_module_bootstrap(init_attr))

    if not issubclass(adapted_cls, Module):
        adapted_cls = type(new_name, (Module, adapted_cls), {})

    if issubclass(adapted_cls, list):
        if "named_parameters" not in vars(adapted_cls):
            setattr(adapted_cls, "named_parameters", _list_named_parameters)
        if "parameters" not in vars(adapted_cls):
            setattr(adapted_cls, "parameters", _list_parameters)
        if "named_buffers" not in vars(adapted_cls):
            setattr(adapted_cls, "named_buffers", _list_named_buffers)
        if "buffers" not in vars(adapted_cls):
            setattr(adapted_cls, "buffers", _list_buffers)
        if "modules" not in vars(adapted_cls):
            setattr(adapted_cls, "modules", _list_modules)

    return adapted_cls


def adapt_max_nn_core(max_nn: Any, *, class_prefix: str = "Nabla") -> dict[str, type[Module]]:
    """Adapt a core subset of real `max.nn` classes into Nabla classes.

    Returns adapted classes for: `Linear`, `Embedding`, `ModuleList`, `Sequential`.
    """
    if not hasattr(max_nn, "Module"):
        raise TypeError("`max_nn` must expose a `Module` class.")

    max_module_base = max_nn.Module

    adapted_linear = adapt_max_module_class(
        max_nn.Linear,
        source_module_base=max_module_base,
        class_name=f"{class_prefix}Linear",
    )
    adapted_embedding = adapt_max_module_class(
        max_nn.Embedding,
        source_module_base=max_module_base,
        class_name=f"{class_prefix}Embedding",
    )
    adapted_module_list = adapt_max_module_class(
        max_nn.ModuleList,
        source_module_base=max_module_base,
        class_name=f"{class_prefix}ModuleList",
    )
    adapted_sequential = adapt_max_module_class(
        max_nn.Sequential,
        base_overrides={max_nn.ModuleList: adapted_module_list},
        class_name=f"{class_prefix}Sequential",
        global_overrides={"ModuleList": adapted_module_list},
    )

    return {
        "Linear": adapted_linear,
        "Embedding": adapted_embedding,
        "ModuleList": adapted_module_list,
        "Sequential": adapted_sequential,
    }
