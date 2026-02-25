# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, ClassVar

from ...core import Tensor, is_tensor, realize_all, register_pytree_node, tree_leaves

_MODULE_CALL_STATE = threading.local()


def _module_tree_flatten(
    module: Module,
) -> tuple[
    list[Any],
    tuple[
        type,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[str, Any], ...],
        bool,
    ],
]:
    param_keys = tuple(sorted(module._parameters.keys()))
    buffer_keys = tuple(sorted(module._buffers.keys()))
    module_keys = tuple(sorted(module._modules.keys()))

    children: list[Any] = []
    children.extend(module._parameters[k] for k in param_keys)
    children.extend(module._buffers[k] for k in buffer_keys)
    children.extend(module._modules[k] for k in module_keys)

    extras: list[tuple[str, Any]] = []
    for key, value in module.__dict__.items():
        if key in {"_parameters", "_buffers", "_modules", "_training"}:
            continue
        if (
            key in module._parameters
            or key in module._buffers
            or key in module._modules
        ):
            continue
        extras.append((key, value))

    return children, (
        type(module),
        param_keys,
        buffer_keys,
        module_keys,
        tuple(sorted(extras, key=lambda item: item[0])),
        bool(module._training),
    )


def _module_tree_unflatten(
    aux_data: tuple[
        type,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[str, Any], ...],
        bool,
    ],
    children: list[Any],
) -> Module:
    cls, param_keys, buffer_keys, module_keys, extras, training = aux_data
    obj = object.__new__(cls)

    object.__setattr__(obj, "_parameters", OrderedDict())
    object.__setattr__(obj, "_buffers", OrderedDict())
    object.__setattr__(obj, "_modules", OrderedDict())
    object.__setattr__(obj, "_training", training)

    cursor = 0
    for key in param_keys:
        value = children[cursor]
        cursor += 1
        obj._parameters[key] = value
        object.__setattr__(obj, key, value)

    for key in buffer_keys:
        value = children[cursor]
        cursor += 1
        obj._buffers[key] = value
        object.__setattr__(obj, key, value)

    for key in module_keys:
        value = children[cursor]
        cursor += 1
        obj._modules[key] = value
        object.__setattr__(obj, key, value)

    for key, value in extras:
        object.__setattr__(obj, key, value)

    return obj


class Module:
    """Base class for all neural-network modules.

    Subclasses must override :meth:`forward`. Parameters (tensors with
    ``requires_grad=True``) assigned to attributes are automatically
    tracked and yielded by :meth:`parameters`. Submodules assigned to
    attributes are recursively tracked by :meth:`modules`.

    Modules are registered as PyTree nodes, so they can be passed
    directly to transforms like :func:`~nabla.vmap`, :func:`~nabla.grad`,
    and :func:`~nabla.compile` without any special wrapping.

    Example::

        class MLP(nabla.nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.fc1 = nabla.nn.Linear(in_dim, 64)
                self.fc2 = nabla.nn.Linear(64, out_dim)

            def forward(self, x):
                return self.fc2(nabla.relu(self.fc1(x)))
    """

    _PYTREE_REGISTERED: ClassVar[bool] = False
    _AUTO_REALIZE_TOPLEVEL_FORWARD: ClassVar[bool] = False
    _AUTO_REALIZE_BACKWARD_GRADS: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, _module_tree_flatten, _module_tree_unflatten)
        cls._PYTREE_REGISTERED = True

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_training", True)

    @classmethod
    def set_execution_policy(
        cls,
        *,
        auto_realize_toplevel_forward: bool | None = None,
        auto_realize_backward_grads: bool | None = None,
    ) -> None:
        """Set global nn.Module execution ergonomics.

        These are nn-level conveniences and do not change raw transform behavior.
        """
        if auto_realize_toplevel_forward is not None:
            cls._AUTO_REALIZE_TOPLEVEL_FORWARD = bool(auto_realize_toplevel_forward)
        if auto_realize_backward_grads is not None:
            cls._AUTO_REALIZE_BACKWARD_GRADS = bool(auto_realize_backward_grads)

    @classmethod
    def get_execution_policy(cls) -> dict[str, bool]:
        return {
            "auto_realize_toplevel_forward": bool(cls._AUTO_REALIZE_TOPLEVEL_FORWARD),
            "auto_realize_backward_grads": bool(cls._AUTO_REALIZE_BACKWARD_GRADS),
        }

    @staticmethod
    def _realize_tensor_tree(tree: Any) -> None:
        tensors = [
            leaf for leaf in tree_leaves(tree) if is_tensor(leaf) and not leaf.real
        ]
        if tensors:
            realize_all(*tensors)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_parameters", "_buffers", "_modules", "_training"}:
            object.__setattr__(self, name, value)
            return

        if hasattr(self, "_parameters") and name in self._parameters:
            del self._parameters[name]
        if hasattr(self, "_buffers") and name in self._buffers:
            del self._buffers[name]
        if hasattr(self, "_modules") and name in self._modules:
            del self._modules[name]

        object.__setattr__(self, name, value)

        if isinstance(value, Module):
            self._modules[name] = value
        elif is_tensor(value):
            if value.requires_grad:
                self._parameters[name] = value
            else:
                self._buffers[name] = value

    def register_buffer(self, name: str, tensor: Tensor | None) -> None:
        setattr(self, name, tensor)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        depth = int(getattr(_MODULE_CALL_STATE, "depth", 0))
        _MODULE_CALL_STATE.depth = depth + 1
        try:
            out = self.forward(*args, **kwargs)
        finally:
            _MODULE_CALL_STATE.depth = depth

        is_toplevel_user_call = depth == 0
        if is_toplevel_user_call and self._AUTO_REALIZE_TOPLEVEL_FORWARD:
            self._realize_tensor_tree(out)
        return out

    def backward(
        self,
        loss: Tensor,
        gradient: Tensor | None = None,
        retain_graph: bool = False,
        create_graph: bool = False,
        *,
        realize_grads: bool | None = None,
    ) -> None:
        """PyTorch-style backward convenience attached to Module.

        Optionally realizes all parameter gradients after backward.
        """
        loss.backward(
            gradient=gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )

        should_realize = (
            self._AUTO_REALIZE_BACKWARD_GRADS
            if realize_grads is None
            else bool(realize_grads)
        )
        if not should_realize:
            return

        grads = [
            grad
            for grad in (p.grad for p in self.parameters())
            if is_tensor(grad) and not grad.real
        ]
        if grads:
            realize_all(*grads)

    def parameters(self) -> Iterator[Tensor]:
        for _, param in self.named_parameters():
            yield param

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(child_prefix)

    def buffers(self) -> Iterator[Tensor]:
        for _, buf in self.named_buffers():
            yield buf

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for name, buf in self._buffers.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, buf

        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_buffers(child_prefix)

    def modules(self) -> Iterator[Module]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def train(self) -> Module:
        for module in self.modules():
            module._training = True
        return self

    def eval(self) -> Module:
        for module in self.modules():
            module._training = False
        return self

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def state_dict(self) -> OrderedDict[str, Tensor]:
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param
        for name, buf in self.named_buffers():
            state[name] = buf
        return state

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor]) -> None:
        for name, tensor in state_dict.items():
            parts = name.split(".")
            module = self
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], tensor)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = "\n".join("  " + line for line in mod_str.split("\n"))
            child_lines.append(f"({name}): {mod_str}")

        lines = extra_lines + child_lines
        main = f"{self.__class__.__name__}("
        if lines:
            main += "\n  " + "\n  ".join(lines) + "\n"
        main += ")"
        return main


register_pytree_node(Module, _module_tree_flatten, _module_tree_unflatten)
Module._PYTREE_REGISTERED = True
