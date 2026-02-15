# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core import Tensor


PathLike = str | Path


def _is_tensor(x: Any) -> bool:
    return isinstance(x, Tensor)


def _collect_tensor_paths(tree: Any, prefix: str, out: list[tuple[str, Tensor]]) -> None:
    if _is_tensor(tree):
        out.append((prefix, tree))
        return

    if isinstance(tree, dict):
        for key in sorted(tree.keys()):
            key_str = str(key)
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            _collect_tensor_paths(tree[key], child_prefix, out)
        return

    if isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            child_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            _collect_tensor_paths(value, child_prefix, out)
        return


def _assign_tensor_path(tree: Any, path: str, value: Tensor) -> Any:
    if path == "":
        return value

    if path.startswith("["):
        close = path.index("]")
        index = int(path[1:close])
        rest = path[close + 1 :]
        if rest.startswith("."):
            rest = rest[1:]

        if isinstance(tree, tuple):
            mutable = list(tree)
            mutable[index] = _assign_tensor_path(mutable[index], rest, value)
            return tuple(mutable)
        if isinstance(tree, list):
            tree[index] = _assign_tensor_path(tree[index], rest, value)
            return tree
        raise TypeError(f"Cannot assign index path {path} into {type(tree)}")

    if "." in path:
        head, rest = path.split(".", 1)
    else:
        head, rest = path, ""

    key: Any = head
    if isinstance(tree, dict) and head not in tree:
        try:
            key = int(head)
        except ValueError:
            key = head

    if isinstance(tree, dict):
        tree[key] = _assign_tensor_path(tree[key], rest, value)
        return tree

    raise TypeError(f"Cannot assign key path {path} into {type(tree)}")


def save_finetune_checkpoint(
    path: PathLike,
    *,
    lora_params: Any,
    optimizer_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save LoRA params and optional optimizer state to Nabla-native checkpoint files.

    Output files in directory `path`:
      - tensors.npz
      - meta.json
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: list[tuple[str, Tensor]] = []
    _collect_tensor_paths(lora_params, "lora", tensors)

    step = None
    if optimizer_state is not None:
        step = int(optimizer_state.get("step", 0))
        if "m" in optimizer_state:
            _collect_tensor_paths(optimizer_state["m"], "opt_m", tensors)
        if "v" in optimizer_state:
            _collect_tensor_paths(optimizer_state["v"], "opt_v", tensors)

    payload = {name: tensor.to_numpy() for name, tensor in tensors}
    np.savez_compressed(out_dir / "tensors.npz", **payload)

    meta = {
        "step": step,
        "has_optimizer": optimizer_state is not None,
        "user_metadata": metadata or {},
        "tensor_keys": sorted(payload.keys()),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_finetune_checkpoint(
    path: PathLike,
    *,
    lora_template: Any,
    optimizer_template: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    """Load LoRA params and optional optimizer state from checkpoint.

    Templates define output pytree structure and tensor dtypes/devices.
    """
    ckpt_dir = Path(path)
    tensors_path = ckpt_dir / "tensors.npz"
    meta_path = ckpt_dir / "meta.json"

    if not tensors_path.exists():
        raise FileNotFoundError(f"Missing checkpoint tensor file: {tensors_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    npz = np.load(tensors_path, allow_pickle=False)

    loaded_lora = lora_template
    lora_tensor_paths: list[tuple[str, Tensor]] = []
    _collect_tensor_paths(lora_template, "lora", lora_tensor_paths)

    for path_name, template_tensor in lora_tensor_paths:
        if path_name not in npz:
            raise KeyError(f"Missing tensor key in checkpoint: {path_name}")
        arr = npz[path_name]
        restored = Tensor.from_dlpack(arr).to(template_tensor.dtype)
        loaded_lora = _assign_tensor_path(loaded_lora, path_name[len("lora") + 1 :], restored)

    loaded_opt = None
    if optimizer_template is not None and meta.get("has_optimizer", False):
        loaded_opt = {
            "step": int(meta.get("step") or 0),
            "m": optimizer_template.get("m"),
            "v": optimizer_template.get("v"),
        }

        for slot in ("m", "v"):
            slot_template = loaded_opt[slot]
            slot_paths: list[tuple[str, Tensor]] = []
            _collect_tensor_paths(slot_template, f"opt_{slot}", slot_paths)
            for path_name, template_tensor in slot_paths:
                if path_name not in npz:
                    raise KeyError(f"Missing tensor key in checkpoint: {path_name}")
                arr = npz[path_name]
                restored = Tensor.from_dlpack(arr).to(template_tensor.dtype)
                stripped = path_name[len(f"opt_{slot}") + 1 :]
                loaded_opt[slot] = _assign_tensor_path(loaded_opt[slot], stripped, restored)

    return loaded_lora, loaded_opt, meta
