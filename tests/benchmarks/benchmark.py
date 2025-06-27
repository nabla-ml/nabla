"""
=======================================================================================
DEFINITIVE BENCHMARKING SUITE FOR NABLA, JAX, AND PYTORCH (CPU) - V16.0 (ANALYSIS & SCALABILITY)
=======================================================================================
This version enhances analysis with visual plots and robust summary scores, and introduces
a new scalability benchmark to test performance across different problem sizes.

Key Features (V16.0):
1.  VISUAL ANALYSIS: Generates and saves bar charts comparing framework performance using
    matplotlib and seaborn, making results easier to interpret.
2.  SCALABILITY BENCHMARK: Adds a new 'scalability' benchmark that runs a core pattern
    (MLP Layer) over a range of increasing input sizes to visualize scaling characteristics.
3.  ROBUST SUMMARY SCORES: Implements a geometric mean of relative performance to calculate
    a single, fair "overall score" for each framework within a benchmark category.
4.  METHODOLOGY DOCUMENTATION: The core timing function is now heavily commented to explain
    the rigorous methodology (warm-up, synchronization, best-of-n) for trustworthiness.
5.  CLEAN INTEGRATION: All new features are cleanly added to the existing script structure.
"""

import argparse
import os
import platform
import sys
import time
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Framework Imports ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.nn as jax_nn
import jax.numpy as jnp
import torch
import torch._dynamo
import torch.func
import torch.nn.functional as F

# Attempt to import Nabla and get its version
try:
    import nabla as nb

    NABLA_VERSION = nb.__version__
except ImportError:
    print(
        "WARNING: Nabla library not found or import failed. Nabla benchmarks will fail."
    )
    nb = None  # Placeholder
    NABLA_VERSION = "Not Found"
except AttributeError:
    NABLA_VERSION = "unknown (nabla imported but __version__ missing)"


# --- Global Configuration ---
SEED = 42
TIMEIT_RUNS_DEFAULT = 10
TIMEIT_REPEATS_DEFAULT = 3


class bcolors:
    HEADER, OKBLUE, OKCYAN, OKGREEN, WARNING, FAIL, ENDC, BOLD = (
        "\033[95m",
        "\033[94m",
        "\033[96m",
        "\033[92m",
        "\033[93m",
        "\033[91m",
        "\033[0m",
        "\033[1m",
    )


# ============================================================================
# 1. HELPER & DATA GENERATION CLASSES (Corrected Version)
# ============================================================================
class FrameworkAPIs(NamedTuple):
    numpy: np
    nabla: nb
    jax: jnp
    torch: torch


class DataManager:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.apis = FrameworkAPIs(np, nb, jnp, torch)

    def get_tensor(self, shape: tuple, framework: str, dtype: str = "float32"):
        if isinstance(shape, list) or (
            isinstance(shape, tuple) and len(shape) > 0 and isinstance(shape[0], list)
        ):
            return [
                self.get_tensor(s, framework, dt)
                for s, dt in zip(shape[0], shape[1], strict=False)
            ]

        numpy_data = None
        if shape == ():
            if "int" in dtype:
                numpy_data = self.rng.integers(0, 100, dtype=dtype)
            else:
                numpy_data = self.rng.random(dtype=np.float32)
        else:
            if dtype == "int64":
                high = shape[0] if len(shape) > 0 and isinstance(shape[0], int) else 100
                numpy_data = self.rng.integers(0, high, shape).astype(np.int64)
            elif dtype == "int32":
                high = shape[0] if len(shape) > 0 and isinstance(shape[0], int) else 100
                numpy_data = self.rng.integers(0, high, shape, dtype=np.int32)
            else:
                numpy_data = self.rng.random(shape, dtype=np.float32)

        if framework == "nabla":
            if nb is None:
                raise ImportError("Nabla not available for tensor creation.")
            return nb.array(numpy_data)
        if framework == "jax":
            return jnp.array(numpy_data)
        if framework == "torch":
            return torch.from_numpy(np.array(numpy_data))
        return numpy_data


# ============================================================================
# 2. BENCHMARK DEFINITIONS
# ============================================================================
@dataclass
class Pattern:
    name: str
    nabla_func: Callable
    jax_func: Callable
    torch_func: Callable
    arg_shapes: list[tuple]
    arg_dtypes: list[str] = None

    def __post_init__(self):
        if self.arg_dtypes is None:
            self.arg_dtypes = ["float32"] * len(self.arg_shapes)


@dataclass
class Transformation:
    name: str
    nabla_transform: Callable
    jax_transform: Callable
    torch_transform: Callable


def get_patterns_and_transforms(dims):
    N, D, B, S, E = dims["N"], dims["D"], dims["B"], dims["S"], dims["E"]
    N_LAYERS_MLP = dims.get("N_LAYERS_MLP", 8)

    basic_patterns = [
        Pattern(
            name="Element-wise",
            nabla_func=lambda x: nb.sum(nb.tanh(x) * nb.sin(x) / (nb.abs(x) + 1e-6))
            if nb
            else None,
            jax_func=lambda x: jnp.sum(jnp.tanh(x) * jnp.sin(x) / (jnp.abs(x) + 1e-6)),
            torch_func=lambda x: torch.sum(
                torch.tanh(x) * torch.sin(x) / (torch.abs(x) + 1e-6)
            ),
            arg_shapes=[(N, D)],
        ),
        Pattern(
            name="Matmul",
            nabla_func=lambda x, w: nb.sum(x @ w) if nb else None,
            jax_func=lambda x, w: jnp.sum(x @ w),
            torch_func=lambda x, w: torch.sum(x @ w),
            arg_shapes=[(N, D), (D, N)],
        ),
        Pattern(
            name="Reduction",
            nabla_func=lambda x: nb.sum(
                nb.sqrt(nb.mean((x - nb.mean(x, axes=1, keep_dims=True)) ** 2, axes=1))
            )
            if nb
            else None,
            jax_func=lambda x: jnp.sum(
                jnp.sqrt(
                    jnp.mean((x - jnp.mean(x, axis=1, keepdims=True)) ** 2, axis=1)
                )
            ),
            torch_func=lambda x: torch.sum(
                torch.sqrt(
                    torch.mean((x - torch.mean(x, dim=1, keepdim=True)) ** 2, dim=1)
                )
            ),
            arg_shapes=[(N, D)],
        ),
        Pattern(
            name="MLP Layer",
            nabla_func=lambda x, w, b: nb.sum(nb.relu(x @ w + b)) if nb else None,
            jax_func=lambda x, w, b: jnp.sum(jax.nn.relu(x @ w + b)),
            torch_func=lambda x, w, b: torch.sum(F.relu(x @ w + b)),
            arg_shapes=[(N, D), (D, D), (D,)],
        ),
    ]

    def _nabla_transformer_fwd(x, y, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        attn_out = nb.softmax(q @ k.transpose((0, 2, 1)) / (E**0.5), axis=-1) @ v
        x = x + attn_out @ w_o
        x = x + nb.relu(x @ w_ff1) @ w_ff2
        stable_x = x - nb.max(x, axes=-1, keep_dims=True)
        return (
            -nb.sum(
                y
                * (stable_x - nb.log(nb.sum(nb.exp(stable_x), axes=-1, keep_dims=True)))
            )
            / B
        )

    def _jax_transformer_fwd(x, y, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        attn_out = jax_nn.softmax((q @ k.transpose(0, 2, 1)) / (E**0.5), axis=-1) @ v
        x = x + attn_out @ w_o
        x = x + jax_nn.relu(x @ w_ff1) @ w_ff2
        return -jnp.sum(y * jax_nn.log_softmax(x, axis=-1)) / B

    def _torch_transformer_loss(transformer_params, x_input, y_target):
        w_q, w_k, w_v, w_o, w_ff1, w_ff2 = transformer_params
        q, k, v = x_input @ w_q, x_input @ w_k, x_input @ w_v
        attn_out = F.softmax((q @ k.transpose(-2, -1)) / (E**0.5), dim=-1) @ v
        x_intermediate = x_input + attn_out @ w_o
        x_final = x_intermediate + F.relu(x_intermediate @ w_ff1) @ w_ff2
        return -torch.sum(y_target * F.log_softmax(x_final, dim=-1)) / B

    def _torch_transformer_grad_wrapper(x, y, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
        params_list = [w_q, w_k, w_v, w_o, w_ff1, w_ff2]
        cloned_params_for_grad = [
            p.clone().detach().requires_grad_(True) if p.is_floating_point() else p
            for p in params_list
        ]
        loss = _torch_transformer_loss(cloned_params_for_grad, x, y)
        grads = torch.autograd.grad(loss, cloned_params_for_grad)
        return grads

    transformer_pattern = Pattern(
        name="Transformer Bwd Pass",
        nabla_func=nb.grad(_nabla_transformer_fwd, argnums=tuple(range(2, 8)))
        if nb
        else None,
        jax_func=jax.grad(_jax_transformer_fwd, argnums=range(2, 8)),
        torch_func=_torch_transformer_grad_wrapper,
        arg_shapes=[
            (B, S, E),
            (B, S, E),
            (E, E),
            (E, E),
            (E, E),
            (E, E),
            (E, 4 * E),
            (4 * E, E),
        ],
    )

    def _define_deep_mlp_funcs(layers_arg):
        def _nabla_deep_mlp(x, weights, biases):
            for i in range(layers_arg):
                x = nb.relu(x @ weights[i] + biases[i])
            return nb.sum(x)

        def _jax_deep_mlp(x, weights, biases):
            for i in range(layers_arg):
                x = jax_nn.relu(x @ weights[i] + biases[i])
            return jnp.sum(x)

        def _torch_deep_mlp(x, weights, biases):
            for i in range(layers_arg):
                x = F.relu(x @ weights[i] + biases[i])
            return torch.sum(x)

        return (_nabla_deep_mlp if nb else None), _jax_deep_mlp, _torch_deep_mlp

    _nabla_mlp, _jax_mlp, _torch_mlp = _define_deep_mlp_funcs(layers_arg=N_LAYERS_MLP)
    deep_mlp_pattern = Pattern(
        name=f"Deep MLP ({N_LAYERS_MLP} Layers)",
        nabla_func=_nabla_mlp,
        jax_func=_jax_mlp,
        torch_func=_torch_mlp,
        arg_shapes=[
            (N, D),
            ([[(D, D)] * N_LAYERS_MLP, ["float32"] * N_LAYERS_MLP]),
            ([[(D,)] * N_LAYERS_MLP, ["float32"] * N_LAYERS_MLP]),
        ],
    )

    nabla_grad_placeholder = (
        lambda f: (lambda *args, **kwargs: "Nabla Grad N/A")
        if nb is None or f is None
        else nb.grad(f)
    )
    nabla_jit_placeholder = (
        lambda f: (lambda *args, **kwargs: "Nabla JIT N/A")
        if nb is None or f is None
        else nb.jit(f)
    )
    standard_transforms = [
        Transformation(
            name="Eager",
            nabla_transform=lambda f: f,
            jax_transform=lambda f: f,
            torch_transform=lambda f: f,
        ),
        Transformation(
            name="Grad",
            nabla_transform=nabla_grad_placeholder,
            jax_transform=jax.grad,
            torch_transform=torch.func.grad,
        ),
        Transformation(
            name="JIT",
            nabla_transform=nabla_jit_placeholder,
            jax_transform=jax.jit,
            torch_transform=lambda f: torch.compile(f, mode="max-autotune"),
        ),
        Transformation(
            name="JIT(Grad)",
            nabla_transform=lambda f: nabla_jit_placeholder(nabla_grad_placeholder(f)),
            jax_transform=lambda f: jax.jit(jax.grad(f)),
            torch_transform=lambda f: torch.compile(
                torch.func.grad(f), mode="max-autotune"
            ),
        ),
    ]
    transformer_transforms = [
        Transformation(
            name="Eager",
            nabla_transform=lambda f: f,
            jax_transform=lambda f: f,
            torch_transform=lambda f: f,
        ),
        Transformation(
            name="JIT",
            nabla_transform=nabla_jit_placeholder,
            jax_transform=jax.jit,
            torch_transform=lambda f: torch.compile(f, mode="max-autotune"),
        ),
    ]

    def _nabla_mlp_simple(x, w, b):
        return nb.sum(nb.relu(x @ w + b))

    def _jax_mlp_simple(x, w, b):
        return jnp.sum(jax.nn.relu(x @ w + b))

    def _torch_mlp_simple(x, w, b):
        return torch.sum(F.relu(x @ w + b))

    vmap_pattern = Pattern(
        name="Vmap (MLP Layer)",
        nabla_func=nb.vmap(_nabla_mlp_simple, in_axes=(0, None, None)) if nb else None,
        jax_func=jax.vmap(_jax_mlp_simple, in_axes=(0, None, None)),
        torch_func=torch.func.vmap(_torch_mlp_simple, in_dims=(0, None, None)),
        arg_shapes=[(N, D), (D, D), (D,)],
    )
    jvp_pattern = Pattern(
        name="JVP (MLP Layer)",
        nabla_func=lambda x, w, b, t_x: nb.jvp(
            _nabla_mlp_simple, (x, w, b), (t_x, nb.ones_like(w), nb.ones_like(b))
        )[1]
        if nb
        else None,
        jax_func=lambda x, w, b, t_x: jax.jvp(
            _jax_mlp_simple, (x, w, b), (t_x, jnp.ones_like(w), jnp.ones_like(b))
        )[1],
        torch_func=lambda x, w, b, t_x: torch.func.jvp(
            _torch_mlp_simple, (x, w, b), (t_x, torch.ones_like(w), torch.ones_like(b))
        )[1],
        arg_shapes=[(N, D), (D, D), (D,), (N, D)],
    )

    def _nabla_vjp_wrapper(x, w, b, ct):
        return nb.vjp(lambda x, w, b: nb.relu(x @ w + b), x, w, b)[1](ct)

    def _jax_vjp_wrapper(x, w, b, ct):
        return jax.vjp(lambda x, w, b: jax.nn.relu(x @ w + b), x, w, b)[1](ct)

    def _torch_vjp_wrapper(x, w, b, ct):
        return torch.func.vjp(lambda x, w, b: F.relu(x @ w + b), x, w, b)[1](ct)

    vjp_pattern = Pattern(
        name="VJP (MLP Layer)",
        nabla_func=_nabla_vjp_wrapper if nb else None,
        jax_func=_jax_vjp_wrapper,
        torch_func=_torch_vjp_wrapper,
        arg_shapes=[(N, D), (D, D), (D,), (N, D)],
    )

    def _nabla_hvp(x, w, b, t_x):
        grad_fn = nb.grad(_nabla_mlp_simple, argnums=0)
        return nb.jvp(lambda x_in: grad_fn(x_in, w, b), (x,), (t_x,))[1]

    def _jax_hvp(x, w, b, t_x):
        grad_fn = jax.grad(_jax_mlp_simple, argnums=0)
        return jax.jvp(lambda x_in: grad_fn(x_in, w, b), (x,), (t_x,))[1]

    def _torch_hvp(x, w, b, t_x):
        grad_fn = torch.func.grad(_torch_mlp_simple, argnums=0)
        return torch.func.jvp(lambda x_in: grad_fn(x_in, w, b), (x,), (t_x,))[1]

    hvp_pattern = Pattern(
        name="Hessian-Vector Product",
        nabla_func=_nabla_hvp if nb else None,
        jax_func=_jax_hvp,
        torch_func=_torch_hvp,
        arg_shapes=[(N, D), (D, D), (D,), (N, D)],
    )

    def _nabla_jvp_vjp(x, w, b, ct, t_x):
        f = _nabla_mlp_simple
        g = lambda xp, wp, bp: nb.vjp(f, xp, wp, bp)[1](ct)[0]
        return nb.jvp(g, (x, w, b), (t_x, nb.ones_like(w), nb.ones_like(b)))[1]

    def _jax_jvp_vjp(x, w, b, ct, t_x):
        f = _jax_mlp_simple
        g = lambda xp, wp, bp: jax.vjp(f, xp, wp, bp)[1](ct)[0]
        return jax.jvp(g, (x, w, b), (t_x, jnp.ones_like(w), jnp.ones_like(b)))[1]

    def _torch_jvp_vjp(x, w, b, ct, t_x):
        f = _torch_mlp_simple
        g = lambda xp, wp, bp: torch.func.vjp(f, xp, wp, bp)[1](ct)[0]
        return torch.func.jvp(
            g, (x, w, b), (t_x, torch.ones_like(w), torch.ones_like(b))
        )[1]

    jvp_vjp_pattern = Pattern(
        name="JVP(VJP(MLP))",
        nabla_func=_nabla_jvp_vjp if nb else None,
        jax_func=_jax_jvp_vjp,
        torch_func=_torch_jvp_vjp,
        arg_shapes=[(N, D), (D, D), (D,), (), (N, D)],
    )

    def _nabla_vjp_jvp(x, w, b, t_x, ct):
        f = _nabla_mlp_simple
        g = lambda xp, wp, bp: nb.jvp(
            f, (xp, wp, bp), (t_x, nb.ones_like(w), nb.ones_like(b))
        )[1]
        return nb.vjp(g, x, w, b)[1](ct)

    def _jax_vjp_jvp(x, w, b, t_x, ct):
        f = _jax_mlp_simple
        g = lambda xp, wp, bp: jax.jvp(
            f, (xp, wp, bp), (t_x, jnp.ones_like(w), jnp.ones_like(b))
        )[1]
        return jax.vjp(g, x, w, b)[1](ct)

    def _torch_vjp_jvp(x, w, b, t_x, ct):
        f = _torch_mlp_simple
        g = lambda xp, wp, bp: torch.func.jvp(
            f, (xp, wp, bp), (t_x, torch.ones_like(w), torch.ones_like(b))
        )[1]
        return torch.func.vjp(g, x, w, b)[1](ct)

    vjp_jvp_pattern = Pattern(
        name="VJP(JVP(MLP))",
        nabla_func=_nabla_vjp_jvp if nb else None,
        jax_func=_jax_vjp_jvp,
        torch_func=_torch_vjp_jvp,
        arg_shapes=[(N, D), (D, D), (D,), (N, D), ()],
    )

    def _nabla_jvp_jvp(x, w, b, t_x1, t_x2):
        f = _nabla_mlp_simple
        g = lambda xp, wp, bp: nb.jvp(
            f, (xp, wp, bp), (t_x1, nb.ones_like(w), nb.ones_like(b))
        )[1]
        return nb.jvp(g, (x, w, b), (t_x2, nb.ones_like(w), nb.ones_like(b)))[1]

    def _jax_jvp_jvp(x, w, b, t_x1, t_x2):
        f = _jax_mlp_simple
        g = lambda xp, wp, bp: jax.jvp(
            f, (xp, wp, bp), (t_x1, jnp.ones_like(w), jnp.ones_like(b))
        )[1]
        return jax.jvp(g, (x, w, b), (t_x2, jnp.ones_like(w), jnp.ones_like(b)))[1]

    def _torch_jvp_jvp(x, w, b, t_x1, t_x2):
        f = _torch_mlp_simple
        g = lambda xp, wp, bp: torch.func.jvp(
            f, (xp, wp, bp), (t_x1, torch.ones_like(w), torch.ones_like(b))
        )[1]
        return torch.func.jvp(
            g, (x, w, b), (t_x2, torch.ones_like(w), torch.ones_like(b))
        )[1]

    jvp_jvp_pattern = Pattern(
        name="JVP(JVP(MLP))",
        nabla_func=_nabla_jvp_jvp if nb else None,
        jax_func=_jax_jvp_jvp,
        torch_func=_torch_jvp_jvp,
        arg_shapes=[(N, D), (D, D), (D,), (N, D), (N, D)],
    )

    def _nabla_vjp_vjp(x, w, b, ct1, ct2):
        f = _nabla_mlp_simple
        g = lambda xp, wp, bp: nb.vjp(f, xp, wp, bp)[1](ct1)[0]
        return nb.vjp(g, x, w, b)[1](ct2)

    def _jax_vjp_vjp(x, w, b, ct1, ct2):
        f = _jax_mlp_simple
        g = lambda xp, wp, bp: jax.vjp(f, xp, wp, bp)[1](ct1)[0]
        return jax.vjp(g, x, w, b)[1](ct2)

    def _torch_vjp_vjp(x, w, b, ct1, ct2):
        f = _torch_mlp_simple
        g = lambda xp, wp, bp: torch.func.vjp(f, xp, wp, bp)[1](ct1)[0]
        return torch.func.vjp(g, x, w, b)[1](ct2)

    vjp_vjp_pattern = Pattern(
        name="VJP(VJP(MLP))",
        nabla_func=_nabla_vjp_vjp if nb else None,
        jax_func=_jax_vjp_vjp,
        torch_func=_torch_vjp_vjp,
        arg_shapes=[(N, D), (D, D), (D,), (), (N, D)],
    )

    advanced_patterns = [
        vmap_pattern,
        jvp_pattern,
        vjp_pattern,
        hvp_pattern,
        jvp_vjp_pattern,
        vjp_jvp_pattern,
        jvp_jvp_pattern,
        vjp_vjp_pattern,
    ]
    advanced_transforms = [
        Transformation(
            name="Eager",
            nabla_transform=lambda f: f,
            jax_transform=lambda f: f,
            torch_transform=lambda f: f,
        ),
        Transformation(
            name="JIT",
            nabla_transform=nabla_jit_placeholder,
            jax_transform=jax.jit,
            torch_transform=lambda f: torch.compile(f, mode="max-autotune"),
        ),
    ]

    return (
        basic_patterns,
        standard_transforms,
        transformer_pattern,
        transformer_transforms,
        deep_mlp_pattern,
        advanced_patterns,
        advanced_transforms,
    )


# ============================================================================
# 3. BENCHMARK RUNNER CLASS
# ============================================================================
class BenchmarkRunner:
    def __init__(
        self,
        config_name: str,
        data_manager: DataManager,
        timeit_runs: int,
        timeit_repeats: int,
        frameworks: list,
        **kwargs,
    ):
        self.config_name = config_name
        self.dm = data_manager
        self.timeit_runs = timeit_runs
        self.timeit_repeats = timeit_repeats
        self.frameworks_to_run = frameworks
        self.kwargs = kwargs
        self.results = []
        print(
            f"\n\n{bcolors.HEADER}{'=' * 80}\n {self.config_name.center(78)} \n {str(self.kwargs).center(78)} \n{'=' * 80}{bcolors.ENDC}"
        )

    def run(self, patterns: list[Pattern], transforms: list[Transformation]):
        torch._dynamo.reset()
        for pattern_obj in patterns:
            print(
                f"\n{bcolors.OKBLUE}{'-' * 20} PATTERN: {pattern_obj.name} {'-' * 20}{bcolors.ENDC}"
            )
            for trans in transforms:
                print(f"  {bcolors.OKCYAN}--> TRANSFORM: {trans.name}{bcolors.ENDC}")
                self._run_single_test(pattern_obj, trans, self.frameworks_to_run)
        return pd.DataFrame(self.results)

    def _run_single_test(
        self, pattern_obj: Pattern, trans: Transformation, frameworks_to_run: list[str]
    ):
        actual_arg_dtypes = (
            pattern_obj.arg_dtypes
            if pattern_obj.arg_dtypes is not None
            else ["float32"] * len(pattern_obj.arg_shapes)
        )
        args_nabla, args_jax, args_torch = None, None, None
        try:
            if "Nabla" in frameworks_to_run and nb and pattern_obj.nabla_func:
                args_nabla = tuple(
                    self.dm.get_tensor(s, "nabla", dt)
                    for s, dt in zip(
                        pattern_obj.arg_shapes, actual_arg_dtypes, strict=True
                    )
                )
            if "JAX" in frameworks_to_run and pattern_obj.jax_func:
                args_jax = tuple(
                    self.dm.get_tensor(s, "jax", dt)
                    for s, dt in zip(
                        pattern_obj.arg_shapes, actual_arg_dtypes, strict=True
                    )
                )
            if "PyTorch" in frameworks_to_run and pattern_obj.torch_func:
                args_torch = tuple(
                    self.dm.get_tensor(s, "torch", dt)
                    for s, dt in zip(
                        pattern_obj.arg_shapes, actual_arg_dtypes, strict=True
                    )
                )
        except Exception as e:
            print(
                f"      {bcolors.FAIL}Argument Preparation FAILED for {pattern_obj.name}: {e}{bcolors.ENDC}"
            )
            self.results.append(
                {
                    "Benchmark": self.config_name,
                    "Pattern": pattern_obj.name,
                    "Transform": trans.name,
                    "Framework": "All",
                    "Error": "Argument Preparation Failed",
                }
            )
            return

        framework_funcs = {
            "Nabla": (trans.nabla_transform(pattern_obj.nabla_func), args_nabla),
            "JAX": (trans.jax_transform(pattern_obj.jax_func), args_jax),
            "PyTorch": (trans.torch_transform(pattern_obj.torch_func), args_torch),
        }
        for fw_name in frameworks_to_run:
            func_tuple = framework_funcs.get(fw_name)
            if func_tuple and func_tuple[0] and func_tuple[1]:
                self._measure_and_store(
                    fw_name, *func_tuple, pattern_obj.name, trans.name
                )
            else:
                if fw_name == "Nabla" and nb is None:
                    print(
                        f"      {bcolors.WARNING}Nabla not available, skipping {pattern_obj.name} - {trans.name}{bcolors.ENDC}"
                    )
                self.results.append(
                    {
                        "Benchmark": self.config_name,
                        "Pattern": pattern_obj.name,
                        "Transform": trans.name,
                        "Framework": fw_name,
                        "Time (ms)": np.nan,
                        "First Run Time (ms)": np.nan,
                        "Error": "Skipped (FW/Func N/A)",
                    }
                )

    def _measure_and_store(
        self, fw_name, func_to_benchmark, args_tuple, pattern_name_str, trans_name_str
    ):
        """
        Measures and records the performance of a given function. This method follows a rigorous
        procedure to ensure fair and accurate timing:
        1.  WARM-UP RUN: A single, untimed call is made first. For JIT-compiled functions,
            this triggers the compilation and caches the result. This ensures we are not
            measuring the one-time compilation cost in our steady-state analysis. This run
            is timed separately to report the "First Run Time (ms)" which includes overhead.
        2.  SYNCHRONIZATION: For asynchronous frameworks (JAX, PyTorch), a synchronization
            call (`jax.block_until_ready`, `torch.cpu.synchronize`) is placed *inside* the
            function being timed. This forces the timer to wait until the computation is
            actually finished, not just when the kernel was launched. This is critical for accuracy.
        3.  STEADY-STATE TIMING: `timeit.Timer.repeat` is used to run the benchmark multiple
            times. It performs `timeit_repeats` loops, and each loop executes the function
            `timeit_runs` times. This helps get a stable measurement.
        4.  BEST-OF-N: We take the `min()` of the results from the repeats. This is a standard
            technique to reduce the impact of system noise and report the performance under
            optimal conditions. The result is the average time per run from the fastest loop.
        """
        try:

            def func_base_call():
                return func_to_benchmark(*args_tuple)

            if fw_name == "JAX":

                def func_call_wrapper_for_timing():
                    res = func_base_call()
                    jax.block_until_ready(res)
            elif fw_name == "PyTorch":

                def func_call_wrapper_for_timing():
                    res = func_base_call()
                    torch.cpu.synchronize()
            else:
                func_call_wrapper_for_timing = func_base_call

            start_time_first_run = time.perf_counter()
            func_call_wrapper_for_timing()
            first_run_ms = (time.perf_counter() - start_time_first_run) * 1000

            timer = timeit.Timer(stmt=func_call_wrapper_for_timing, globals=globals())
            times = timer.repeat(repeat=self.timeit_repeats, number=self.timeit_runs)
            steady_state_ms = (min(times) / self.timeit_runs) * 1000

            print(
                f"      {fw_name:<8} First Run: {first_run_ms:8.3f} ms | Steady State: {steady_state_ms:8.3f} ms"
            )
            self.results.append(
                {
                    "Benchmark": self.config_name,
                    "Pattern": pattern_name_str,
                    "Transform": trans_name_str,
                    "Framework": fw_name,
                    "Time (ms)": steady_state_ms,
                    "First Run Time (ms)": first_run_ms,
                }
            )
        except Exception as e:
            print(
                f"      {bcolors.FAIL}{fw_name:<8} FAILED ({type(e).__name__}): {str(e)[:150]}{bcolors.ENDC}"
            )
            self.results.append(
                {
                    "Benchmark": self.config_name,
                    "Pattern": pattern_name_str,
                    "Transform": trans_name_str,
                    "Framework": fw_name,
                    "Time (ms)": np.nan,
                    "First Run Time (ms)": np.nan,
                    "Error": str(e)[:100],
                }
            )


# ============================================================================
# 4. RESULTS ANALYSIS CLASS
# ============================================================================
class ResultAnalyzer:
    def __init__(self, df: pd.DataFrame, frameworks: list):
        self.df = df.copy()
        self.frameworks = frameworks
        if "First Run Time (ms)" not in self.df.columns and not self.df.empty:
            self.df["First Run Time (ms)"] = np.nan

    def _create_summary_df(self, time_column_name):
        df_subset = self.df.dropna(subset=[time_column_name])
        if df_subset.empty:
            return pd.DataFrame()
        pivot_df = df_subset.pivot_table(
            index=["Benchmark", "Pattern"],
            columns=["Framework", "Transform"],
            values=time_column_name,
        )
        if not pivot_df.empty:
            current_frameworks_in_pivot = pivot_df.columns.get_level_values(
                "Framework"
            ).unique()
            frameworks_to_reindex = [
                fw for fw in self.frameworks if fw in current_frameworks_in_pivot
            ]
            if frameworks_to_reindex:
                pivot_df = pivot_df.reindex(
                    frameworks_to_reindex, axis=1, level="Framework"
                )
        return pivot_df

    def _build_speedup_table(self, pivot_df, base_transform, jit_transform):
        speedup_data = []
        required_cols = []
        for fw in self.frameworks:
            if (fw, base_transform) in pivot_df.columns:
                required_cols.append((fw, base_transform))
            if (fw, jit_transform) in pivot_df.columns:
                required_cols.append((fw, jit_transform))
        if not required_cols:
            return pd.DataFrame()
        filtered_pivot = pivot_df.dropna(how="all", subset=required_cols)
        if filtered_pivot.empty:
            return pd.DataFrame()
        for idx in filtered_pivot.index:
            row_dict = {"Benchmark": idx[0], "Pattern": idx[1]}
            has_data = False
            for fw in self.frameworks:
                try:
                    eager_time = filtered_pivot.loc[idx, (fw, base_transform)]
                    jit_time = filtered_pivot.loc[idx, (fw, jit_transform)]
                    if pd.notna(eager_time) and pd.notna(jit_time) and jit_time > 1e-9:
                        row_dict[f"{fw}"] = eager_time / jit_time
                        has_data = True
                    else:
                        row_dict[f"{fw}"] = np.nan
                except KeyError:
                    row_dict[f"{fw}"] = np.nan
            if has_data:
                speedup_data.append(row_dict)
        if not speedup_data:
            return pd.DataFrame()
        df = pd.DataFrame(speedup_data).set_index(["Benchmark", "Pattern"])
        df.columns = [f"{c} Speedup (x)" for c in df.columns]
        return df

    def _build_comparison_table(self, pivot_df, transform_name):
        comparison_data = []
        try:
            if not any(
                transform_name in col
                for col in pivot_df.columns.get_level_values("Transform")
            ):
                return pd.DataFrame()
            jit_df = pivot_df.xs(transform_name, level="Transform", axis=1).dropna(
                how="all"
            )
        except KeyError:
            return pd.DataFrame()
        for idx in jit_df.index:
            row_dict = {"Benchmark": idx[0], "Pattern": idx[1]}
            times = jit_df.loc[idx].dropna()
            if times.empty:
                continue
            best_fw, best_time = times.idxmin(), times.min()
            row_dict["Best Framework"] = best_fw
            for fw in self.frameworks:
                time = times.get(fw, np.nan)
                row_dict[f"{fw} Time (ms)"] = time
                if pd.notna(time) and best_time > 1e-9:
                    row_dict[f"{fw} Rel. to Best"] = time / best_time
                else:
                    row_dict[f"{fw} Rel. to Best"] = np.nan
            comparison_data.append(row_dict)
        if not comparison_data:
            return pd.DataFrame()
        return pd.DataFrame(comparison_data).set_index(["Benchmark", "Pattern"])

    def display_table(self, df, title):
        if df.empty:
            return
        print(
            f"\n\n{bcolors.HEADER}{'=' * 110}\n{title.center(110)}\n{'=' * 110}{bcolors.ENDC}"
        )
        key_benchmarks = [
            "TRANSFORMER",
            "ADVANCED-BENCH",
            "MACRO-BENCH",
            "MICRO-BENCH",
            "SCALABILITY-BENCH",
        ]
        benchmark_index = df.index.get_level_values("Benchmark").unique()
        sorted_benchmarks = [b for b in key_benchmarks if b in benchmark_index]
        other_benchmarks = [
            b
            for b in benchmark_index
            if b not in sorted_benchmarks and not b.startswith("SCALABILITY")
        ]
        scalability_benchmarks = sorted(
            [b for b in benchmark_index if b.startswith("SCALABILITY")]
        )
        df = df.reindex(
            sorted_benchmarks + other_benchmarks + scalability_benchmarks,
            level="Benchmark",
        )
        display_df = df.copy().astype(object)
        if "Best Framework" in display_df.columns:
            for idx in display_df.index:
                best_fw = display_df.loc[idx, "Best Framework"]
                if pd.notna(best_fw):
                    time_col = f"{best_fw} Time (ms)"
                    if time_col in display_df.columns and pd.notna(
                        display_df.loc[idx, time_col]
                    ):
                        display_df.loc[idx, time_col] = (
                            f"{display_df.loc[idx, time_col]:8.3f}*"
                        )
        formatters = {
            c: (lambda x: f"{x:8.3f}" if isinstance(x, (int, float)) else x)
            if "Time (ms)" in c
            else (lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
            if "Speedup" in c or "Rel. to Best" in c
            else (lambda x: f"{x}" if pd.notna(x) else "-")
            for c in df.columns
        }
        with pd.option_context(
            "display.max_rows",
            None,
            "display.width",
            200,
            "display.colheader_justify",
            "right",
        ):
            print(display_df.to_string(formatters=formatters, na_rep="-"))

    def display_summary_score(self, pivot_df, transform_name, title):
        try:
            jit_df = pivot_df.xs(transform_name, level="Transform", axis=1).dropna(
                how="all"
            )
        except KeyError:
            return
        if jit_df.empty:
            return
        best_times = jit_df.min(axis=1)
        relative_perf = jit_df.div(best_times, axis=0)
        geo_mean = relative_perf.apply(
            lambda x: np.exp(np.log(x.dropna()).mean())
        ).sort_values()
        print(
            f"\n\n{bcolors.HEADER}{'=' * 60}\n{title.center(60)}\n{'(Geometric Mean of Performance Relative to Best - Lower is Better)'.center(60)}\n{'=' * 60}{bcolors.ENDC}"
        )
        print(geo_mean.to_string(float_format="%.2fx"))

    def plot_comparison(
        self, time_column_name, transform_name, benchmark_name, title_suffix
    ):
        pivot_df = self._create_summary_df(time_column_name)
        if pivot_df.empty:
            return
        try:
            df_to_plot = pivot_df.xs(transform_name, level="Transform", axis=1).dropna(
                how="all"
            )
            if benchmark_name == "SCALABILITY-BENCH":
                patterns_to_plot = [
                    p
                    for p in df_to_plot.index.get_level_values("Pattern")
                    if "MLP Layer" in p
                ]
                df_to_plot = df_to_plot[
                    df_to_plot.index.get_level_values("Pattern").isin(patterns_to_plot)
                ]
                df_to_plot["Size"] = (
                    df_to_plot.index.get_level_values("Pattern")
                    .str.extract(r"(\d+)x")
                    .astype(int)
                )
                df_to_plot = df_to_plot.reset_index().melt(
                    id_vars=["Pattern", "Size"],
                    var_name="Framework",
                    value_name="Time (ms)",
                )
            else:
                df_to_plot = df_to_plot.loc[benchmark_name]
                df_to_plot = df_to_plot.stack().reset_index()
                df_to_plot.columns = ["Pattern", "Framework", "Time (ms)"]
        except KeyError:
            print(
                f"{bcolors.WARNING}\nCould not generate plot: Data for Transform='{transform_name}' and Benchmark='{benchmark_name}' not found.{bcolors.ENDC}"
            )
            return
        if df_to_plot.empty:
            print(
                f"{bcolors.WARNING}\nCould not generate plot: No data points for '{benchmark_name}' after filtering.{bcolors.ENDC}"
            )
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        if benchmark_name == "SCALABILITY-BENCH":
            sns.lineplot(
                data=df_to_plot,
                x="Size",
                y="Time (ms)",
                hue="Framework",
                marker="o",
                ax=ax,
                palette="viridis",
            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Matrix Size (N in N,N)", fontsize=12)
            ax.set_title(
                f"Framework Scaling: {title_suffix}", fontsize=16, weight="bold"
            )
        else:
            sns.barplot(
                data=df_to_plot,
                x="Pattern",
                y="Time (ms)",
                hue="Framework",
                ax=ax,
                palette="viridis",
            )
            ax.set_xlabel("Benchmark Pattern", fontsize=12)
            ax.tick_params(axis="x", rotation=15)
            if df_to_plot["Time (ms)"].max() / df_to_plot["Time (ms)"].min() > 10:
                ax.set_yscale("log")
                ax.set_ylabel("Time (ms) - Lower is Better (Log Scale)", fontsize=12)
            ax.set_title(
                f"Framework Comparison: {benchmark_name} ({title_suffix})",
                fontsize=16,
                weight="bold",
            )

        ax.set_ylabel("Time (ms) - Lower is Better", fontsize=12)
        ax.legend(title="Framework", fontsize=10)
        fig.tight_layout()
        filename = f"benchmark_{benchmark_name.lower().replace(' ', '_')}_{transform_name.lower().replace('(', '_').replace(')', '')}.png"
        plt.savefig(filename)
        print(f"\n{bcolors.OKGREEN}Generated plot: {filename}{bcolors.ENDC}")

    def process_and_display(self):
        if self.df.empty:
            print(f"{bcolors.WARNING}No results to analyze.{bcolors.ENDC}")
            return

        print(
            f"\n{bcolors.OKBLUE}{bcolors.BOLD}--- STEADY STATE PERFORMANCE ---{bcolors.ENDC}"
        )
        pivot_steady = self._create_summary_df("Time (ms)")
        if not pivot_steady.empty:
            self.display_table(
                self._build_speedup_table(pivot_steady, "Eager", "JIT"),
                "INTRA-FRAMEWORK JIT SPEEDUP (Eager vs JIT)",
            )
            self.display_table(
                self._build_comparison_table(pivot_steady, "JIT"),
                "CROSS-FRAMEWORK JIT COMPARISON (LOWER IS BETTER)",
            )
            self.display_table(
                self._build_speedup_table(pivot_steady, "Grad", "JIT(Grad)"),
                "INTRA-FRAMEWORK JIT SPEEDUP (Grad vs JIT(Grad))",
            )
            self.display_table(
                self._build_comparison_table(pivot_steady, "JIT(Grad)"),
                "CROSS-FRAMEWORK JIT COMPARISON (GRADIENT - LOWER IS BETTER)",
            )
            for bench_name in pivot_steady.index.get_level_values("Benchmark").unique():
                if "SCALABILITY" in bench_name:
                    continue
                bench_pivot = pivot_steady.loc[bench_name]
                self.display_summary_score(
                    bench_pivot, "JIT", f"OVERALL JIT SCORE: {bench_name}"
                )
                self.display_summary_score(
                    bench_pivot, "JIT(Grad)", f"OVERALL JIT(GRAD) SCORE: {bench_name}"
                )

        print(
            f"\n{bcolors.OKBLUE}{bcolors.BOLD}--- FIRST RUN (COMPILATION OVERHEAD) PERFORMANCE ---{bcolors.ENDC}"
        )
        pivot_first_run = self._create_summary_df("First Run Time (ms)")
        if not pivot_first_run.empty:
            self.display_table(
                self._build_comparison_table(pivot_first_run, "JIT"),
                "CROSS-FRAMEWORK JIT COMPILATION TIME (FORWARD - LOWER IS BETTER)",
            )
            self.display_table(
                self._build_comparison_table(pivot_first_run, "JIT(Grad)"),
                "CROSS-FRAMEWORK JIT COMPILATION TIME (GRADIENT - LOWER IS BETTER)",
            )

        print(f"\n{bcolors.OKBLUE}{bcolors.BOLD}--- GENERATING PLOTS ---{bcolors.ENDC}")
        benchmarks_to_plot = self.df["Benchmark"].unique()
        for bench in benchmarks_to_plot:
            if "SCALABILITY" in bench:
                continue  # Skip individual size plots
            self.plot_comparison("Time (ms)", "JIT", bench, "JIT Steady State")
            if any(x in bench for x in ["MICRO", "MACRO", "ADVANCED"]):
                self.plot_comparison(
                    "Time (ms)", "JIT(Grad)", bench, "JIT(Grad) Steady State"
                )
        if any("SCALABILITY" in b for b in benchmarks_to_plot):
            self.plot_comparison(
                "Time (ms)",
                "JIT(Grad)",
                "SCALABILITY-BENCH",
                "MLP Layer Gradient Scaling",
            )


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def print_environment_info():
    print(f"{bcolors.BOLD}Environment Information:{bcolors.ENDC}")
    print(f"  - Python Version: {sys.version.split(' ')[0]}")
    print(
        f"  - Platform: {platform.system()} {platform.release()} ({platform.machine()})"
    )
    try:
        import cpuinfo

        print(f"  - CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    except:
        print("  - CPU: (Install 'py-cpuinfo' for details)")
    print(
        f"  - Library Versions:\n    - Nabla:   {NABLA_VERSION}\n    - JAX:     {jax.__version__}\n    - PyTorch: {torch.__version__}\n    - NumPy:   {np.__version__}\n    - Pandas:  {pd.__version__}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Nabla/JAX/PyTorch Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["micro", "macro", "transformer", "advanced"],
        choices=["micro", "macro", "transformer", "advanced", "scalability", "all"],
        help="Which benchmarks to run.",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["Nabla", "JAX", "PyTorch"],
        choices=["Nabla", "JAX", "PyTorch"],
        help="Which frameworks to test.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=TIMEIT_RUNS_DEFAULT,
        help="Number of runs per timing loop.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=TIMEIT_REPEATS_DEFAULT,
        help="Number of repeat timing loops.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_results.csv",
        help="Path to save the raw timing results CSV file.",
    )
    args = parser.parse_args()

    if "all" in args.benchmarks:
        args.benchmarks = ["micro", "macro", "transformer", "advanced", "scalability"]
    if nb is None and "Nabla" in args.frameworks:
        print(
            f"{bcolors.WARNING}Nabla framework selected but not found. Removing Nabla from tests.{bcolors.ENDC}"
        )
        args.frameworks = [fw for fw in args.frameworks if fw != "Nabla"]
        if not args.frameworks:
            print(f"{bcolors.FAIL}No frameworks left to test. Exiting.{bcolors.ENDC}")
            return

    print_environment_info()
    dm = DataManager(SEED)
    all_results_df = pd.DataFrame()
    print(
        f"\n{bcolors.BOLD}Running benchmarks: {', '.join(args.benchmarks)}{bcolors.ENDC}"
    )
    print(
        f"{bcolors.BOLD}Testing frameworks: {', '.join(args.frameworks)}{bcolors.ENDC}"
    )
    print(
        f"{bcolors.BOLD}Timing config: {args.repeats} repeats, {args.runs} runs each.{bcolors.ENDC}"
    )

    common_kwargs = {
        "data_manager": dm,
        "timeit_runs": args.runs,
        "timeit_repeats": args.repeats,
        "frameworks": args.frameworks,
    }

    if "micro" in args.benchmarks:
        dims = {"N": 4, "D": 8, "B": 0, "S": 0, "E": 0}
        patterns, transforms, _, _, _, _, _ = get_patterns_and_transforms(dims)
        all_results_df = pd.concat(
            [
                all_results_df,
                BenchmarkRunner("MICRO-BENCH", **common_kwargs, **dims).run(
                    patterns, transforms
                ),
            ]
        )
    if "macro" in args.benchmarks:
        dims = {"N": 512, "D": 1024, "B": 0, "S": 0, "E": 0, "N_LAYERS_MLP": 16}
        b_p, s_t, _, _, mlp_p, _, _ = get_patterns_and_transforms(dims)
        all_results_df = pd.concat(
            [
                all_results_df,
                BenchmarkRunner("MACRO-BENCH", **common_kwargs, **dims).run(
                    b_p + [mlp_p], s_t
                ),
            ]
        )
    if "transformer" in args.benchmarks:
        dims = {"N": 0, "D": 0, "B": 16, "S": 256, "E": 512}
        _, _, trans_p, trans_ts, _, _, _ = get_patterns_and_transforms(dims)
        all_results_df = pd.concat(
            [
                all_results_df,
                BenchmarkRunner("TRANSFORMER", **common_kwargs, **dims).run(
                    [trans_p], trans_ts
                ),
            ]
        )
    if "advanced" in args.benchmarks:
        dims = {"N": 256, "D": 512, "B": 0, "S": 0, "E": 0}
        _, _, _, _, _, adv_p, adv_t = get_patterns_and_transforms(dims)
        all_results_df = pd.concat(
            [
                all_results_df,
                BenchmarkRunner("ADVANCED-BENCH", **common_kwargs, **dims).run(
                    adv_p, adv_t
                ),
            ]
        )
    if "scalability" in args.benchmarks:
        scalability_results = []
        for size in [64, 128, 256, 512, 1024, 2048]:
            dims = {"N": size, "D": size, "B": 0, "S": 0, "E": 0}

            def get_mlp_pattern(dims_dict):
                N, D = dims_dict["N"], dims_dict["D"]
                return Pattern(
                    name=f"MLP Layer {N}x{D}",
                    nabla_func=lambda x, w, b: nb.sum(nb.relu(x @ w + b))
                    if nb
                    else None,
                    jax_func=lambda x, w, b: jnp.sum(jax.nn.relu(x @ w + b)),
                    torch_func=lambda x, w, b: torch.sum(F.relu(x @ w + b)),
                    arg_shapes=[(N, D), (D, D), (D,)],
                )

            nabla_grad_placeholder = (
                lambda f: (lambda *args, **kwargs: "Nabla Grad N/A")
                if nb is None or f is None
                else nb.grad(f)
            )
            nabla_jit_placeholder = (
                lambda f: (lambda *args, **kwargs: "Nabla JIT N/A")
                if nb is None or f is None
                else nb.jit(f)
            )
            jit_grad_transform = Transformation(
                name="JIT(Grad)",
                nabla_transform=lambda f: nabla_jit_placeholder(
                    nabla_grad_placeholder(f)
                ),
                jax_transform=lambda f: jax.jit(jax.grad(f)),
                torch_transform=lambda f: torch.compile(
                    torch.func.grad(f), mode="max-autotune"
                ),
            )
            runner = BenchmarkRunner(
                f"SCALABILITY-BENCH (Size={size})", **common_kwargs, **dims
            )
            scalability_results.append(
                runner.run([get_mlp_pattern(dims)], [jit_grad_transform])
            )
        if scalability_results:
            all_results_df = pd.concat([all_results_df] + scalability_results)

    if args.output_csv and not all_results_df.empty:
        all_results_df.to_csv(args.output_csv, index=False)
        print(
            f"\n{bcolors.OKGREEN}Raw results saved to {args.output_csv}{bcolors.ENDC}"
        )

    if not all_results_df.empty:
        analyzer = ResultAnalyzer(all_results_df, args.frameworks)
        analyzer.process_and_display()
    else:
        print(
            f"\n{bcolors.WARNING}No results were generated. Skipping analysis.{bcolors.ENDC}"
        )


if __name__ == "__main__":
    main()


# Output on my machine:
# ================================================================================
# Environment Information:
#   - Python Version: 3.12.4
#   - Platform: Darwin 23.6.0 (arm64)
#   - CPU: Apple M3
#   - Library Versions:
#     - Nabla:   unknown (nabla imported but __version__ missing)
#     - JAX:     0.5.1
#     - PyTorch: 2.6.0
#     - NumPy:   2.3.0
#     - Pandas:  2.2.3

# Running benchmarks: micro, macro, transformer, advanced
# Testing frameworks: Nabla, JAX, PyTorch
# Timing config: 3 repeats, 10 runs each.


# ================================================================================
#                                   MICRO-BENCH
#                     {'N': 4, 'D': 8, 'B': 0, 'S': 0, 'E': 0}
# ================================================================================

# -------------------- PATTERN: Element-wise --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.316 ms | Steady State:    0.047 ms
#       JAX      First Run:  103.006 ms | Steady State:    0.025 ms
#       PyTorch  First Run:    0.147 ms | Steady State:    0.006 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.445 ms | Steady State:    0.231 ms
#       JAX      First Run:  229.189 ms | Steady State:    0.935 ms
#       PyTorch  First Run:    9.682 ms | Steady State:    0.089 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  203.292 ms | Steady State:    0.100 ms
#       JAX      First Run:   35.239 ms | Steady State:    0.004 ms
#       PyTorch  First Run: 1975.524 ms | Steady State:    0.005 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  203.329 ms | Steady State:    0.056 ms
#       JAX      First Run:   46.474 ms | Steady State:    0.005 ms
#       PyTorch  First Run:   96.668 ms | Steady State:    0.008 ms

# -------------------- PATTERN: Matmul --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.136 ms | Steady State:    0.017 ms
#       JAX      First Run:   26.167 ms | Steady State:    0.009 ms
#       PyTorch  First Run:    0.067 ms | Steady State:    0.002 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.213 ms | Steady State:    0.076 ms
#       JAX      First Run:   51.149 ms | Steady State:    0.337 ms
#       PyTorch  First Run:    0.243 ms | Steady State:    0.040 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  200.542 ms | Steady State:    0.075 ms
#       JAX      First Run:   19.026 ms | Steady State:    0.005 ms
#       PyTorch  First Run:   21.525 ms | Steady State:    0.006 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  198.872 ms | Steady State:    0.080 ms
#       JAX      First Run:   25.963 ms | Steady State:    0.007 ms
#       PyTorch  First Run:   73.945 ms | Steady State:    0.010 ms

# -------------------- PATTERN: Reduction --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.288 ms | Steady State:    0.087 ms
#       JAX      First Run:   86.985 ms | Steady State:    0.035 ms
#       PyTorch  First Run:    0.131 ms | Steady State:    0.008 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.550 ms | Steady State:    0.311 ms
#       JAX      First Run:  187.114 ms | Steady State:    0.840 ms
#       PyTorch  First Run:    0.366 ms | Steady State:    0.092 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  264.055 ms | Steady State:    0.157 ms
#       JAX      First Run:   34.440 ms | Steady State:    0.007 ms
#       PyTorch  First Run:   26.298 ms | Steady State:    0.007 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  205.870 ms | Steady State:    0.140 ms
#       JAX      First Run:   36.828 ms | Steady State:    0.007 ms
#       PyTorch  First Run:   79.173 ms | Steady State:    0.010 ms

# -------------------- PATTERN: MLP Layer --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.151 ms | Steady State:    0.035 ms
#       JAX      First Run:   39.946 ms | Steady State:    0.053 ms
#       PyTorch  First Run:    0.086 ms | Steady State:    0.003 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.292 ms | Steady State:    0.151 ms
#       JAX      First Run:  111.068 ms | Steady State:    0.784 ms
#       PyTorch  First Run:    0.262 ms | Steady State:    0.048 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  265.125 ms | Steady State:    0.195 ms
#       JAX      First Run:   33.601 ms | Steady State:    0.007 ms
#       PyTorch  First Run:   28.641 ms | Steady State:    0.008 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  193.743 ms | Steady State:    0.123 ms
#       JAX      First Run:   33.925 ms | Steady State:    0.007 ms
#       PyTorch  First Run:  208.531 ms | Steady State:    0.012 ms


# ================================================================================
#                                   MACRO-BENCH
#        {'N': 512, 'D': 1024, 'B': 0, 'S': 0, 'E': 0, 'N_LAYERS_MLP': 16}
# ================================================================================

# -------------------- PATTERN: Element-wise --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    2.278 ms | Steady State:    1.584 ms
#       JAX      First Run:  114.202 ms | Steady State:    1.035 ms
#       PyTorch  First Run:    0.985 ms | Steady State:    0.776 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    6.744 ms | Steady State:    5.797 ms
#       JAX      First Run:  269.338 ms | Steady State:    3.743 ms
#       PyTorch  First Run:    4.645 ms | Steady State:    2.992 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  219.130 ms | Steady State:    1.847 ms
#       JAX      First Run:   27.866 ms | Steady State:    0.727 ms
#       PyTorch  First Run:   24.793 ms | Steady State:    0.349 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  216.963 ms | Steady State:    0.659 ms
#       JAX      First Run:   24.151 ms | Steady State:    0.527 ms
#       PyTorch  First Run:   77.286 ms | Steady State:    0.664 ms

# -------------------- PATTERN: Matmul --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.623 ms | Steady State:    0.383 ms
#       JAX      First Run:   35.628 ms | Steady State:    1.367 ms
#       PyTorch  First Run:    0.513 ms | Steady State:    0.338 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    2.289 ms | Steady State:    1.671 ms
#       JAX      First Run:   75.882 ms | Steady State:    3.261 ms
#       PyTorch  First Run:    1.223 ms | Steady State:    0.934 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  212.481 ms | Steady State:    0.531 ms
#       JAX      First Run:   25.078 ms | Steady State:    1.309 ms
#       PyTorch  First Run:   22.273 ms | Steady State:    0.350 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  220.731 ms | Steady State:    0.427 ms
#       JAX      First Run:   28.643 ms | Steady State:    1.508 ms
#       PyTorch  First Run:   71.214 ms | Steady State:    0.610 ms

# -------------------- PATTERN: Reduction --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    1.330 ms | Steady State:    1.002 ms
#       JAX      First Run:  120.042 ms | Steady State:    0.360 ms
#       PyTorch  First Run:    1.274 ms | Steady State:    0.101 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    3.942 ms | Steady State:    3.372 ms
#       JAX      First Run:  250.948 ms | Steady State:    1.529 ms
#       PyTorch  First Run:    2.227 ms | Steady State:    0.792 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  194.185 ms | Steady State:    0.088 ms
#       JAX      First Run:   31.329 ms | Steady State:    0.141 ms
#       PyTorch  First Run:   23.263 ms | Steady State:    0.057 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  309.737 ms | Steady State:    0.198 ms
#       JAX      First Run:   41.120 ms | Steady State:    0.205 ms
#       PyTorch  First Run:   77.253 ms | Steady State:    0.189 ms

# -------------------- PATTERN: MLP Layer --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    1.614 ms | Steady State:    1.216 ms
#       JAX      First Run:   49.577 ms | Steady State:    2.783 ms
#       PyTorch  First Run:    1.400 ms | Steady State:    0.898 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    4.634 ms | Steady State:    3.946 ms
#       JAX      First Run:  122.821 ms | Steady State:    6.165 ms
#       PyTorch  First Run:    2.525 ms | Steady State:    1.918 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  262.869 ms | Steady State:    1.006 ms
#       JAX      First Run:   30.240 ms | Steady State:    2.619 ms
#       PyTorch  First Run:   26.755 ms | Steady State:    0.896 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  199.414 ms | Steady State:    1.545 ms
#       JAX      First Run:   49.118 ms | Steady State:    5.092 ms
#       PyTorch  First Run:   82.378 ms | Steady State:    1.927 ms

# -------------------- PATTERN: Deep MLP (16 Layers) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:   18.463 ms | Steady State:   18.825 ms
#       JAX      First Run:   39.354 ms | Steady State:   38.672 ms
#       PyTorch  First Run:   17.004 ms | Steady State:   17.026 ms
#   --> TRANSFORM: Grad
# /Users/tillife/Documents/CodingProjects/nabla/nabla/ops/binary.py:148: RuntimeWarning: invalid value encountered in divide
#   np_result = np.divide(args[0].to_numpy(), args[1].to_numpy())
#       Nabla    First Run:   67.803 ms | Steady State:   68.498 ms
#       JAX      First Run:   93.187 ms | Steady State:   87.355 ms
#       PyTorch  First Run:   42.476 ms | Steady State:   32.076 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  272.538 ms | Steady State:   13.602 ms
#       JAX      First Run:   85.285 ms | Steady State:   36.125 ms
#       PyTorch  First Run:  118.211 ms | Steady State:   14.683 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  293.552 ms | Steady State:   29.946 ms
#       JAX      First Run:  154.842 ms | Steady State:   75.337 ms
#       PyTorch  First Run:  306.885 ms | Steady State:   33.279 ms


# ================================================================================
#                                   TRANSFORMER
#                  {'N': 0, 'D': 0, 'B': 16, 'S': 256, 'E': 512}
# ================================================================================

# -------------------- PATTERN: Transformer Bwd Pass --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:  141.688 ms | Steady State:  168.187 ms
#       JAX      First Run:  899.896 ms | Steady State:  190.118 ms
#       PyTorch  First Run:   97.043 ms | Steady State:   81.431 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  372.617 ms | Steady State:   90.673 ms
#       JAX      First Run:  256.625 ms | Steady State:  173.074 ms
#       PyTorch  First Run:  415.183 ms | Steady State:   89.730 ms


# ================================================================================
#                                  ADVANCED-BENCH
#                   {'N': 256, 'D': 512, 'B': 0, 'S': 0, 'E': 0}
# ================================================================================

# -------------------- PATTERN: Vmap (MLP Layer) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    1.097 ms | Steady State:    0.789 ms
#       JAX      First Run:   82.809 ms | Steady State:    0.574 ms
#       PyTorch  First Run:    0.690 ms | Steady State:    0.128 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  258.376 ms | Steady State:    0.171 ms
#       JAX      First Run:   28.411 ms | Steady State:    0.406 ms
#       PyTorch  First Run:   70.556 ms | Steady State:    0.120 ms

# -------------------- PATTERN: JVP (MLP Layer) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    1.083 ms | Steady State:    0.979 ms
#       JAX      First Run:  198.686 ms | Steady State:    1.843 ms
#       PyTorch  First Run:   50.194 ms | Steady State:    0.422 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  228.029 ms | Steady State:    0.403 ms
#       JAX      First Run:   30.841 ms | Steady State:    1.272 ms
#       PyTorch  First Run:  116.111 ms | Steady State:    0.379 ms

# -------------------- PATTERN: VJP (MLP Layer) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.927 ms | Steady State:    0.764 ms
#       JAX      First Run:  144.543 ms | Steady State:    1.594 ms
#       PyTorch  First Run:    0.813 ms | Steady State:    0.426 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  194.012 ms | Steady State:    0.427 ms
#       JAX      First Run:   30.676 ms | Steady State:    1.206 ms
#       PyTorch  First Run:   90.585 ms | Steady State:    0.329 ms

# -------------------- PATTERN: Hessian-Vector Product --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    3.728 ms | Steady State:    2.893 ms
#       JAX      First Run:  134.636 ms | Steady State:    1.762 ms
#       PyTorch  First Run:    2.044 ms | Steady State:    1.169 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  205.996 ms | Steady State:    0.610 ms
#       JAX      First Run:   27.508 ms | Steady State:    0.011 ms
#       PyTorch  First Run:  159.558 ms | Steady State:    0.363 ms

# -------------------- PATTERN: JVP(VJP(MLP)) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    3.222 ms | Steady State:    2.744 ms
#       JAX      First Run:  160.200 ms | Steady State:    3.665 ms
#       PyTorch  First Run:    2.777 ms | Steady State:    1.806 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  224.690 ms | Steady State:    0.625 ms
#       JAX      First Run:   30.635 ms | Steady State:    0.907 ms
#       PyTorch  First Run:  335.794 ms | Steady State:    0.586 ms

# -------------------- PATTERN: VJP(JVP(MLP)) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    3.571 ms | Steady State:    2.786 ms
#       JAX      First Run:  162.041 ms | Steady State:    3.074 ms
#       PyTorch  First Run:    2.068 ms | Steady State:    1.519 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  680.090 ms | Steady State:    1.091 ms
#       JAX      First Run:   32.121 ms | Steady State:    1.167 ms
#       PyTorch  First Run:  421.127 ms | Steady State:    0.552 ms

# -------------------- PATTERN: JVP(JVP(MLP)) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    3.491 ms | Steady State:    3.195 ms
#       JAX      First Run:  100.704 ms | Steady State:    3.471 ms
#       PyTorch  First Run:    1.997 ms | Steady State:    1.355 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  595.018 ms | Steady State:    1.168 ms
#       JAX      First Run:   31.096 ms | Steady State:    1.376 ms
#       PyTorch  First Run:  214.329 ms | Steady State:    0.809 ms

# -------------------- PATTERN: VJP(VJP(MLP)) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    3.421 ms | Steady State:    2.147 ms
#       JAX      First Run:  161.043 ms | Steady State:    3.129 ms
#       PyTorch  First Run:    2.310 ms | Steady State:    0.980 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  207.324 ms | Steady State:    0.691 ms
#       JAX      First Run:   31.329 ms | Steady State:    0.896 ms
#       PyTorch  First Run:  266.673 ms | Steady State:    0.451 ms

# Raw results saved to benchmark_results.csv

# --- STEADY STATE PERFORMANCE ---


# ==============================================================================================================
#                                   INTRA-FRAMEWORK JIT SPEEDUP (Eager vs JIT)
# ==============================================================================================================
#                                       Nabla Speedup (x) JAX Speedup (x) PyTorch Speedup (x)
# Benchmark      Pattern
# TRANSFORMER    Transformer Bwd Pass               1.85x           1.10x               0.91x
# ADVANCED-BENCH Hessian-Vector Product             4.74x         165.30x               3.22x
#                JVP (MLP Layer)                    2.43x           1.45x               1.11x
#                JVP(JVP(MLP))                      2.74x           2.52x               1.67x
#                JVP(VJP(MLP))                      4.39x           4.04x               3.08x
#                VJP (MLP Layer)                    1.79x           1.32x               1.30x
#                VJP(JVP(MLP))                      2.55x           2.63x               2.75x
#                VJP(VJP(MLP))                      3.11x           3.49x               2.17x
#                Vmap (MLP Layer)                   4.60x           1.41x               1.07x
# MACRO-BENCH    Deep MLP (16 Layers)               1.38x           1.07x               1.16x
#                Element-wise                       0.86x           1.42x               2.22x
#                MLP Layer                          1.21x           1.06x               1.00x
#                Matmul                             0.72x           1.04x               0.96x
#                Reduction                         11.41x           2.56x               1.78x
# MICRO-BENCH    Element-wise                       0.47x           6.56x               1.25x
#                MLP Layer                          0.18x           8.06x               0.38x
#                Matmul                             0.22x           1.82x               0.27x
#                Reduction                          0.55x           4.98x               1.21x


# ==============================================================================================================
#                                CROSS-FRAMEWORK JIT COMPARISON (LOWER IS BETTER)
# ==============================================================================================================
#                                       Best Framework Nabla Time (ms) Nabla Rel. to Best JAX Time (ms) JAX Rel. to Best PyTorch Time (ms) PyTorch Rel. to Best
# Benchmark      Pattern
# TRANSFORMER    Transformer Bwd Pass          PyTorch          90.673              1.01x       173.074            1.93x           89.730*                1.00x
# ADVANCED-BENCH Hessian-Vector Product            JAX           0.610             57.23x        0.011*            1.00x             0.363               34.09x
#                JVP (MLP Layer)               PyTorch           0.403              1.06x         1.272            3.35x            0.379*                1.00x
#                JVP(JVP(MLP))                 PyTorch           1.168              1.44x         1.376            1.70x            0.809*                1.00x
#                JVP(VJP(MLP))                 PyTorch           0.625              1.07x         0.907            1.55x            0.586*                1.00x
#                VJP (MLP Layer)               PyTorch           0.427              1.30x         1.206            3.67x            0.329*                1.00x
#                VJP(JVP(MLP))                 PyTorch           1.091              1.98x         1.167            2.11x            0.552*                1.00x
#                VJP(VJP(MLP))                 PyTorch           0.691              1.53x         0.896            1.99x            0.451*                1.00x
#                Vmap (MLP Layer)              PyTorch           0.171              1.43x         0.406            3.40x            0.120*                1.00x
# MACRO-BENCH    Deep MLP (16 Layers)            Nabla         13.602*              1.00x        36.125            2.66x            14.683                1.08x
#                Element-wise                  PyTorch           1.847              5.29x         0.727            2.08x            0.349*                1.00x
#                MLP Layer                     PyTorch           1.006              1.12x         2.619            2.92x            0.896*                1.00x
#                Matmul                        PyTorch           0.531              1.52x         1.309            3.74x            0.350*                1.00x
#                Reduction                     PyTorch           0.088              1.54x         0.141            2.47x            0.057*                1.00x
# MICRO-BENCH    Element-wise                      JAX           0.100             25.65x        0.004*            1.00x             0.005                1.31x
#                MLP Layer                         JAX           0.195             29.36x        0.007*            1.00x             0.008                1.24x
#                Matmul                            JAX           0.075             14.48x        0.005*            1.00x             0.006                1.21x
#                Reduction                     PyTorch           0.157             23.47x         0.007            1.05x            0.007*                1.00x


# ==============================================================================================================
#                                INTRA-FRAMEWORK JIT SPEEDUP (Grad vs JIT(Grad))
# ==============================================================================================================
#                                  Nabla Speedup (x) JAX Speedup (x) PyTorch Speedup (x)
# Benchmark   Pattern
# MACRO-BENCH Deep MLP (16 Layers)             2.29x           1.16x               0.96x
#             Element-wise                     8.80x           7.10x               4.51x
#             MLP Layer                        2.55x           1.21x               1.00x
#             Matmul                           3.92x           2.16x               1.53x
#             Reduction                       16.99x           7.47x               4.19x
# MICRO-BENCH Element-wise                     4.11x         198.60x              10.87x
#             MLP Layer                        1.23x         106.69x               4.14x
#             Matmul                           0.94x          49.95x               3.87x
#             Reduction                        2.23x         128.85x               9.66x


# ==============================================================================================================
#                          CROSS-FRAMEWORK JIT COMPARISON (GRADIENT - LOWER IS BETTER)
# ==============================================================================================================
#                                  Best Framework Nabla Time (ms) Nabla Rel. to Best JAX Time (ms) JAX Rel. to Best PyTorch Time (ms) PyTorch Rel. to Best
# Benchmark   Pattern
# MACRO-BENCH Deep MLP (16 Layers)          Nabla         29.946*              1.00x        75.337            2.52x            33.279                1.11x
#             Element-wise                    JAX           0.659              1.25x        0.527*            1.00x             0.664                1.26x
#             MLP Layer                     Nabla          1.545*              1.00x         5.092            3.30x             1.927                1.25x
#             Matmul                        Nabla          0.427*              1.00x         1.508            3.53x             0.610                1.43x
#             Reduction                   PyTorch           0.198              1.05x         0.205            1.08x            0.189*                1.00x
# MICRO-BENCH Element-wise                    JAX           0.056             11.95x        0.005*            1.00x             0.008                1.73x
#             MLP Layer                       JAX           0.123             16.71x        0.007*            1.00x             0.012                1.58x
#             Matmul                          JAX           0.080             11.87x        0.007*            1.00x             0.010                1.52x
#             Reduction                       JAX           0.140             21.46x        0.007*            1.00x             0.010                1.46x


# ============================================================
#              OVERALL JIT SCORE: ADVANCED-BENCH
# (Geometric Mean of Performance Relative to Best - Lower is Better)
# ============================================================
# Framework
# PyTorch   1.55x
# JAX       2.15x
# Nabla     2.19x


# ============================================================
#                OVERALL JIT SCORE: MACRO-BENCH
# (Geometric Mean of Performance Relative to Best - Lower is Better)
# ============================================================
# Framework
# PyTorch   1.02x
# Nabla     1.69x
# JAX       2.72x


# ============================================================
#             OVERALL JIT(GRAD) SCORE: MACRO-BENCH
# (Geometric Mean of Performance Relative to Best - Lower is Better)
# ============================================================
# Framework
# Nabla     1.06x
# PyTorch   1.20x
# JAX       2.00x


# ============================================================
#                OVERALL JIT SCORE: MICRO-BENCH
# (Geometric Mean of Performance Relative to Best - Lower is Better)
# ============================================================
# Framework
# JAX        1.01x
# PyTorch    1.18x
# Nabla     22.49x


# ============================================================
#             OVERALL JIT(GRAD) SCORE: MICRO-BENCH
# (Geometric Mean of Performance Relative to Best - Lower is Better)
# ============================================================
# Framework
# JAX        1.00x
# PyTorch    1.57x
# Nabla     15.02x


# ============================================================
#                OVERALL JIT SCORE: TRANSFORMER
# (Geometric Mean of Performance Relative to Best - Lower is Better)
# ============================================================
# Framework
# PyTorch   1.00x
# Nabla     1.01x
# JAX       1.93x

# --- FIRST RUN (COMPILATION OVERHEAD) PERFORMANCE ---


# ==============================================================================================================
#                        CROSS-FRAMEWORK JIT COMPILATION TIME (FORWARD - LOWER IS BETTER)
# ==============================================================================================================
#                                       Best Framework Nabla Time (ms) Nabla Rel. to Best JAX Time (ms) JAX Rel. to Best PyTorch Time (ms) PyTorch Rel. to Best
# Benchmark      Pattern
# TRANSFORMER    Transformer Bwd Pass              JAX         372.617              1.45x      256.625*            1.00x           415.183                1.62x
# ADVANCED-BENCH Hessian-Vector Product            JAX         205.996              7.49x       27.508*            1.00x           159.558                5.80x
#                JVP (MLP Layer)                   JAX         228.029              7.39x       30.841*            1.00x           116.111                3.76x
#                JVP(JVP(MLP))                     JAX         595.018             19.13x       31.096*            1.00x           214.329                6.89x
#                JVP(VJP(MLP))                     JAX         224.690              7.33x       30.635*            1.00x           335.794               10.96x
#                VJP (MLP Layer)                   JAX         194.012              6.32x       30.676*            1.00x            90.585                2.95x
#                VJP(JVP(MLP))                     JAX         680.090             21.17x       32.121*            1.00x           421.127               13.11x
#                VJP(VJP(MLP))                     JAX         207.324              6.62x       31.329*            1.00x           266.673                8.51x
#                Vmap (MLP Layer)                  JAX         258.376              9.09x       28.411*            1.00x            70.556                2.48x
# MACRO-BENCH    Deep MLP (16 Layers)              JAX         272.538              3.20x       85.285*            1.00x           118.211                1.39x
#                Element-wise                  PyTorch         219.130              8.84x        27.866            1.12x           24.793*                1.00x
#                MLP Layer                     PyTorch         262.869              9.83x        30.240            1.13x           26.755*                1.00x
#                Matmul                        PyTorch         212.481              9.54x        25.078            1.13x           22.273*                1.00x
#                Reduction                     PyTorch         194.185              8.35x        31.329            1.35x           23.263*                1.00x
# MICRO-BENCH    Element-wise                      JAX         203.292              5.77x       35.239*            1.00x          1975.524               56.06x
#                MLP Layer                     PyTorch         265.125              9.26x        33.601            1.17x           28.641*                1.00x
#                Matmul                            JAX         200.542             10.54x       19.026*            1.00x            21.525                1.13x
#                Reduction                     PyTorch         264.055             10.04x        34.440            1.31x           26.298*                1.00x


# ==============================================================================================================
#                       CROSS-FRAMEWORK JIT COMPILATION TIME (GRADIENT - LOWER IS BETTER)
# ==============================================================================================================
#                                  Best Framework Nabla Time (ms) Nabla Rel. to Best JAX Time (ms) JAX Rel. to Best PyTorch Time (ms) PyTorch Rel. to Best
# Benchmark   Pattern
# MACRO-BENCH Deep MLP (16 Layers)            JAX         293.552              1.90x      154.842*            1.00x           306.885                1.98x
#             Element-wise                    JAX         216.963              8.98x       24.151*            1.00x            77.286                3.20x
#             MLP Layer                       JAX         199.414              4.06x       49.118*            1.00x            82.378                1.68x
#             Matmul                          JAX         220.731              7.71x       28.643*            1.00x            71.214                2.49x
#             Reduction                       JAX         309.737              7.53x       41.120*            1.00x            77.253                1.88x
# MICRO-BENCH Element-wise                    JAX         203.329              4.38x       46.474*            1.00x            96.668                2.08x
#             MLP Layer                       JAX         193.743              5.71x       33.925*            1.00x           208.531                6.15x
#             Matmul                          JAX         198.872              7.66x       25.963*            1.00x            73.945                2.85x
#             Reduction                       JAX         205.870              5.59x       36.828*            1.00x            79.173                2.15x

# --- GENERATING PLOTS ---

# Generated plot: benchmark_micro-bench_jit.png

# Generated plot: benchmark_micro-bench_jit_grad.png

# Generated plot: benchmark_macro-bench_jit.png

# Generated plot: benchmark_macro-bench_jit_grad.png

# Generated plot: benchmark_transformer_jit.png

# Generated plot: benchmark_advanced-bench_jit.png

# Could not generate plot: Data for Transform='JIT(Grad)' and Benchmark='ADVANCED-BENCH' not found.
