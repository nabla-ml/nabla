from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Framework Imports (Assumed to be available) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.nn as jax_nn
import jax.numpy as jnp
import torch
import torch._dynamo
import torch.func
import torch.nn.functional as torch_functional

import nabla as nb

# --- Version Information ---
NABLA_VERSION = "0.1.0"
JAX_VERSION = jax.__version__
TORCH_VERSION = torch.__version__

# --- Global Configuration ---
SEED = 42
TIMEIT_RUNS_DEFAULT = 10
TIMEIT_REPEATS_DEFAULT = 3


class BColors:
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
# 1. HELPER & DATA GENERATION CLASSES
# ============================================================================
class FrameworkAPIs(NamedTuple):
    numpy: Any
    nabla: Any
    jax: Any
    torch: Any


class DataManager:
    """Handles creation of tensors for different frameworks from a single seed."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.apis = FrameworkAPIs(np, nb, jnp, torch)

    def get_tensor(self, shape: tuple, framework: str, dtype: str = "float32") -> Any:
        """Generates a tensor for a specific framework."""
        if isinstance(shape, list) or (
            isinstance(shape, tuple) and len(shape) > 0 and isinstance(shape[0], list)
        ):
            # Handle list of shapes for MLP weights/biases
            return [
                self.get_tensor(s, framework, dt)
                for s, dt in zip(shape[0], shape[1], strict=True)
            ]

        numpy_data = None
        if shape == ():  # Scalar
            if "int" in dtype:
                numpy_data = self.rng.integers(0, 100, dtype=dtype)
            else:
                numpy_data = self.rng.random(dtype=np.float32)
        else:
            if dtype == "int64":
                high = shape[0] if len(shape) > 0 else 100
                numpy_data = self.rng.integers(0, high, shape).astype(np.int64)
            elif dtype == "int32":
                high = shape[0] if len(shape) > 0 else 100
                numpy_data = self.rng.integers(0, high, shape, dtype=np.int32)
            else:
                numpy_data = self.rng.random(shape, dtype=np.float32)

        if framework == "nabla":
            return nb.tensor(numpy_data)
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
    arg_shapes: list
    arg_dtypes: list[str] | None = None

    def __post_init__(self):
        if self.arg_dtypes is None:
            self.arg_dtypes = ["float32"] * len(self.arg_shapes)


@dataclass
class Transformation:
    name: str
    nabla_transform: Callable
    jax_transform: Callable
    torch_transform: Callable


def get_patterns_and_transforms(dims: dict[str, int]):
    """Creates all benchmark patterns and transformations based on input dimensions."""
    N, D, B, S, E = dims["N"], dims["D"], dims["B"], dims["S"], dims["E"]  # noqa: N806
    N_LAYERS_MLP = dims.get("N_LAYERS_MLP", 8)  # noqa: N806

    basic_patterns = [
        Pattern(
            name="Element-wise",
            nabla_func=lambda x: nb.sum(nb.tanh(x) * nb.sin(x) / (nb.abs(x) + 1e-6)),
            jax_func=lambda x: jnp.sum(jnp.tanh(x) * jnp.sin(x) / (jnp.abs(x) + 1e-6)),
            torch_func=lambda x: torch.sum(
                torch.tanh(x) * torch.sin(x) / (torch.abs(x) + 1e-6)
            ),
            arg_shapes=[(N, D)],
        ),
        Pattern(
            name="Matmul",
            nabla_func=lambda x, w: nb.sum(x @ w),
            jax_func=lambda x, w: jnp.sum(x @ w),
            torch_func=lambda x, w: torch.sum(x @ w),
            arg_shapes=[(N, D), (D, N)],
        ),
        Pattern(
            name="Reduction",
            nabla_func=lambda x: nb.sum(
                nb.sqrt(nb.mean((x - nb.mean(x, axes=1, keep_dims=True)) ** 2, axes=1))
            ),
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
            nabla_func=lambda x, w, b: nb.sum(nb.relu(x @ w + b)),
            jax_func=lambda x, w, b: jnp.sum(jax.nn.relu(x @ w + b)),
            torch_func=lambda x, w, b: torch.sum(torch_functional.relu(x @ w + b)),
            arg_shapes=[(N, D), (D, D), (D,)],
        ),
    ]

    def _nabla_transformer_fwd(x, y, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        attn_out = nb.softmax(q @ k.transpose((0, 2, 1)) / (E**0.5), axis=-1) @ v
        x = x + attn_out @ w_o
        x = x + nb.relu(x @ w_ff1) @ w_ff2
        stable_x = x - nb.max(x, axes=-1, keep_dims=True)
        log_probs = stable_x - nb.log(nb.sum(nb.exp(stable_x), axes=-1, keep_dims=True))
        return -nb.sum(y * log_probs) / B

    def _jax_transformer_fwd(x, y, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        attn_out = jax_nn.softmax(q @ k.transpose(0, 2, 1) / (E**0.5), axis=-1) @ v
        x = x + attn_out @ w_o
        x = x + jax_nn.relu(x @ w_ff1) @ w_ff2
        return -jnp.sum(y * jax_nn.log_softmax(x, axis=-1)) / B

    def _torch_transformer_loss(transformer_params, x_input, y_target):
        w_q, w_k, w_v, w_o, w_ff1, w_ff2 = transformer_params
        q, k, v = x_input @ w_q, x_input @ w_k, x_input @ w_v
        attn_out = (
            torch_functional.softmax(q @ k.transpose(-2, -1) / (E**0.5), dim=-1) @ v
        )
        x_intermediate = x_input + attn_out @ w_o
        x_final = x_intermediate + torch_functional.relu(x_intermediate @ w_ff1) @ w_ff2
        return -torch.sum(y_target * torch_functional.log_softmax(x_final, dim=-1)) / B

    def _torch_transformer_grad_wrapper(x, y, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
        params = [w_q, w_k, w_v, w_o, w_ff1, w_ff2]
        grad_params = [p.clone().detach().requires_grad_(True) for p in params]
        loss = _torch_transformer_loss(grad_params, x, y)
        return torch.autograd.grad(loss, grad_params)

    transformer_pattern = Pattern(
        name="Transformer Bwd Pass",
        nabla_func=nb.grad(_nabla_transformer_fwd, argnums=tuple(range(2, 8))),
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

    def _define_deep_mlp_funcs(layers: int):
        def _nabla_deep_mlp(x, weights, biases):
            for i in range(layers):
                x = nb.relu(x @ weights[i] + biases[i])
            return nb.sum(x)

        def _jax_deep_mlp(x, weights, biases):
            for i in range(layers):
                x = jax_nn.relu(x @ weights[i] + biases[i])
            return jnp.sum(x)

        def _torch_deep_mlp(x, weights, biases):
            for i in range(layers):
                x = torch_functional.relu(x @ weights[i] + biases[i])
            return torch.sum(x)

        return _nabla_deep_mlp, _jax_deep_mlp, _torch_deep_mlp

    _nabla_mlp, _jax_mlp, _torch_mlp = _define_deep_mlp_funcs(layers=N_LAYERS_MLP)
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

    standard_transforms = [
        Transformation("Eager", lambda f: f, lambda f: f, lambda f: f),
        Transformation("Grad", nb.grad, jax.grad, torch.func.grad),  # type: ignore[attr-defined]
        Transformation(
            "JIT", nb.jit, jax.jit, lambda f: torch.compile(f, mode="max-autotune")
        ),
        Transformation(
            "JIT(Grad)",
            lambda f: nb.jit(nb.grad(f)),
            lambda f: jax.jit(jax.grad(f)),
            lambda f: torch.compile(torch.func.grad(f), mode="max-autotune"),  # type: ignore[attr-defined]
        ),
    ]
    transformer_transforms = [
        Transformation("Eager", lambda f: f, lambda f: f, lambda f: f),
        Transformation(
            "JIT", nb.jit, jax.jit, lambda f: torch.compile(f, mode="max-autotune")
        ),
    ]

    # --- Advanced Patterns (VMap, JVP, VJP, etc.) ---
    def _nabla_mlp_simple(x, w, b):
        return nb.sum(nb.relu(x @ w + b))

    def _jax_mlp_simple(x, w, b):
        return jnp.sum(jax.nn.relu(x @ w + b))

    def _torch_mlp_simple(x, w, b):
        return torch.sum(torch_functional.relu(x @ w + b))

    def _nabla_mlp_layer_nosum(x, w, b):
        return nb.relu(x @ w + b)

    vmap_pattern = Pattern(
        "Vmap (MLP Layer)",
        nb.vmap(_nabla_mlp_simple, in_axes=(0, None, None)),
        jax.vmap(_jax_mlp_simple, in_axes=(0, None, None)),
        torch.func.vmap(_torch_mlp_simple, in_dims=(0, None, None)),  # type: ignore[attr-defined]
        arg_shapes=[(N, D), (D, D), (D,)],
    )
    jvp_pattern = Pattern(
        "JVP (MLP Layer)",
        lambda x, w, b, t_x: nb.jvp(
            _nabla_mlp_simple, (x, w, b), (t_x, nb.ones_like(w), nb.ones_like(b))
        )[1],
        lambda x, w, b, t_x: jax.jvp(
            _jax_mlp_simple, (x, w, b), (t_x, jnp.ones_like(w), jnp.ones_like(b))
        )[1],
        lambda x, w, b, t_x: torch.func.jvp(
            _torch_mlp_simple, (x, w, b), (t_x, torch.ones_like(w), torch.ones_like(b))
        )[1],  # type: ignore[attr-defined]
        arg_shapes=[(N, D), (D, D), (D,), (N, D)],
    )
    vjp_pattern = Pattern(
        "VJP (MLP Layer)",
        lambda x, w, b, ct: nb.vjp(_nabla_mlp_layer_nosum, x, w, b)[1](ct),
        lambda x, w, b, ct: jax.vjp(
            lambda *a: jax.nn.relu(a[0] @ a[1] + a[2]), x, w, b
        )[1](ct),
        lambda x, w, b, ct: torch.func.vjp(
            lambda *a: torch_functional.relu(a[0] @ a[1] + a[2]), x, w, b
        )[1](ct),  # type: ignore[attr-defined]
        arg_shapes=[(N, D), (D, D), (D,), (N, D)],
    )
    hvp_pattern = Pattern(
        "Hessian-Vector Product",
        lambda x, w, b, t_x: nb.jvp(
            lambda x_in: nb.grad(_nabla_mlp_simple, argnums=0)(x_in, w, b), (x,), (t_x,)
        )[1],
        lambda x, w, b, t_x: jax.jvp(
            lambda x_in: jax.grad(_jax_mlp_simple, argnums=0)(x_in, w, b), (x,), (t_x,)
        )[1],
        lambda x, w, b, t_x: torch.func.jvp(  # type: ignore[attr-defined]
            lambda x_in: torch.func.grad(_torch_mlp_simple, argnums=0)(x_in, w, b),  # type: ignore[attr-defined]
            (x,),
            (t_x,),
        )[1],
        arg_shapes=[(N, D), (D, D), (D,), (N, D)],
    )
    jvp_vjp_pattern = Pattern(
        "JVP(VJP(MLP))",
        lambda x, w, b, ct, t_x: nb.jvp(
            lambda *a: nb.vjp(_nabla_mlp_simple, *a)[1](ct)[0],
            (x, w, b),
            (t_x, nb.ones_like(w), nb.ones_like(b)),
        )[1],
        lambda x, w, b, ct, t_x: jax.jvp(
            lambda *a: jax.vjp(_jax_mlp_simple, *a)[1](ct)[0],
            (x, w, b),
            (t_x, jnp.ones_like(w), jnp.ones_like(b)),
        )[1],
        lambda x, w, b, ct, t_x: torch.func.jvp(
            lambda *a: torch.func.vjp(_torch_mlp_simple, *a)[1](ct)[0],
            (x, w, b),
            (t_x, torch.ones_like(w), torch.ones_like(b)),
        )[1],  # type: ignore[attr-defined]
        arg_shapes=[(N, D), (D, D), (D,), (), (N, D)],
    )
    vjp_jvp_pattern = Pattern(
        "VJP(JVP(MLP))",
        lambda x, w, b, t_x, ct: nb.vjp(
            lambda *a: nb.jvp(
                _nabla_mlp_simple, a, (t_x, nb.ones_like(w), nb.ones_like(b))
            )[1],
            x,
            w,
            b,
        )[1](ct),
        lambda x, w, b, t_x, ct: jax.vjp(
            lambda *a: jax.jvp(
                _jax_mlp_simple, a, (t_x, jnp.ones_like(w), jnp.ones_like(b))
            )[1],
            x,
            w,
            b,
        )[1](ct),
        lambda x, w, b, t_x, ct: torch.func.vjp(
            lambda *a: torch.func.jvp(
                _torch_mlp_simple, a, (t_x, torch.ones_like(w), torch.ones_like(b))
            )[1],
            x,
            w,
            b,
        )[1](ct),  # type: ignore[attr-defined]
        arg_shapes=[(N, D), (D, D), (D,), (N, D), ()],
    )
    advanced_patterns = [
        vmap_pattern,
        jvp_pattern,
        vjp_pattern,
        hvp_pattern,
        jvp_vjp_pattern,
        vjp_jvp_pattern,
    ]
    advanced_transforms = [
        Transformation("Eager", lambda f: f, lambda f: f, lambda f: f),
        Transformation(
            "JIT", nb.jit, jax.jit, lambda f: torch.compile(f, mode="max-autotune")
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


def create_mlp_pattern(N: int, D: int) -> Pattern:  # noqa: N803
    """Creates a standard MLP Layer pattern for given dimensions N and D."""
    return Pattern(
        name=f"MLP Layer {N}x{D}",
        nabla_func=lambda x, w, b: nb.sum(nb.relu(x @ w + b)),
        jax_func=lambda x, w, b: jnp.sum(jax.nn.relu(x @ w + b)),
        torch_func=lambda x, w, b: torch.sum(torch_functional.relu(x @ w + b)),
        arg_shapes=[(N, D), (D, D), (D,)],
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
        frameworks: list[str],
        **kwargs,
    ):
        self.config_name = config_name
        self.dm = data_manager
        self.timeit_runs = timeit_runs
        self.timeit_repeats = timeit_repeats
        self.frameworks_to_run = frameworks
        self.kwargs = kwargs
        self.results: list[dict] = []
        print(
            f"\n\n{BColors.HEADER}{'=' * 80}\n"
            f" {self.config_name.center(78)} \n"
            f" {str(self.kwargs).center(78)} \n"
            f"{'=' * 80}{BColors.ENDC}"
        )

    def run(
        self, patterns: list[Pattern], transforms: list[Transformation]
    ) -> pd.DataFrame:
        """Runs a set of patterns through a set of transformations."""
        torch._dynamo.reset()
        for pattern in patterns:
            print(
                f"\n{BColors.OKBLUE}{'-' * 20} PATTERN: {pattern.name} {'-' * 20}{BColors.ENDC}"
            )
            for trans in transforms:
                print(f"  {BColors.OKCYAN}--> MODE: {trans.name}{BColors.ENDC}")
                self._run_single_test(pattern, trans)
        return pd.DataFrame(self.results)

    def _run_single_test(self, pattern: Pattern, trans: Transformation):
        """Prepares arguments and executes the benchmark for all frameworks."""
        # The __post_init__ call ensures arg_dtypes is not None, but we assert for static analysis.
        assert pattern.arg_dtypes is not None, "Pattern dtypes must be initialized"
        try:
            args_nabla = (
                tuple(
                    self.dm.get_tensor(s, "nabla", dt)
                    for s, dt in zip(
                        pattern.arg_shapes, pattern.arg_dtypes, strict=True
                    )
                )
                if "Nabla" in self.frameworks_to_run
                else None
            )

            args_jax = (
                tuple(
                    self.dm.get_tensor(s, "jax", dt)
                    for s, dt in zip(
                        pattern.arg_shapes, pattern.arg_dtypes, strict=True
                    )
                )
                if "JAX" in self.frameworks_to_run
                else None
            )

            args_torch = (
                tuple(
                    self.dm.get_tensor(s, "torch", dt)
                    for s, dt in zip(
                        pattern.arg_shapes, pattern.arg_dtypes, strict=True
                    )
                )
                if "PyTorch" in self.frameworks_to_run
                else None
            )
        except Exception as e:
            print(
                f"      {BColors.FAIL}Argument Preparation FAILED for {pattern.name}: {e}{BColors.ENDC}"
            )
            self.results.append(
                {
                    "Benchmark": self.config_name,
                    "Pattern": pattern.name,
                    "Transform": trans.name,
                    "Framework": "All",
                    "Error": "Argument Prep Failed",
                }
            )
            return

        framework_funcs = {
            "Nabla": (trans.nabla_transform(pattern.nabla_func), args_nabla),
            "JAX": (trans.jax_transform(pattern.jax_func), args_jax),
            "PyTorch": (trans.torch_transform(pattern.torch_func), args_torch),
        }

        for fw_name in self.frameworks_to_run:
            func_tuple = framework_funcs.get(fw_name)
            # The `all(func_tuple)` check ensures neither the function nor args are None.
            if func_tuple and all(func_tuple):
                func, args = func_tuple
                assert args is not None, "Args should not be None here"
                self._measure_and_store(fw_name, func, args, pattern.name, trans.name)
            else:
                self.results.append(
                    {
                        "Benchmark": self.config_name,
                        "Pattern": pattern.name,
                        "Transform": trans.name,
                        "Framework": fw_name,
                        "Time (ms)": np.nan,
                        "First Run Time (ms)": np.nan,
                        "Error": "Skipped (N/A)",
                    }
                )

    def _measure_and_store(
        self,
        fw_name: str,
        func: Callable,
        args: tuple[Any, ...],
        pattern_name: str,
        trans_name: str,
    ):
        """
        Measures and records the performance of a given function. This method follows a rigorous
        procedure to ensure fair and accurate timing:
        1.  WARM-UP RUN: A single, untimed call is made first. For JIT-compiled functions,
            this triggers the compilation and caches the result. This ensures we are not
            measuring the one-time compilation cost in our steady-state analysis. This run
            is timed separately to report the "First Run Time (ms)" which includes overhead.
        2.  SYNCHRONIZATION: For asynchronous frameworks (JAX, PyTorch on GPU), a synchronization
            call (`jax.block_until_ready`, `torch.synchronize`) is placed *inside* the
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
            # Create a wrapper that includes the appropriate synchronization call
            if fw_name == "JAX":

                def timed_func():
                    res = func(*args)
                    jax.block_until_ready(res)
            elif fw_name == "PyTorch":
                # torch.cpu.synchronize() is a no-op but included for consistency
                def timed_func():
                    func(*args)
                    torch.cpu.synchronize()
            else:  # Nabla is synchronous

                def timed_func():
                    func(*args)

            # 1. Warm-up and First Run Time
            start_time_first_run = time.perf_counter()
            timed_func()
            first_run_ms = (time.perf_counter() - start_time_first_run) * 1000

            # 2. Steady-State Timing
            timer = timeit.Timer(stmt=timed_func)
            times = timer.repeat(repeat=self.timeit_repeats, number=self.timeit_runs)
            steady_state_ms = (min(times) / self.timeit_runs) * 1000

            print(
                f"      {fw_name:<8} First Run: {first_run_ms:8.3f} ms | Steady State: {steady_state_ms:8.3f} ms"
            )
            self.results.append(
                {
                    "Benchmark": self.config_name,
                    "Pattern": pattern_name,
                    "Transform": trans_name,
                    "Framework": fw_name,
                    "Time (ms)": steady_state_ms,
                    "First Run Time (ms)": first_run_ms,
                }
            )
        except Exception as e:
            print(
                f"      {BColors.FAIL}{fw_name:<8} FAILED ({type(e).__name__}): {str(e)[:150]}{BColors.ENDC}"
            )
            self.results.append(
                {
                    "Benchmark": self.config_name,
                    "Pattern": pattern_name,
                    "Transform": trans_name,
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
    def __init__(self, df: pd.DataFrame, frameworks: list[str]):
        self.df = df.copy()
        self.frameworks = frameworks

    def _create_summary_df(self, time_column: str) -> pd.DataFrame:
        """Pivots the raw results for easier comparison."""
        df_subset = self.df.dropna(subset=[time_column])
        if df_subset.empty:
            return pd.DataFrame()

        pivot_df = df_subset.pivot_table(
            index=["Benchmark", "Pattern"],
            columns=["Framework", "Transform"],
            values=time_column,
        )
        if not pivot_df.empty:
            # Ensure framework order is consistent
            frameworks_in_pivot = pivot_df.columns.get_level_values(
                "Framework"
            ).unique()
            frameworks_to_reindex = [
                fw for fw in self.frameworks if fw in frameworks_in_pivot
            ]
            if frameworks_to_reindex:
                pivot_df = pivot_df.reindex(
                    frameworks_to_reindex, axis=1, level="Framework"
                )
        return pivot_df

    def _build_comprehensive_table(
        self,
        pivot_df: pd.DataFrame,
        base_transform: str,
        jit_transform: str,
        time_col_suffix: str,
    ) -> pd.DataFrame:
        """Builds a single table comparing Eager, JIT, and Speedup for all frameworks."""
        if pivot_df.empty:
            return pd.DataFrame()

        base_cols = [
            (fw, base_transform)
            for fw in self.frameworks
            if (fw, base_transform) in pivot_df.columns
        ]
        jit_cols = [
            (fw, jit_transform)
            for fw in self.frameworks
            if (fw, jit_transform) in pivot_df.columns
        ]

        relevant_cols = base_cols + jit_cols
        if not relevant_cols:
            return pd.DataFrame()

        filtered_pivot = pivot_df.dropna(how="all", subset=relevant_cols).copy()
        if filtered_pivot.empty:
            return pd.DataFrame()

        if jit_cols:
            filtered_pivot["Best JIT Time"] = filtered_pivot[jit_cols].min(
                axis="columns"
            )
        else:
            filtered_pivot["Best JIT Time"] = np.nan

        summary_data = []
        for idx, row_series in filtered_pivot.iterrows():
            assert isinstance(idx, tuple) and len(idx) >= 2, (
                "Index should be a tuple with at least 2 elements"
            )
            row_dict = {"Benchmark": idx[0], "Pattern": idx[1]}

            best_time_for_row = row_series["Best JIT Time"]

            for fw in self.frameworks:
                base_time = row_series.get((fw, base_transform), np.nan)
                jit_time = row_series.get((fw, jit_transform), np.nan)

                row_dict[f"{fw} {base_transform} {time_col_suffix}"] = base_time

                # --- FIX FOR RUNTIME ValueError ---
                # This block robustly checks if the current framework's JIT time is the best.
                is_best = False
                # Ensure both values are valid numbers before comparing.
                # Use try-except to safely handle the comparison
                try:
                    if not pd.isna(jit_time) and not pd.isna(best_time_for_row):
                        # Convert to float and compare with tolerance
                        jit_val = float(jit_time)
                        best_val = float(best_time_for_row)
                        is_best = np.isclose(jit_val, best_val)
                except (ValueError, TypeError):
                    is_best = False

                if is_best:
                    try:
                        row_dict[f"{fw} {jit_transform} {time_col_suffix}"] = (
                            f"{float(jit_time):8.3f}*"
                        )
                    except (ValueError, TypeError):
                        row_dict[f"{fw} {jit_transform} {time_col_suffix}"] = jit_time
                else:
                    row_dict[f"{fw} {jit_transform} {time_col_suffix}"] = jit_time

                # Calculate speedup safely
                try:
                    if not pd.isna(base_time) and not pd.isna(jit_time):
                        base_val = float(base_time)
                        jit_val = float(jit_time)
                        if jit_val > 1e-9:
                            row_dict[f"{fw} Speedup (x)"] = base_val / jit_val
                        else:
                            row_dict[f"{fw} Speedup (x)"] = np.nan
                    else:
                        row_dict[f"{fw} Speedup (x)"] = np.nan
                except (ValueError, TypeError):
                    row_dict[f"{fw} Speedup (x)"] = np.nan
            summary_data.append(row_dict)

        if not summary_data:
            return pd.DataFrame()

        final_df = pd.DataFrame(summary_data).set_index(["Benchmark", "Pattern"])
        col_order = []
        for fw in self.frameworks:
            col_order.extend(
                [
                    f"{fw} {base_transform} {time_col_suffix}",
                    f"{fw} {jit_transform} {time_col_suffix}",
                    f"{fw} Speedup (x)",
                ]
            )

        # Only include columns that were actually created.
        return final_df[[c for c in col_order if c in final_df.columns]]

    def display_table(self, df: pd.DataFrame, title: str):
        if df.empty:
            return
        print(
            f"\n\n{BColors.HEADER}{'=' * 110}\n{title.center(110)}\n{'=' * 110}{BColors.ENDC}"
        )

        # Sort benchmarks for consistent order
        all_benchmarks = df.index.get_level_values("Benchmark").unique()
        key_order = ["MICRO-BENCH", "MACRO-BENCH", "TRANSFORMER", "ADVANCED-BENCH"]
        sorted_benchmarks = [b for b in key_order if b in all_benchmarks]
        other_benchmarks = sorted(
            [b for b in all_benchmarks if b not in sorted_benchmarks]
        )
        df = df.reindex(sorted_benchmarks + other_benchmarks, level="Benchmark")

        # Formatters for clean printing
        formatters = {}
        for col in df.columns:
            if "Time (ms)" in col or "Compile Time (ms)" in col:
                formatters[col] = (
                    lambda x: f"{x:8.3f}"
                    if isinstance(x, int | float)
                    else (str(x) if pd.notna(x) else "-")
                )
            elif "Speedup" in col:
                formatters[col] = lambda x: f"{x:.2f}x" if pd.notna(x) else "-"

        with pd.option_context(
            "display.max_rows",
            None,
            "display.width",
            200,
            "display.colheader_justify",
            "right",
        ):
            print(df.to_string(formatters=formatters, na_rep="-"))

    def display_summary_score(
        self, pivot_df: pd.DataFrame, transform_name: str, title: str
    ):
        try:
            transform_df = pivot_df.xs(
                transform_name, level="Transform", axis=1
            ).dropna(how="all")
        except KeyError:
            return  # This transform doesn't exist for this benchmark

        if transform_df.empty:
            return

        best_times = transform_df.min(axis=1)  # type: ignore[arg-type]
        relative_perf = transform_df.div(best_times, axis=0)
        # Geometric mean ignores NaNs automatically
        geo_mean = np.exp(np.log(relative_perf).mean()).sort_values()

        print(
            f"\n\n{BColors.HEADER}{'=' * 70}\n{title.center(70)}\n"
            f"{'(Geometric Mean of Perf. Relative to Best - Lower is Better)'.center(70)}\n"
            f"{'=' * 70}{BColors.ENDC}"
        )
        print(geo_mean.to_string(float_format="%.2fx"))

    def plot_comparison(
        self, time_column: str, transform: str, benchmark: str, title_suffix: str
    ):
        pivot_df = self._create_summary_df(time_column)
        if pivot_df.empty:
            return

        try:
            # Check if the transform exists
            if transform not in pivot_df.columns.get_level_values("Transform"):
                print(
                    f"{BColors.WARNING}\nTransform '{transform}' not found in pivot table. Available transforms: {list(pivot_df.columns.get_level_values('Transform').unique())}{BColors.ENDC}"
                )
                return

            df_to_plot = pivot_df.xs(transform, level="Transform", axis=1).dropna(
                how="all"
            )
            if benchmark.startswith("SCALABILITY"):
                # For scalability, we need to handle multiple benchmark entries
                scalability_data = df_to_plot[
                    df_to_plot.index.get_level_values("Benchmark").str.startswith(
                        "SCALABILITY"
                    )
                ]
                if scalability_data.empty:
                    # Debug: print available benchmarks
                    available_benchmarks = (
                        df_to_plot.index.get_level_values("Benchmark").unique().tolist()
                    )
                    print(
                        f"{BColors.WARNING}\nCould not generate plot: No scalability data found for transform '{transform}'.{BColors.ENDC}"
                    )
                    print(
                        f"{BColors.WARNING}Available benchmarks: {available_benchmarks}{BColors.ENDC}"
                    )
                    return

                # Extract size from the benchmark name (e.g., "SCALABILITY-BENCH (Size=64)")
                benchmark_names = scalability_data.index.get_level_values("Benchmark")
                sizes = benchmark_names.str.extract(r"Size=(\d+)")[
                    0
                ]  # Extract size from benchmark name

                # Reset index to make manipulation easier
                scalability_reset = scalability_data.reset_index()
                scalability_reset["Size"] = (
                    pd.to_numeric(sizes, errors="coerce").fillna(0).astype(int)
                )

                # Melt the dataframe to get it in the right format for plotting
                df_to_plot = scalability_reset.melt(
                    id_vars=["Benchmark", "Pattern", "Size"],
                    var_name="Framework",
                    value_name="Time (ms)",
                )

                # Remove any rows with invalid sizes or missing data
                df_to_plot = df_to_plot.dropna(subset=["Size", "Time (ms)"])
                df_to_plot = df_to_plot[df_to_plot["Size"] > 0]  # Remove invalid sizes
            else:
                df_to_plot = df_to_plot.loc[benchmark].stack().reset_index()
                df_to_plot.columns = ["Pattern", "Framework", "Time (ms)"]
        except (KeyError, IndexError):
            print(
                f"{BColors.WARNING}\nCould not generate plot: Data for Transform='{transform}', Benchmark='{benchmark}' not found.{BColors.ENDC}"
            )
            return
        if df_to_plot.empty:
            print(
                f"{BColors.WARNING}\nCould not generate plot: No data for '{benchmark}' after filtering.{BColors.ENDC}"
            )
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        palette = "viridis"

        if benchmark.startswith("SCALABILITY"):
            sns.lineplot(
                data=df_to_plot,
                x="Size",
                y="Time (ms)",
                hue="Framework",
                marker="o",
                ax=ax,
                palette=palette,
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
                palette=palette,
            )
            ax.set_xlabel("Benchmark Pattern", fontsize=12)
            ax.tick_params(axis="x", rotation=15)
            if df_to_plot["Time (ms)"].max() / df_to_plot["Time (ms)"].min() > 10:
                ax.set_yscale("log")
                ax.set_ylabel("Time (ms) - Lower is Better (Log Scale)", fontsize=12)

        ax.set_ylabel("Time (ms) - Lower is Better", fontsize=12)
        ax.legend(title="Framework", fontsize=10)
        fig.tight_layout()

        filename = f"benchmark_{benchmark.lower().replace(' ', '_')}_{transform.lower().replace('(', '_').replace(')', '')}.png"
        plt.savefig(filename)
        print(f"\n{BColors.OKGREEN}Generated plot: {filename}{BColors.ENDC}")

    def process_and_display(self):
        if self.df.empty:
            print(f"{BColors.WARNING}No results to analyze.{BColors.ENDC}")
            return

        print(
            f"\n{BColors.OKBLUE}{BColors.BOLD}--- STEADY STATE PERFORMANCE ---{BColors.ENDC}"
        )
        pivot_steady = self._create_summary_df("Time (ms)")
        if not pivot_steady.empty:
            self.display_table(
                self._build_comprehensive_table(
                    pivot_steady, "Eager", "JIT", "Time (ms)"
                ),
                "FORWARD PASS PERFORMANCE (LOWER TIME | HIGHER SPEEDUP IS BETTER)",
            )
            self.display_table(
                self._build_comprehensive_table(
                    pivot_steady, "Grad", "JIT(Grad)", "Time (ms)"
                ),
                "GRADIENT PASS PERFORMANCE (LOWER TIME | HIGHER SPEEDUP IS BETTER)",
            )
            for bench_name in pivot_steady.index.get_level_values("Benchmark").unique():
                if not bench_name.startswith("SCALABILITY"):
                    bench_pivot = pivot_steady.loc[[bench_name]]
                    self.display_summary_score(
                        bench_pivot, "JIT", f"OVERALL JIT SCORE: {bench_name}"
                    )
                    self.display_summary_score(
                        bench_pivot,
                        "JIT(Grad)",
                        f"OVERALL JIT(GRAD) SCORE: {bench_name}",
                    )

        print(
            f"\n{BColors.OKBLUE}{BColors.BOLD}--- FIRST RUN (COMPILATION OVERHEAD) PERFORMANCE ---{BColors.ENDC}"
        )
        pivot_first_run = self._create_summary_df("First Run Time (ms)")
        if not pivot_first_run.empty:
            self.display_table(
                self._build_comprehensive_table(
                    pivot_first_run, "Eager", "JIT", "Compile Time (ms)"
                ),
                "FORWARD PASS COMPILATION OVERHEAD",
            )
            self.display_table(
                self._build_comprehensive_table(
                    pivot_first_run, "Grad", "JIT(Grad)", "Compile Time (ms)"
                ),
                "GRADIENT PASS COMPILATION OVERHEAD",
            )

        print(f"\n{BColors.OKBLUE}{BColors.BOLD}--- GENERATING PLOTS ---{BColors.ENDC}")
        benchmarks_to_plot = self.df["Benchmark"].unique()
        pivot_steady = self._create_summary_df("Time (ms)")

        if pivot_steady.empty:
            print(f"{BColors.WARNING}No data available for plotting.{BColors.ENDC}")
            return

        available_transforms = pivot_steady.columns.get_level_values(
            "Transform"
        ).unique()

        # Plot regular benchmarks
        for bench in benchmarks_to_plot:
            if bench.startswith("SCALABILITY"):
                continue

            # Only plot if the transform exists for this benchmark
            if "JIT" in available_transforms:
                try:
                    bench_data = pivot_steady.loc[[bench]]
                    if "JIT" in bench_data.columns.get_level_values("Transform"):
                        self.plot_comparison(
                            "Time (ms)", "JIT", bench, "JIT Steady State"
                        )
                except (KeyError, IndexError):
                    pass

            if "JIT(Grad)" in available_transforms:
                try:
                    bench_data = pivot_steady.loc[[bench]]
                    if "JIT(Grad)" in bench_data.columns.get_level_values("Transform"):
                        self.plot_comparison(
                            "Time (ms)", "JIT(Grad)", bench, "JIT(Grad) Steady State"
                        )
                except (KeyError, IndexError):
                    pass

        # Plot scalability benchmark if present
        if any(b.startswith("SCALABILITY") for b in benchmarks_to_plot):
            try:
                self.plot_comparison(
                    "Time (ms)",
                    "JIT(Grad)",
                    "SCALABILITY",
                    "MLP Layer Gradient Scaling",
                )
            except Exception as e:
                print(
                    f"{BColors.WARNING}Could not generate scalability plot: {e}{BColors.ENDC}"
                )


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def print_environment_info():
    print(f"{BColors.BOLD}Environment Information:{BColors.ENDC}")
    print(f"  - Python Version: {sys.version.split(' ')[0]}")
    print(
        f"  - Platform: {platform.system()} {platform.release()} ({platform.machine()})"
    )
    try:
        import cpuinfo

        print(f"  - CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    except ImportError:
        print("  - CPU: (Install 'py-cpuinfo' for details)")
    print(
        f"  - Library Versions:\n    - Nabla:   {NABLA_VERSION}\n    - JAX:     {JAX_VERSION}\n    - PyTorch: {TORCH_VERSION}\n    - NumPy:   {np.__version__}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Nabla/JAX/PyTorch Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["micro", "macro", "transformer", "advanced", "scalability"],
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

    print_environment_info()
    dm = DataManager(SEED)
    all_results = []

    print(
        f"\n{BColors.BOLD}Running benchmarks: {', '.join(args.benchmarks)}{BColors.ENDC}"
    )
    print(
        f"{BColors.BOLD}Testing frameworks: {', '.join(args.frameworks)}{BColors.ENDC}"
    )
    print(
        f"{BColors.BOLD}Timing config: {args.repeats} repeats, {args.runs} runs each.{BColors.ENDC}"
    )

    if "micro" in args.benchmarks:
        dims = {"N": 4, "D": 8, "B": 0, "S": 0, "E": 0}
        p, t, _, _, _, _, _ = get_patterns_and_transforms(dims)
        runner = BenchmarkRunner(
            "MICRO-BENCH", dm, args.runs, args.repeats, args.frameworks, **dims
        )
        all_results.append(runner.run(p, t))

    if "macro" in args.benchmarks:
        dims = {"N": 512, "D": 1024, "B": 0, "S": 0, "E": 0, "N_LAYERS_MLP": 16}
        b_p, s_t, _, _, mlp_p, _, _ = get_patterns_and_transforms(dims)
        runner = BenchmarkRunner(
            "MACRO-BENCH", dm, args.runs, args.repeats, args.frameworks, **dims
        )
        all_results.append(runner.run(b_p + [mlp_p], s_t))

    if "transformer" in args.benchmarks:
        dims = {"N": 0, "D": 0, "B": 16, "S": 256, "E": 512}
        _, _, trans_p, trans_t, _, _, _ = get_patterns_and_transforms(dims)
        runner = BenchmarkRunner(
            "TRANSFORMER", dm, args.runs, args.repeats, args.frameworks, **dims
        )
        all_results.append(runner.run([trans_p], trans_t))

    if "advanced" in args.benchmarks:
        dims = {"N": 256, "D": 512, "B": 0, "S": 0, "E": 0}
        _, _, _, _, _, adv_p, adv_t = get_patterns_and_transforms(dims)
        runner = BenchmarkRunner(
            "ADVANCED-BENCH", dm, args.runs, args.repeats, args.frameworks, **dims
        )
        all_results.append(runner.run(adv_p, adv_t))

    if "scalability" in args.benchmarks:
        jit_grad_transform = Transformation(
            "JIT(Grad)",
            lambda f: nb.jit(nb.grad(f)),
            lambda f: jax.jit(jax.grad(f)),
            lambda f: torch.compile(torch.func.grad(f), mode="max-autotune"),  # type: ignore[attr-defined]
        )
        for size in [64, 128, 256, 512, 1024, 2048]:
            dims = {"N": size, "D": size, "B": 0, "S": 0, "E": 0}
            mlp_pattern = create_mlp_pattern(N=size, D=size)
            runner = BenchmarkRunner(
                f"SCALABILITY-BENCH (Size={size})",
                dm,
                args.runs,
                args.repeats,
                args.frameworks,
                **dims,
            )
            all_results.append(runner.run([mlp_pattern], [jit_grad_transform]))

    if not all_results:
        print(f"\n{BColors.WARNING}No benchmark results were generated.{BColors.ENDC}")
        return

    all_results_df = pd.concat(all_results, ignore_index=True)

    if args.output_csv:
        all_results_df.to_csv(args.output_csv, index=False)
        print(
            f"\n{BColors.OKGREEN}Raw results saved to {args.output_csv}{BColors.ENDC}"
        )

    analyzer = ResultAnalyzer(all_results_df, args.frameworks)
    analyzer.process_and_display()


if __name__ == "__main__":
    main()
