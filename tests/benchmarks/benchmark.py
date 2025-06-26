"""
=======================================================================================
DEFINITIVE BENCHMARKING SUITE FOR NABLA, JAX, AND PYTORCH (CPU) - V15.4 (FAIR-COMPLEXITY & SCALED)
=======================================================================================
This version incorporates feedback to further refine benchmarking fairness and focus.
It removes the conditional control flow benchmark, ensures robust testing of complex,
JIT-friendly functions, corrects "First Run" timing to include JIT compilation
overhead, resolves PyTorch MLP Grad argument prep, and implements correct asynchronous
timing for JAX/PyTorch.
This version also increases problem sizes for MACRO/TRANSFORMER benchmarks and MLP
depth to better differentiate performance on more substantial workloads.
Finally, it adds relative speed comparison columns for Nabla against JAX/PyTorch.

Key Features:
1.  REMOVED CONTROL FLOW BENCHMARK.
2.  COMPLEXITY-AWARE JIT TEST: 'Deep MLP' (16 layers) and scaled 'Transformer'
    remain key tests for JIT performance.
3.  CORRECTED FIRST RUN TIMING: Includes JIT compilation.
4.  PYTORCH MLP GRAD FIX: Resolved NameError, refined argument prep.
5.  CORRECT ASYNC TIMING: Uses block_until_ready/synchronize in timeit loop.
6.  SCALED WORKLOADS: Increased dimensions for MACRO, TRANSFORMER, and MLP depth.
7.  RELATIVE SPEED COLUMNS: Shows Nabla performance vs JAX/PyTorch.
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

import numpy as np
import pandas as pd

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
    print("WARNING: Nabla library not found or import failed. Nabla benchmarks will fail.")
    nb = None # Placeholder
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
# 1. HELPER & DATA GENERATION CLASSES
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
        if isinstance(shape[0], list): # Handle list-based shapes for MLP weights/biases
            return [self.get_tensor(s, framework, dt) for s, dt in zip(shape[0], shape[1])]

        if dtype == "int64":
            high = shape[0] if len(shape) > 0 and isinstance(shape[0], int) else 100
            data = self.rng.integers(0, high, shape).astype(np.int64)
        elif dtype == "int32":
            high = shape[0] if len(shape) > 0 and isinstance(shape[0], int) else 100
            data = self.rng.integers(0, high, shape, dtype=np.int32)
        else:
            data = self.rng.random(shape, dtype=np.float32)

        if framework == "nabla":
            if nb is None: raise ImportError("Nabla not available for tensor creation.")
            return nb.array(data) if not isinstance(data, list) else [nb.array(d) for d in data]
        if framework == "jax":
            return jnp.array(data) if not isinstance(data, list) else [jnp.array(d) for d in data]
        if framework == "torch":
            return torch.from_numpy(data) if not isinstance(data, list) else [torch.from_numpy(d) for d in data]
        return data


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
    N_LAYERS_MLP = dims.get("N_LAYERS_MLP", 8) # Default to 8 if not specified

    # Basic Patterns (Nabla functions will be guarded if nb is None)
    basic_patterns = [
        Pattern(
            name="Element-wise",
            nabla_func=lambda x: nb.sum(nb.tanh(x) * nb.sin(x) / (nb.abs(x) + 1e-6)) if nb else None,
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
            ) if nb else None,
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
            name="MLP Layer", # Single MLP layer
            nabla_func=lambda x, w, b: nb.sum(nb.relu(x @ w + b)) if nb else None,
            jax_func=lambda x, w, b: jnp.sum(jax.nn.relu(x @ w + b)),
            torch_func=lambda x, w, b: torch.sum(F.relu(x @ w + b)),
            arg_shapes=[(N, D), (D, D), (D,)],
        ),
    ]

    # --- Transformer Pattern ---
    def _nabla_transformer_fwd(x, y, w_q, w_k, w_v, w_o, w_ff1, w_ff2):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        attn_out = nb.softmax(q @ k.transpose((0, 2, 1)) / (E**0.5), axis=-1) @ v
        x = x + attn_out @ w_o
        x = x + nb.relu(x @ w_ff1) @ w_ff2
        stable_x = x - nb.max(x, axes=-1, keep_dims=True)
        return -nb.sum(y * (stable_x - nb.log(nb.sum(nb.exp(stable_x), axes=-1, keep_dims=True)))) / B

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
        cloned_params_for_grad = []
        for p in params_list:
            if p.is_floating_point():
                p_clone = p.clone().detach().requires_grad_(True)
                cloned_params_for_grad.append(p_clone)
            else:
                cloned_params_for_grad.append(p)
        loss = _torch_transformer_loss(cloned_params_for_grad, x, y)
        grads = torch.autograd.grad(loss, cloned_params_for_grad)
        return grads

    transformer_pattern = Pattern(
        name="Transformer Bwd Pass",
        nabla_func=nb.grad(_nabla_transformer_fwd, argnums=tuple(range(2, 8))) if nb else None,
        jax_func=jax.grad(_jax_transformer_fwd, argnums=range(2, 8)),
        torch_func=_torch_transformer_grad_wrapper,
        arg_shapes=[(B, S, E), (B, S, E), (E, E), (E, E), (E, E), (E, E), (E, 4 * E), (4 * E, E)],
    )

    # --- Deep MLP Pattern for testing JIT fusion ---
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
            ([[(D, D)] * N_LAYERS_MLP, ["float32"] * N_LAYERS_MLP]), # Weights
            ([[(D,)] * N_LAYERS_MLP, ["float32"] * N_LAYERS_MLP]),   # Biases
        ],
    )

    # --- Transformations ---
    nabla_grad_placeholder = lambda f: (lambda *args, **kwargs: "Nabla Grad N/A") if nb is None or f is None else nb.grad(f)
    nabla_jit_placeholder = lambda f: (lambda *args, **kwargs: "Nabla JIT N/A") if nb is None or f is None else nb.jit(f)
    
    standard_transforms = [
        Transformation(name="Eager", nabla_transform=lambda f: f, jax_transform=lambda f: f, torch_transform=lambda f: f),
        Transformation(name="Grad", nabla_transform=nabla_grad_placeholder, jax_transform=jax.grad, torch_transform=torch.func.grad),
        Transformation(name="JIT", nabla_transform=nabla_jit_placeholder, jax_transform=jax.jit, torch_transform=torch.compile),
        Transformation(name="JIT(Grad)",
                       nabla_transform=lambda f: nabla_jit_placeholder(nabla_grad_placeholder(f)),
                       jax_transform=lambda f: jax.jit(jax.grad(f)),
                       torch_transform=lambda f: torch.compile(torch.func.grad(f))),
    ]
    transformer_transforms = [
        Transformation(name="Eager", nabla_transform=lambda f: f, jax_transform=lambda f: f, torch_transform=lambda f: f),
        Transformation(name="JIT", nabla_transform=nabla_jit_placeholder, jax_transform=jax.jit, torch_transform=torch.compile),
    ]

    return basic_patterns, standard_transforms, transformer_pattern, transformer_transforms, deep_mlp_pattern


# ============================================================================
# 3. BENCHMARK RUNNER CLASS
# ============================================================================
class BenchmarkRunner:
    def __init__(self, config_name: str, data_manager: DataManager, timeit_runs: int, timeit_repeats: int, frameworks: list, **kwargs):
        self.config_name = config_name
        self.dm = data_manager
        self.timeit_runs = timeit_runs
        self.timeit_repeats = timeit_repeats
        self.frameworks_to_run = frameworks
        self.kwargs = kwargs
        self.results = []
        print(f"\n\n{bcolors.HEADER}{'=' * 80}\n {self.config_name.center(78)} \n {str(self.kwargs).center(78)} \n{'=' * 80}{bcolors.ENDC}")

    def run(self, patterns: list[Pattern], transforms: list[Transformation]):
        torch._dynamo.reset()
        for pattern_obj in patterns:
            print(f"\n{bcolors.OKBLUE}{'-' * 20} PATTERN: {pattern_obj.name} {'-' * 20}{bcolors.ENDC}")
            for trans in transforms:
                print(f"  {bcolors.OKCYAN}--> TRANSFORM: {trans.name}{bcolors.ENDC}")
                self._run_single_test(pattern_obj, trans, self.frameworks_to_run)
        return pd.DataFrame(self.results)

    def _run_single_test(self, pattern_obj: Pattern, trans: Transformation, frameworks_to_run: list[str]):
        actual_arg_dtypes = pattern_obj.arg_dtypes if pattern_obj.arg_dtypes is not None else ["float32"] * len(pattern_obj.arg_shapes)
        
        args_nabla = None
        if "Nabla" in frameworks_to_run and nb is not None and pattern_obj.nabla_func is not None:
            try:
                args_nabla = tuple(self.dm.get_tensor(s, "nabla", dt) for s, dt in zip(pattern_obj.arg_shapes, actual_arg_dtypes, strict=True))
            except Exception as e:
                print(f"      {bcolors.WARNING}Nabla arg prep failed for {pattern_obj.name}: {e}{bcolors.ENDC}")
                args_nabla = "Error" # Mark as error

        args_jax = None
        if "JAX" in frameworks_to_run and pattern_obj.jax_func is not None:
             args_jax = tuple(self.dm.get_tensor(s, "jax", dt) for s, dt in zip(pattern_obj.arg_shapes, actual_arg_dtypes, strict=True))

        args_torch = None
        if "PyTorch" in frameworks_to_run and pattern_obj.torch_func is not None:
            args_torch = tuple(self.dm.get_tensor(s, "torch", dt) for s, dt in zip(pattern_obj.arg_shapes, actual_arg_dtypes, strict=True))


        framework_funcs = {
            "Nabla": (trans.nabla_transform(pattern_obj.nabla_func), args_nabla) if nb and pattern_obj.nabla_func else (None, None),
            "JAX": (trans.jax_transform(pattern_obj.jax_func), args_jax) if pattern_obj.jax_func else (None, None),
            "PyTorch": (trans.torch_transform(pattern_obj.torch_func), args_torch) if pattern_obj.torch_func else (None,None),
        }
        for fw_name in frameworks_to_run:
            func_tuple = framework_funcs.get(fw_name)
            if func_tuple and func_tuple[0] is not None and func_tuple[1] != "Error":
                self._measure_and_store(fw_name, *func_tuple, pattern_obj.name, trans.name)
            elif func_tuple and func_tuple[1] == "Error":
                 self.results.append({
                    "Benchmark": self.config_name, "Pattern": pattern_obj.name, "Transform": trans.name,
                    "Framework": fw_name, "Time (ms)": np.nan, "First Run Time (ms)": np.nan,
                    "Error": "Argument Preparation Failed"
                })
            else: # Framework not requested, or function not available (e.g. Nabla not installed)
                 if fw_name == "Nabla" and nb is None:
                    print(f"      {bcolors.WARNING}Nabla not available, skipping {pattern_obj.name} - {trans.name}{bcolors.ENDC}")
                 self.results.append({
                    "Benchmark": self.config_name, "Pattern": pattern_obj.name, "Transform": trans.name,
                    "Framework": fw_name, "Time (ms)": np.nan, "First Run Time (ms)": np.nan,
                    "Error": "Skipped (FW N/A or Func N/A)"
                })


    def _measure_and_store(self, fw_name, func_to_benchmark, args_tuple, pattern_name_str, trans_name_str):
        try:
            local_args_for_run = list(args_tuple)

            if fw_name == "PyTorch" and ("Grad" in trans_name_str):
                # For Deep MLP, grad is w.r.t. x (args[0])
                if "Deep MLP" in pattern_name_str: # Covers "Deep MLP (X Layers)"
                    if isinstance(local_args_for_run[0], torch.Tensor) and local_args_for_run[0].is_floating_point():
                        local_args_for_run[0] = local_args_for_run[0].clone().detach().requires_grad_(True)
                elif pattern_name_str != "Transformer Bwd Pass": # Standard cases
                    if len(local_args_for_run) > 0 and isinstance(local_args_for_run[0], torch.Tensor) and local_args_for_run[0].is_floating_point():
                        local_args_for_run[0] = local_args_for_run[0].clone().detach().requires_grad_(True)
            
            final_args_for_run = tuple(local_args_for_run)

            if "Deep MLP" in pattern_name_str:
                def func_base_call():
                    return func_to_benchmark(final_args_for_run[0], final_args_for_run[1], final_args_for_run[2])
            elif pattern_name_str == "Transformer Bwd Pass":
                 def func_base_call():
                    return func_to_benchmark(*final_args_for_run)
            else:
                def func_base_call():
                    return func_to_benchmark(*final_args_for_run)

            if fw_name == "JAX":
                def func_call_wrapper_for_timing():
                    res = func_base_call()
                    jax.block_until_ready(jax.tree_util.tree_leaves(res))
                    return res
            elif fw_name == "PyTorch":
                def func_call_wrapper_for_timing():
                    res = func_base_call()
                    torch.cpu.synchronize()
                    return res
            else: # Nabla (or other assumed synchronous)
                func_call_wrapper_for_timing = func_base_call
            
            # First run
            start_time_first_run = time.perf_counter()
            result_first_run = func_call_wrapper_for_timing()
            first_run_ms = (time.perf_counter() - start_time_first_run) * 1000

            # Steady state
            timer = timeit.Timer(stmt="func_call_wrapper_for_timing()", 
                                 globals={"func_call_wrapper_for_timing": func_call_wrapper_for_timing})
            times = timer.repeat(repeat=self.timeit_repeats, number=self.timeit_runs)
            steady_state_ms = (min(times) / self.timeit_runs) * 1000

            print(f"      {fw_name:<8} First Run: {first_run_ms:8.3f} ms | Steady State: {steady_state_ms:8.3f} ms")
            self.results.append({
                "Benchmark": self.config_name, "Pattern": pattern_name_str, "Transform": trans_name_str,
                "Framework": fw_name, "Time (ms)": steady_state_ms, "First Run Time (ms)": first_run_ms
            })
        except Exception as e:
            # import traceback # Uncomment for debugging
            # traceback.print_exc()
            print(f"      {bcolors.FAIL}{fw_name:<8} FAILED ({type(e).__name__}): {str(e)[:150]}{bcolors.ENDC}")
            self.results.append({
                "Benchmark": self.config_name, "Pattern": pattern_name_str, "Transform": trans_name_str,
                "Framework": fw_name, "Time (ms)": np.nan, "First Run Time (ms)": np.nan,
                "Error": str(e)[:100]
            })


# ============================================================================
# 4. RESULTS ANALYSIS CLASS
# ============================================================================
class ResultAnalyzer:
    def __init__(self, df: pd.DataFrame, frameworks: list):
        self.df = df.copy()
        self.frameworks = frameworks
        if "First Run Time (ms)" not in self.df.columns and not self.df.empty:
             self.df["First Run Time (ms)"] = np.nan

    def _build_analysis_table(self, pivot_df, time_metrics, speedup_metric_name, base_time_col_name="Time (ms)"):
        pivot_df = pivot_df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if pivot_df.empty:
            return pd.DataFrame()

        analysis_data = []
        for idx in pivot_df.index:
            row_dict = {"Benchmark": idx[0], "Pattern": idx[1]}
            metric_times = {}
            has_data_for_row = False

            for metric_transform_name in time_metrics:
                fw_times_for_transform = {}
                for fw in self.frameworks:
                    try:
                        time_val = pivot_df.loc[idx, (fw, metric_transform_name)]
                        fw_times_for_transform[fw] = time_val
                        if pd.notna(time_val): has_data_for_row = True
                    except KeyError:
                        fw_times_for_transform[fw] = np.nan
                metric_times[metric_transform_name] = fw_times_for_transform
            
            if not has_data_for_row: continue

            base_s = pd.Series(metric_times[time_metrics[0]])
            opt_s = pd.Series(metric_times[time_metrics[1]]) # Series of optimized times (JIT or JIT(Grad))
            
            opt_times_series_for_best = opt_s.dropna()
            best_fw = opt_times_series_for_best.idxmin() if not opt_times_series_for_best.empty else "N/A"
            
            speedups = pd.Series(index=base_s.index, dtype=float)
            for fw_ in base_s.index:
                if pd.notna(base_s.get(fw_)) and pd.notna(opt_s.get(fw_)) and opt_s.get(fw_, np.nan) != 0:
                    speedups[fw_] = base_s.get(fw_) / opt_s.get(fw_)
                else:
                    speedups[fw_] = np.nan

            # Row for the "optimized" metric (e.g., JIT times)
            row_for_optimized_metric = {
                **row_dict,
                "Metric": f"{time_metrics[1]} ({base_time_col_name})",
                **metric_times[time_metrics[1]], # Adds Nabla, JAX, PyTorch times
                "Best Framework": best_fw
            }

            # Calculate and add Nabla comparison ratios
            nabla_opt_time = opt_s.get("Nabla", np.nan)
            jax_opt_time = opt_s.get("JAX", np.nan)
            pytorch_opt_time = opt_s.get("PyTorch", np.nan)

            if pd.notna(nabla_opt_time) and nabla_opt_time > 1e-9: # Avoid division by zero or tiny numbers
                if "JAX" in self.frameworks and pd.notna(jax_opt_time):
                    row_for_optimized_metric["Nabla vs JAX (Ratio)"] = jax_opt_time / nabla_opt_time
                else:
                    row_for_optimized_metric["Nabla vs JAX (Ratio)"] = np.nan
                
                if "PyTorch" in self.frameworks and pd.notna(pytorch_opt_time):
                    row_for_optimized_metric["Nabla vs PyTorch (Ratio)"] = pytorch_opt_time / nabla_opt_time
                else:
                    row_for_optimized_metric["Nabla vs PyTorch (Ratio)"] = np.nan
            else:
                row_for_optimized_metric["Nabla vs JAX (Ratio)"] = np.nan
                row_for_optimized_metric["Nabla vs PyTorch (Ratio)"] = np.nan

            analysis_data.append({**row_dict, "Metric": f"{time_metrics[0]} ({base_time_col_name})", **metric_times[time_metrics[0]]})
            analysis_data.append(row_for_optimized_metric)
            analysis_data.append({**row_dict, "Metric": speedup_metric_name, **speedups.to_dict()})

        if not analysis_data: return pd.DataFrame()
        final_df = pd.DataFrame(analysis_data).set_index(["Benchmark", "Pattern", "Metric"])
        return final_df

    def display_table(self, analysis_df, title, subtitle):
        if analysis_df.empty:
            print(f"\n{bcolors.WARNING}No data to display for '{title}' analysis.{bcolors.ENDC}")
            return

        print(f"\n\n{bcolors.HEADER}{'=' * 140}\n{title.center(140)}\n{'=' * 140}{bcolors.ENDC}") # Increased width
        print(f"\n{subtitle}")

        for fw in self.frameworks:
            if fw not in analysis_df.columns: analysis_df[fw] = np.nan
        
        # Define column order, including new ratio columns
        ordered_cols = self.frameworks + ["Best Framework", "Nabla vs JAX (Ratio)", "Nabla vs PyTorch (Ratio)"]
        analysis_df = analysis_df.reindex(columns=[c for c in ordered_cols if c in analysis_df.columns])

        # Fill NaNs for display
        for col in ["Best Framework", "Nabla vs JAX (Ratio)", "Nabla vs PyTorch (Ratio)"]:
            if col in analysis_df.columns:
                analysis_df[col] = analysis_df[col].fillna(np.nan if "Ratio" in col else "")


        styled_df = analysis_df.astype(object) 
        for col in analysis_df.columns:
            if col == "Best Framework": # Already handled by fillna("")
                styled_df[col] = analysis_df[col].fillna("")
                continue
            for idx in analysis_df.index:
                val = analysis_df.loc[idx, col]
                metric_name_tuple_part = idx[2] if isinstance(idx, tuple) and len(idx) > 2 else ""

                if pd.isna(val):
                    styled_df.loc[idx, col] = "-"
                elif "Speedup" in metric_name_tuple_part or "Ratio" in col:
                    styled_df.loc[idx, col] = f"{val:.2f}x"
                else: # Time
                    styled_df.loc[idx, col] = f"{val:.3f}"
        
        with pd.option_context("display.max_rows", None, "display.width", 220, "display.float_format", '{:,.3f}'.format): # Increased width
            print(styled_df.to_string())

    def process_and_display_analysis(self, time_column_name, analysis_title_suffix):
        if self.df.empty or time_column_name not in self.df.columns:
            print(f"{bcolors.WARNING}No '{time_column_name}' data to analyze for {analysis_title_suffix}.{bcolors.ENDC}")
            return

        df_subset = self.df.dropna(subset=[time_column_name])
        if df_subset.empty:
            print(f"{bcolors.WARNING}No valid '{time_column_name}' data points to analyze for {analysis_title_suffix}.{bcolors.ENDC}")
            return

        pivot_df = df_subset.pivot_table(
            index=["Benchmark", "Pattern"],
            columns=["Framework", "Transform"],
            values=time_column_name,
        )
        
        if not pivot_df.empty:
            current_frameworks_in_pivot = pivot_df.columns.get_level_values("Framework").unique()
            frameworks_to_reindex = [fw for fw in self.frameworks if fw in current_frameworks_in_pivot]
            if frameworks_to_reindex:
                pivot_df = pivot_df.reindex(frameworks_to_reindex, axis=1, level="Framework")

        all_transforms = df_subset["Transform"].unique() if "Transform" in df_subset else []
        
        if "Eager" in all_transforms and "JIT" in all_transforms:
            cols_to_select = []
            for fw in self.frameworks:
                if (fw, "Eager") in pivot_df.columns: cols_to_select.append((fw, "Eager"))
                if (fw, "JIT") in pivot_df.columns: cols_to_select.append((fw, "JIT"))
            
            if cols_to_select:
                # Filter pivot_df to only include patterns that have JIT results for at least one framework.
                # This avoids building analysis tables for patterns where JIT might have failed everywhere.
                valid_indices = pivot_df[[(fw, "JIT") for fw in self.frameworks if (fw, "JIT") in pivot_df.columns]].dropna(how='all').index
                if not valid_indices.empty:
                    fwd_pivot_filtered = pivot_df.loc[valid_indices, pd.MultiIndex.from_tuples(cols_to_select)]
                    fwd_analysis = self._build_analysis_table(fwd_pivot_filtered, ["Eager", "JIT"], "JIT Speedup (x)", base_time_col_name=time_column_name)
                    self.display_table(fwd_analysis, f"FORWARD PASS PERFORMANCE ANALYSIS ({analysis_title_suffix})", f"Compares Eager vs. JIT ({time_column_name}). 'Best Framework' is for JIT. Ratios are OtherTime/NablaTime.")


        if "Grad" in all_transforms and "JIT(Grad)" in all_transforms:
            cols_to_select_grad = []
            for fw in self.frameworks:
                if (fw, "Grad") in pivot_df.columns: cols_to_select_grad.append((fw, "Grad"))
                if (fw, "JIT(Grad)") in pivot_df.columns: cols_to_select_grad.append((fw, "JIT(Grad)"))
            
            if cols_to_select_grad:
                valid_indices_grad = pivot_df[[(fw, "JIT(Grad)") for fw in self.frameworks if (fw, "JIT(Grad)") in pivot_df.columns]].dropna(how='all').index
                if not valid_indices_grad.empty:
                    grad_pivot_filtered = pivot_df.loc[valid_indices_grad, pd.MultiIndex.from_tuples(cols_to_select_grad)]
                    grad_analysis = self._build_analysis_table(grad_pivot_filtered, ["Grad", "JIT(Grad)"], "JIT(Grad) Speedup (x)", base_time_col_name=time_column_name)
                    self.display_table(grad_analysis, f"GRADIENT PERFORMANCE ANALYSIS ({analysis_title_suffix})", f"Compares Eager Grad vs. JIT'd Grad ({time_column_name}). 'Best Framework' is for JIT(Grad). Ratios are OtherTime/NablaTime.")

    def process_and_display(self):
        if self.df.empty:
            print(f"{bcolors.WARNING}No results to analyze.{bcolors.ENDC}")
            return
        
        if 'Transform' not in self.df.columns:
            print(f"{bcolors.WARNING}Skipping detailed analysis as 'Transform' column is missing.{bcolors.ENDC}")
            with pd.option_context("display.max_rows", None, "display.width", 200): print(self.df.to_string())
            return
        
        self.process_and_display_analysis("Time (ms)", "Steady State")
        
        if "First Run Time (ms)" in self.df.columns:
            self.process_and_display_analysis("First Run Time (ms)", "First Run (incl. JIT Comp.)")
        else:
            print(f"{bcolors.WARNING}'First Run Time (ms)' column not found, skipping its analysis.{bcolors.ENDC}")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def print_environment_info():
    print(f"{bcolors.BOLD}Environment Information:{bcolors.ENDC}")
    print(f"  - Python Version: {sys.version.split(' ')[0]}")
    print(f"  - Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    try:
        import cpuinfo
        print(f"  - CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    except ImportError:
        print("  - CPU: (Install 'py-cpuinfo' for details)")

    print(f"  - Library Versions:\n    - Nabla:   {NABLA_VERSION}\n    - JAX:     {jax.__version__}\n    - PyTorch: {torch.__version__}\n    - NumPy:   {np.__version__}\n    - Pandas:  {pd.__version__}")


def main():
    parser = argparse.ArgumentParser(description="Nabla/JAX/PyTorch Benchmarking Suite", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmarks", nargs="+", default=["micro", "macro", "transformer"], choices=["micro", "macro", "transformer"], help="Which benchmarks to run.")
    parser.add_argument("--frameworks", nargs="+", default=["Nabla", "JAX", "PyTorch"], choices=["Nabla", "JAX", "PyTorch"], help="Which frameworks to test.")
    parser.add_argument("--runs", type=int, default=TIMEIT_RUNS_DEFAULT, help="Number of runs per timing loop.")
    parser.add_argument("--repeats", type=int, default=TIMEIT_REPEATS_DEFAULT, help="Number of repeat timing loops.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save the raw timing results CSV file.")
    args = parser.parse_args()

    # Filter frameworks if Nabla is not available
    if nb is None and "Nabla" in args.frameworks:
        print(f"{bcolors.WARNING}Nabla framework selected but not found. Removing Nabla from tests.{bcolors.ENDC}")
        args.frameworks = [fw for fw in args.frameworks if fw != "Nabla"]
        if not args.frameworks:
            print(f"{bcolors.FAIL}No frameworks left to test after removing Nabla. Exiting.{bcolors.ENDC}")
            return

    print_environment_info()
    dm = DataManager(SEED)
    all_results_df = pd.DataFrame()

    print(f"\n{bcolors.BOLD}Running benchmarks: {', '.join(args.benchmarks)}{bcolors.ENDC}")
    print(f"\n{bcolors.BOLD}Testing frameworks: {', '.join(args.frameworks)}{bcolors.ENDC}")
    print(f"\n{bcolors.BOLD}Timing config: {args.repeats} repeats, {args.runs} runs each.{bcolors.ENDC}")

    common_kwargs = {"data_manager": dm, "timeit_runs": args.runs, "timeit_repeats": args.repeats, "frameworks": args.frameworks}

    if "micro" in args.benchmarks:
        # Micro benchmark dimensions (N_LAYERS_MLP will default to 8 here as not specified)
        dims_micro = {"N": 4, "D": 8, "B": 0, "S": 0, "E": 0}
        basic_ps_micro, std_ts_micro, _, _, mlp_p_micro = get_patterns_and_transforms(dims_micro)
        # For micro, maybe only basic patterns, not the deep MLP which is N_LAYERS_MLP=8 by default
        # If we want a "micro" deep MLP, it would use the default 8 layers.
        # Let's stick to simple patterns for micro.
        all_results_df = pd.concat([all_results_df, BenchmarkRunner("MICRO-BENCH", **common_kwargs, **dims_micro).run(basic_ps_micro, std_ts_micro)])

    if "macro" in args.benchmarks:
        dims_macro = {"N": 512, "D": 1024, "B": 0, "S": 0, "E": 0, "N_LAYERS_MLP": 16} # Scaled + 16 layers
        basic_ps_macro, std_ts_macro, _, _, mlp_p_macro = get_patterns_and_transforms(dims_macro)
        patterns_for_macro = basic_ps_macro + [mlp_p_macro]
        all_results_df = pd.concat([all_results_df, BenchmarkRunner("MACRO-BENCH", **common_kwargs, **dims_macro).run(patterns_for_macro, std_ts_macro)])

    if "transformer" in args.benchmarks:
        dims_transformer = {"N": 0, "D": 0, "B": 16, "S": 256, "E": 512} # Scaled
        # N_LAYERS_MLP is irrelevant for transformer pattern, so get_patterns_and_transforms will use default for MLP if it were used
        _, _, trans_p, trans_ts, _ = get_patterns_and_transforms(dims_transformer)
        all_results_df = pd.concat([all_results_df, BenchmarkRunner("TRANSFORMER", **common_kwargs, **dims_transformer).run([trans_p], trans_ts)])

    if args.output_csv and not all_results_df.empty:
        all_results_df.to_csv(args.output_csv, index=False)
        print(f"\n{bcolors.OKGREEN}Raw results saved to {args.output_csv}{bcolors.ENDC}")
    elif args.output_csv and all_results_df.empty:
        print(f"\n{bcolors.WARNING}No results generated to save to {args.output_csv}{bcolors.ENDC}")


    if not all_results_df.empty:
        analyzer = ResultAnalyzer(all_results_df, args.frameworks)
        analyzer.process_and_display()
    else:
        print(f"\n{bcolors.WARNING}No results were generated. Skipping analysis.{bcolors.ENDC}")


if __name__ == "__main__":
    main()




# Current output:
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

# Running benchmarks: micro, macro, transformer

# Testing frameworks: Nabla, JAX, PyTorch

# Timing config: 3 repeats, 10 runs each.


# ================================================================================
#                                   MICRO-BENCH                                   
#                     {'N': 4, 'D': 8, 'B': 0, 'S': 0, 'E': 0}                    
# ================================================================================

# -------------------- PATTERN: Element-wise --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.640 ms | Steady State:    0.050 ms
#       JAX      First Run:  107.452 ms | Steady State:    0.031 ms
#       PyTorch  First Run:    0.139 ms | Steady State:    0.006 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.758 ms | Steady State:    0.257 ms
#       JAX      First Run:  244.627 ms | Steady State:    0.981 ms
#       PyTorch  First Run:    8.742 ms | Steady State:    0.093 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  382.954 ms | Steady State:    0.098 ms
#       JAX      First Run:   34.626 ms | Steady State:    0.005 ms
#       PyTorch  First Run: 2625.361 ms | Steady State:    0.006 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  201.158 ms | Steady State:    0.029 ms
#       JAX      First Run:   34.391 ms | Steady State:    0.004 ms
#       PyTorch  First Run:  327.048 ms | Steady State:    0.041 ms

# -------------------- PATTERN: Matmul --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.380 ms | Steady State:    0.047 ms
#       JAX      First Run:   41.016 ms | Steady State:    0.010 ms
#       PyTorch  First Run:    1.290 ms | Steady State:    0.004 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.247 ms | Steady State:    0.083 ms
#       JAX      First Run:   67.241 ms | Steady State:    0.447 ms
#       PyTorch  First Run:    1.077 ms | Steady State:    0.042 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  188.645 ms | Steady State:    0.086 ms
#       JAX      First Run:   19.281 ms | Steady State:    0.005 ms
#       PyTorch  First Run:   22.218 ms | Steady State:    0.007 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  190.360 ms | Steady State:    0.065 ms
#       JAX      First Run:   23.168 ms | Steady State:    0.006 ms
#       PyTorch  First Run:   73.546 ms | Steady State:    0.010 ms

# -------------------- PATTERN: Reduction --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.261 ms | Steady State:    0.088 ms
#       JAX      First Run:   91.526 ms | Steady State:    0.037 ms
#       PyTorch  First Run:    2.761 ms | Steady State:    0.008 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.776 ms | Steady State:    0.329 ms
#       JAX      First Run:  196.079 ms | Steady State:    0.884 ms
#       PyTorch  First Run:    3.922 ms | Steady State:    0.114 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  215.449 ms | Steady State:    0.152 ms
#       JAX      First Run:   34.028 ms | Steady State:    0.006 ms
#       PyTorch  First Run:   26.542 ms | Steady State:    0.007 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  304.747 ms | Steady State:    0.072 ms
#       JAX      First Run:   26.856 ms | Steady State:    0.005 ms
#       PyTorch  First Run:  101.651 ms | Steady State:    0.017 ms

# -------------------- PATTERN: MLP Layer --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.148 ms | Steady State:    0.032 ms
#       JAX      First Run:   41.489 ms | Steady State:    0.058 ms
#       PyTorch  First Run:    1.419 ms | Steady State:    0.003 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    0.336 ms | Steady State:    0.160 ms
#       JAX      First Run:  118.913 ms | Steady State:    0.845 ms
#       PyTorch  First Run:    1.954 ms | Steady State:    0.056 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  264.292 ms | Steady State:    0.096 ms
#       JAX      First Run:   29.734 ms | Steady State:    0.008 ms
#       PyTorch  First Run:   26.808 ms | Steady State:    0.008 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  215.278 ms | Steady State:    0.090 ms
#       JAX      First Run:   30.757 ms | Steady State:    0.008 ms
#       PyTorch  First Run:  110.932 ms | Steady State:    0.020 ms


# ================================================================================
#                                   MACRO-BENCH                                   
#        {'N': 512, 'D': 1024, 'B': 0, 'S': 0, 'E': 0, 'N_LAYERS_MLP': 16}        
# ================================================================================

# -------------------- PATTERN: Element-wise --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    2.089 ms | Steady State:    1.687 ms
#       JAX      First Run:  127.935 ms | Steady State:    0.986 ms
#       PyTorch  First Run:    1.806 ms | Steady State:    0.742 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    6.536 ms | Steady State:    5.853 ms
#       JAX      First Run:  274.169 ms | Steady State:    3.567 ms
#       PyTorch  First Run:    3.278 ms | Steady State:    2.588 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run: 2620.217 ms | Steady State:    1.845 ms
#       JAX      First Run:   25.594 ms | Steady State:    0.734 ms
#       PyTorch  First Run: 1858.696 ms | Steady State:    0.520 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run: 2771.481 ms | Steady State:    0.715 ms
#       JAX      First Run:   25.944 ms | Steady State:    0.519 ms
#       PyTorch  First Run:  853.562 ms | Steady State:    0.743 ms

# -------------------- PATTERN: Matmul --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    0.788 ms | Steady State:    0.378 ms
#       JAX      First Run:   36.711 ms | Steady State:    1.348 ms
#       PyTorch  First Run:    0.598 ms | Steady State:    0.339 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    3.282 ms | Steady State:    1.831 ms
#       JAX      First Run:   78.846 ms | Steady State:    2.975 ms
#       PyTorch  First Run:    1.232 ms | Steady State:    1.167 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  172.628 ms | Steady State:    0.517 ms
#       JAX      First Run:   25.019 ms | Steady State:    1.255 ms
#       PyTorch  First Run:   22.383 ms | Steady State:    0.348 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  179.332 ms | Steady State:    0.427 ms
#       JAX      First Run:   27.449 ms | Steady State:    1.401 ms
#       PyTorch  First Run:   71.601 ms | Steady State:    0.483 ms

# -------------------- PATTERN: Reduction --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    1.132 ms | Steady State:    0.980 ms
#       JAX      First Run:  117.632 ms | Steady State:    0.303 ms
#       PyTorch  First Run:    0.341 ms | Steady State:    0.112 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    4.116 ms | Steady State:    3.605 ms
#       JAX      First Run:  245.264 ms | Steady State:    1.598 ms
#       PyTorch  First Run:    1.153 ms | Steady State:    0.549 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  206.968 ms | Steady State:    0.085 ms
#       JAX      First Run:   31.024 ms | Steady State:    0.126 ms
#       PyTorch  First Run:   24.705 ms | Steady State:    0.055 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  257.192 ms | Steady State:    0.170 ms
#       JAX      First Run:   38.958 ms | Steady State:    0.167 ms
#       PyTorch  First Run:  942.977 ms | Steady State:    0.186 ms

# -------------------- PATTERN: MLP Layer --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:    2.288 ms | Steady State:    1.106 ms
#       JAX      First Run:   51.114 ms | Steady State:    2.690 ms
#       PyTorch  First Run:    1.440 ms | Steady State:    0.932 ms
#   --> TRANSFORM: Grad
#       Nabla    First Run:    4.979 ms | Steady State:    3.959 ms
#       JAX      First Run:  122.972 ms | Steady State:    6.068 ms
#       PyTorch  First Run:    2.356 ms | Steady State:    1.973 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run:  175.585 ms | Steady State:    0.986 ms
#       JAX      First Run:   27.821 ms | Steady State:    2.523 ms
#       PyTorch  First Run:   28.536 ms | Steady State:    0.826 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run:  200.187 ms | Steady State:    1.531 ms
#       JAX      First Run:   25.089 ms | Steady State:    4.765 ms
#       PyTorch  First Run:  861.425 ms | Steady State:    2.239 ms

# -------------------- PATTERN: Deep MLP (16 Layers) --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:   21.863 ms | Steady State:   16.560 ms
#       JAX      First Run:   42.413 ms | Steady State:   39.436 ms
#       PyTorch  First Run:   18.195 ms | Steady State:   16.716 ms
#   --> TRANSFORM: Grad
# /Users/tillife/Documents/CodingProjects/nabla/nabla/ops/binary.py:148: RuntimeWarning: invalid value encountered in divide
#   np_result = np.divide(args[0].to_numpy(), args[1].to_numpy())
#       Nabla    First Run:   90.882 ms | Steady State:   74.109 ms
#       JAX      First Run:   93.523 ms | Steady State:   86.763 ms
#       PyTorch  First Run:   42.562 ms | Steady State:   36.075 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run: 2639.288 ms | Steady State:   13.437 ms
#       JAX      First Run:   89.128 ms | Steady State:   36.531 ms
#       PyTorch  First Run:  906.384 ms | Steady State:   15.076 ms
#   --> TRANSFORM: JIT(Grad)
#       Nabla    First Run: 2933.495 ms | Steady State:   29.642 ms
#       JAX      First Run:  154.670 ms | Steady State:   77.362 ms
#       PyTorch  First Run: 1618.582 ms | Steady State:   47.823 ms


# ================================================================================
#                                   TRANSFORMER                                   
#                  {'N': 0, 'D': 0, 'B': 16, 'S': 256, 'E': 512}                  
# ================================================================================

# -------------------- PATTERN: Transformer Bwd Pass --------------------
#   --> TRANSFORM: Eager
#       Nabla    First Run:  168.265 ms | Steady State:  175.018 ms
#       JAX      First Run:  912.142 ms | Steady State:  194.931 ms
#       PyTorch  First Run:   99.962 ms | Steady State:   89.712 ms
#   --> TRANSFORM: JIT
#       Nabla    First Run: 8299.188 ms | Steady State:   92.245 ms
#       JAX      First Run:  255.335 ms | Steady State:  179.384 ms
#       PyTorch  First Run: 3206.381 ms | Steady State:   93.003 ms


# ============================================================================================================================================
#                                               FORWARD PASS PERFORMANCE ANALYSIS (Steady State)                                              
# ============================================================================================================================================

# Compares Eager vs. JIT (Time (ms)). 'Best Framework' is for JIT. Ratios are OtherTime/NablaTime.
#                                                       Nabla      JAX PyTorch Best Framework Nabla vs JAX (Ratio) Nabla vs PyTorch (Ratio)
# Benchmark   Pattern              Metric                                                                                                  
# MACRO-BENCH Deep MLP (16 Layers) Eager (Time (ms))   16.560   39.436  16.716                                   -                        -
#                                  JIT (Time (ms))     13.437   36.531  15.076          Nabla                2.72x                    1.12x
#                                  JIT Speedup (x)      1.23x    1.08x   1.11x                                   -                        -
#             Element-wise         Eager (Time (ms))    1.687    0.986   0.742                                   -                        -
#                                  JIT (Time (ms))      1.845    0.734   0.520        PyTorch                0.40x                    0.28x
#                                  JIT Speedup (x)      0.91x    1.34x   1.43x                                   -                        -
#             MLP Layer            Eager (Time (ms))    1.106    2.690   0.932                                   -                        -
#                                  JIT (Time (ms))      0.986    2.523   0.826        PyTorch                2.56x                    0.84x
#                                  JIT Speedup (x)      1.12x    1.07x   1.13x                                   -                        -
#             Matmul               Eager (Time (ms))    0.378    1.348   0.339                                   -                        -
#                                  JIT (Time (ms))      0.517    1.255   0.348        PyTorch                2.43x                    0.67x
#                                  JIT Speedup (x)      0.73x    1.07x   0.97x                                   -                        -
#             Reduction            Eager (Time (ms))    0.980    0.303   0.112                                   -                        -
#                                  JIT (Time (ms))      0.085    0.126   0.055        PyTorch                1.48x                    0.64x
#                                  JIT Speedup (x)     11.50x    2.40x   2.04x                                   -                        -
# MICRO-BENCH Element-wise         Eager (Time (ms))    0.050    0.031   0.006                                   -                        -
#                                  JIT (Time (ms))      0.098    0.005   0.006            JAX                0.05x                    0.06x
#                                  JIT Speedup (x)      0.51x    6.68x   1.15x                                   -                        -
#             MLP Layer            Eager (Time (ms))    0.032    0.058   0.003                                   -                        -
#                                  JIT (Time (ms))      0.096    0.008   0.008            JAX                0.08x                    0.08x
#                                  JIT Speedup (x)      0.33x    7.67x   0.41x                                   -                        -
#             Matmul               Eager (Time (ms))    0.047    0.010   0.004                                   -                        -
#                                  JIT (Time (ms))      0.086    0.005   0.007            JAX                0.06x                    0.08x
#                                  JIT Speedup (x)      0.55x    2.02x   0.58x                                   -                        -
#             Reduction            Eager (Time (ms))    0.088    0.037   0.008                                   -                        -
#                                  JIT (Time (ms))      0.152    0.006   0.007            JAX                0.04x                    0.04x
#                                  JIT Speedup (x)      0.58x    5.99x   1.21x                                   -                        -
# TRANSFORMER Transformer Bwd Pass Eager (Time (ms))  175.018  194.931  89.712                                   -                        -
#                                  JIT (Time (ms))     92.245  179.384  93.003          Nabla                1.94x                    1.01x
#                                  JIT Speedup (x)      1.90x    1.09x   0.96x                                   -                        -


# ============================================================================================================================================
#                                                 GRADIENT PERFORMANCE ANALYSIS (Steady State)                                                
# ============================================================================================================================================

# Compares Eager Grad vs. JIT'd Grad (Time (ms)). 'Best Framework' is for JIT(Grad). Ratios are OtherTime/NablaTime.
#                                                          Nabla      JAX PyTorch Best Framework Nabla vs JAX (Ratio) Nabla vs PyTorch (Ratio)
# Benchmark   Pattern              Metric                                                                                                     
# MACRO-BENCH Deep MLP (16 Layers) Grad (Time (ms))       74.109   86.763  36.075                                   -                        -
#                                  JIT(Grad) (Time (ms))  29.642   77.362  47.823          Nabla                2.61x                    1.61x
#                                  JIT(Grad) Speedup (x)   2.50x    1.12x   0.75x                                   -                        -
#             Element-wise         Grad (Time (ms))        5.853    3.567   2.588                                   -                        -
#                                  JIT(Grad) (Time (ms))   0.715    0.519   0.743            JAX                0.73x                    1.04x
#                                  JIT(Grad) Speedup (x)   8.19x    6.87x   3.48x                                   -                        -
#             MLP Layer            Grad (Time (ms))        3.959    6.068   1.973                                   -                        -
#                                  JIT(Grad) (Time (ms))   1.531    4.765   2.239          Nabla                3.11x                    1.46x
#                                  JIT(Grad) Speedup (x)   2.59x    1.27x   0.88x                                   -                        -
#             Matmul               Grad (Time (ms))        1.831    2.975   1.167                                   -                        -
#                                  JIT(Grad) (Time (ms))   0.427    1.401   0.483          Nabla                3.28x                    1.13x
#                                  JIT(Grad) Speedup (x)   4.29x    2.12x   2.42x                                   -                        -
#             Reduction            Grad (Time (ms))        3.605    1.598   0.549                                   -                        -
#                                  JIT(Grad) (Time (ms))   0.170    0.167   0.186            JAX                0.98x                    1.10x
#                                  JIT(Grad) Speedup (x)  21.20x    9.54x   2.95x                                   -                        -
# MICRO-BENCH Element-wise         Grad (Time (ms))        0.257    0.981   0.093                                   -                        -
#                                  JIT(Grad) (Time (ms))   0.029    0.004   0.041            JAX                0.13x                    1.39x
#                                  JIT(Grad) Speedup (x)   8.77x  249.98x   2.28x                                   -                        -
#             MLP Layer            Grad (Time (ms))        0.160    0.845   0.056                                   -                        -
#                                  JIT(Grad) (Time (ms))   0.090    0.008   0.020            JAX                0.09x                    0.22x
#                                  JIT(Grad) Speedup (x)   1.77x  101.35x   2.79x                                   -                        -
#             Matmul               Grad (Time (ms))        0.083    0.447   0.042                                   -                        -
#                                  JIT(Grad) (Time (ms))   0.065    0.006   0.010            JAX                0.09x                    0.16x
#                                  JIT(Grad) Speedup (x)   1.26x   73.40x   4.02x                                   -                        -
#             Reduction            Grad (Time (ms))        0.329    0.884   0.114                                   -                        -
#                                  JIT(Grad) (Time (ms))   0.072    0.005   0.017            JAX                0.07x                    0.23x
#                                  JIT(Grad) Speedup (x)   4.55x  166.88x   6.91x                                   -                        -


# ============================================================================================================================================
#                                       FORWARD PASS PERFORMANCE ANALYSIS (First Run (incl. JIT Comp.))                                       
# ============================================================================================================================================

# Compares Eager vs. JIT (First Run Time (ms)). 'Best Framework' is for JIT. Ratios are OtherTime/NablaTime.
#                                                                  Nabla      JAX   PyTorch Best Framework Nabla vs JAX (Ratio) Nabla vs PyTorch (Ratio)
# Benchmark   Pattern              Metric                                                                                                               
# MACRO-BENCH Deep MLP (16 Layers) Eager (First Run Time (ms))    21.863   42.413    18.195                                   -                        -
#                                  JIT (First Run Time (ms))    2639.288   89.128   906.384            JAX                0.03x                    0.34x
#                                  JIT Speedup (x)                 0.01x    0.48x     0.02x                                   -                        -
#             Element-wise         Eager (First Run Time (ms))     2.089  127.935     1.806                                   -                        -
#                                  JIT (First Run Time (ms))    2620.217   25.594  1858.696            JAX                0.01x                    0.71x
#                                  JIT Speedup (x)                 0.00x    5.00x     0.00x                                   -                        -
#             MLP Layer            Eager (First Run Time (ms))     2.288   51.114     1.440                                   -                        -
#                                  JIT (First Run Time (ms))     175.585   27.821    28.536            JAX                0.16x                    0.16x
#                                  JIT Speedup (x)                 0.01x    1.84x     0.05x                                   -                        -
#             Matmul               Eager (First Run Time (ms))     0.788   36.711     0.598                                   -                        -
#                                  JIT (First Run Time (ms))     172.628   25.019    22.383        PyTorch                0.14x                    0.13x
#                                  JIT Speedup (x)                 0.00x    1.47x     0.03x                                   -                        -
#             Reduction            Eager (First Run Time (ms))     1.132  117.632     0.341                                   -                        -
#                                  JIT (First Run Time (ms))     206.968   31.024    24.705        PyTorch                0.15x                    0.12x
#                                  JIT Speedup (x)                 0.01x    3.79x     0.01x                                   -                        -
# MICRO-BENCH Element-wise         Eager (First Run Time (ms))     0.640  107.452     0.139                                   -                        -
#                                  JIT (First Run Time (ms))     382.954   34.626  2625.361            JAX                0.09x                    6.86x
#                                  JIT Speedup (x)                 0.00x    3.10x     0.00x                                   -                        -
#             MLP Layer            Eager (First Run Time (ms))     0.148   41.489     1.419                                   -                        -
#                                  JIT (First Run Time (ms))     264.292   29.734    26.808        PyTorch                0.11x                    0.10x
#                                  JIT Speedup (x)                 0.00x    1.40x     0.05x                                   -                        -
#             Matmul               Eager (First Run Time (ms))     0.380   41.016     1.290                                   -                        -
#                                  JIT (First Run Time (ms))     188.645   19.281    22.218            JAX                0.10x                    0.12x
#                                  JIT Speedup (x)                 0.00x    2.13x     0.06x                                   -                        -
#             Reduction            Eager (First Run Time (ms))     0.261   91.526     2.761                                   -                        -
#                                  JIT (First Run Time (ms))     215.449   34.028    26.542        PyTorch                0.16x                    0.12x
#                                  JIT Speedup (x)                 0.00x    2.69x     0.10x                                   -                        -
# TRANSFORMER Transformer Bwd Pass Eager (First Run Time (ms))   168.265  912.142    99.962                                   -                        -
#                                  JIT (First Run Time (ms))    8299.188  255.335  3206.381            JAX                0.03x                    0.39x
#                                  JIT Speedup (x)                 0.02x    3.57x     0.03x                                   -                        -


# ============================================================================================================================================
#                                         GRADIENT PERFORMANCE ANALYSIS (First Run (incl. JIT Comp.))                                         
# ============================================================================================================================================

# Compares Eager Grad vs. JIT'd Grad (First Run Time (ms)). 'Best Framework' is for JIT(Grad). Ratios are OtherTime/NablaTime.
#                                                                      Nabla      JAX   PyTorch Best Framework Nabla vs JAX (Ratio) Nabla vs PyTorch (Ratio)
# Benchmark   Pattern              Metric                                                                                                                   
# MACRO-BENCH Deep MLP (16 Layers) Grad (First Run Time (ms))         90.882   93.523    42.562                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))  2933.495  154.670  1618.582            JAX                0.05x                    0.55x
#                                  JIT(Grad) Speedup (x)               0.03x    0.60x     0.03x                                   -                        -
#             Element-wise         Grad (First Run Time (ms))          6.536  274.169     3.278                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))  2771.481   25.944   853.562            JAX                0.01x                    0.31x
#                                  JIT(Grad) Speedup (x)               0.00x   10.57x     0.00x                                   -                        -
#             MLP Layer            Grad (First Run Time (ms))          4.979  122.972     2.356                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))   200.187   25.089   861.425            JAX                0.13x                    4.30x
#                                  JIT(Grad) Speedup (x)               0.02x    4.90x     0.00x                                   -                        -
#             Matmul               Grad (First Run Time (ms))          3.282   78.846     1.232                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))   179.332   27.449    71.601            JAX                0.15x                    0.40x
#                                  JIT(Grad) Speedup (x)               0.02x    2.87x     0.02x                                   -                        -
#             Reduction            Grad (First Run Time (ms))          4.116  245.264     1.153                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))   257.192   38.958   942.977            JAX                0.15x                    3.67x
#                                  JIT(Grad) Speedup (x)               0.02x    6.30x     0.00x                                   -                        -
# MICRO-BENCH Element-wise         Grad (First Run Time (ms))          0.758  244.627     8.742                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))   201.158   34.391   327.048            JAX                0.17x                    1.63x
#                                  JIT(Grad) Speedup (x)               0.00x    7.11x     0.03x                                   -                        -
#             MLP Layer            Grad (First Run Time (ms))          0.336  118.913     1.954                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))   215.278   30.757   110.932            JAX                0.14x                    0.52x
#                                  JIT(Grad) Speedup (x)               0.00x    3.87x     0.02x                                   -                        -
#             Matmul               Grad (First Run Time (ms))          0.247   67.241     1.077                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))   190.360   23.168    73.546            JAX                0.12x                    0.39x
#                                  JIT(Grad) Speedup (x)               0.00x    2.90x     0.01x                                   -                        -
#             Reduction            Grad (First Run Time (ms))          0.776  196.079     3.922                                   -                        -
#                                  JIT(Grad) (First Run Time (ms))   304.747   26.856   101.651            JAX                0.09x                    0.33x
#                                  JIT(Grad) Speedup (x)               0.00x    7.30x     0.04x                                   -                        -