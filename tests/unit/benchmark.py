#!/usr/bin/env python
"""
====================================================================================
DEFINITIVE BENCHMARKING SUITE FOR NABLA, JAX, AND PYTORCH (CPU) - V11 (STABLE-FINAL)
====================================================================================
This script contains fixes for all previously identified framework and analysis bugs.

Key Features:
1.  STABLE IMPLEMENTATIONS: Corrects the `TypeError` in Nabla's grad call.
2.  CORRECT ANALYSIS: Implements a fully robust pandas analysis pipeline that correctly
    builds MultiIndex DataFrames and is resilient to failed benchmark runs.
3.  THESIS-READY OUTPUT: Generates a clear, multi-level pivot table.
"""
import time
import timeit
import argparse
import platform
import sys
from typing import Callable, NamedTuple, List
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

# --- Framework Imports ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import nabla as nb
import jax
import jax.numpy as jnp
import jax.nn as jax_nn
import torch
import torch.nn.functional as F
import torch.func
import torch._dynamo

# --- Global Configuration ---
SEED = 42
TIMEIT_RUNS_DEFAULT = 10
TIMEIT_REPEATS_DEFAULT = 3

class bcolors:
    HEADER, OKBLUE, OKCYAN, OKGREEN, WARNING, FAIL, ENDC, BOLD = '\033[95m', '\033[94m', '\033[96m', '\033[92m', '\033[93m', '\033[91m', '\033[0m', '\033[1m'

# ============================================================================
# 1. HELPER & DATA GENERATION CLASSES (Unchanged)
# ============================================================================
class FrameworkAPIs(NamedTuple):
    numpy: np; nabla: nb; jax: jnp; torch: torch

class DataManager:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.apis = FrameworkAPIs(np, nb, jnp, torch)

    def get_tensor(self, shape: tuple, framework: str):
        data = self.rng.random(shape, dtype=np.float32)
        if framework == 'nabla': return self.apis.nabla.array(data)
        if framework == 'jax': return self.apis.jax.array(data)
        if framework == 'torch': return self.apis.torch.from_numpy(data)
        return data

# ============================================================================
# 2. BENCHMARK DEFINITIONS (Nabla grad fix)
# ============================================================================
@dataclass
class Pattern:
    name: str; nabla_func: Callable; jax_func: Callable; torch_func: Callable; arg_shapes: list[tuple]

@dataclass
class Transformation:
    name: str; nabla_transform: Callable; jax_transform: Callable; torch_transform: Callable

def get_patterns_and_transforms(N, D, B, S, E):
    patterns = [
        Pattern(name="Element-wise", nabla_func=lambda x: nb.sum(nb.tanh(x)*nb.sin(x)/(nb.abs(x)+1e-6)), jax_func=lambda x: jnp.sum(jnp.tanh(x)*jnp.sin(x)/(jnp.abs(x)+1e-6)), torch_func=lambda x: torch.sum(torch.tanh(x)*torch.sin(x)/(torch.abs(x)+1e-6)), arg_shapes=[(N,D)]),
        Pattern(name="Matmul", nabla_func=lambda x,w: nb.sum(x@w), jax_func=lambda x,w: jnp.sum(x@w), torch_func=lambda x,w: torch.sum(x@w), arg_shapes=[(N,D),(D,N)]),
        Pattern(name="Reduction", nabla_func=lambda x: nb.sum(nb.sqrt(nb.mean((x-nb.mean(x,axes=1,keep_dims=True))**2,axes=1))), jax_func=lambda x: jnp.sum(jnp.sqrt(jnp.mean((x-jnp.mean(x,axis=1,keepdims=True))**2,axis=1))), torch_func=lambda x: torch.sum(torch.sqrt(torch.mean((x-torch.mean(x,dim=1,keepdim=True))**2,dim=1))), arg_shapes=[(N,D)]),
        Pattern(name="MLP Layer", nabla_func=lambda x,w,b: nb.sum(nb.relu(x@w+b)), jax_func=lambda x,w,b: jnp.sum(jax.nn.relu(x@w+b)), torch_func=lambda x,w,b: torch.sum(F.relu(x@w+b)), arg_shapes=[(N,D),(D,D),(D,)]),
    ]

    def _nabla_transformer_fwd(x,y,w_q,w_k,w_v,w_o,w_ff1,w_ff2):
        q,k,v=x@w_q,x@w_k,x@w_v; attn_out=nb.softmax(q@k.transpose((0,2,1))/(E**0.5),axis=-1)@v; x=x+attn_out@w_o; x=x+nb.relu(x@w_ff1)@w_ff2; stable_x = x - nb.max(x, axes=-1, keep_dims=True); return -nb.sum(y*(stable_x - nb.log(nb.sum(nb.exp(stable_x), axes=-1, keep_dims=True))))/B
    def _jax_transformer_fwd(x,y,w_q,w_k,w_v,w_o,w_ff1,w_ff2):
        q,k,v=x@w_q,x@w_k,x@w_v; attn_out=jax_nn.softmax((q@k.transpose(0,2,1))/(E**0.5),axis=-1)@v; x=x+attn_out@w_o; x=x+jax_nn.relu(x@w_ff1)@w_ff2; return -jnp.sum(y*jax_nn.log_softmax(x,axis=-1))/B
    def _torch_transformer_loss(params,x,y):
        w_q,w_k,w_v,w_o,w_ff1,w_ff2=params; q,k,v=x@w_q,x@w_k,x@w_v; attn_out=F.softmax((q@k.transpose(-2,-1))/(E**0.5),dim=-1)@v; x=x+attn_out@w_o; x=x+F.relu(x@w_ff1)@w_ff2; return -torch.sum(y*F.log_softmax(x,dim=-1))/B
    def _torch_transformer_grad_wrapper(x,y,w_q,w_k,w_v,w_o,w_ff1,w_ff2):
        params=[w_q,w_k,w_v,w_o,w_ff1,w_ff2]; return torch.func.grad(_torch_transformer_loss)(params,x,y)

    transformer_pattern = Pattern(name="Transformer Bwd Pass",
        # FIX: Pass a tuple to argnums, not a range iterator.
        nabla_func=nb.grad(_nabla_transformer_fwd, argnums=tuple(range(2,8))),
        jax_func=jax.grad(_jax_transformer_fwd, argnums=range(2,8)),
        torch_func=_torch_transformer_grad_wrapper,
        arg_shapes=[(B,S,E),(B,S,E),(E,E),(E,E),(E,E),(E,E),(E,4*E),(4*E,E)])

    standard_transforms = [
        Transformation(name="Eager", nabla_transform=lambda f:f, jax_transform=lambda f:f, torch_transform=lambda f:f),
        Transformation(name="Grad", nabla_transform=nb.grad, jax_transform=jax.grad, torch_transform=torch.func.grad),
        Transformation(name="JIT", nabla_transform=nb.jit, jax_transform=jax.jit, torch_transform=torch.compile),
        Transformation(name="JIT(Grad)", nabla_transform=lambda f:nb.jit(nb.grad(f)), jax_transform=lambda f:jax.jit(jax.grad(f)), torch_transform=lambda f:torch.compile(torch.func.grad(f))),
    ]
    transformer_transforms = [
        Transformation(name="Eager", nabla_transform=lambda f:f, jax_transform=lambda f:f, torch_transform=lambda f:f),
        Transformation(name="JIT", nabla_transform=nb.jit, jax_transform=jax.jit, torch_transform=torch.compile),
    ]

    return patterns, standard_transforms, transformer_pattern, transformer_transforms

# ============================================================================
# 3. BENCHMARK RUNNER CLASS (Unchanged)
# ============================================================================
class BenchmarkRunner:
    def __init__(self, config_name: str, data_manager: DataManager, timeit_runs: int, timeit_repeats: int, frameworks: list, **kwargs):
        self.config_name = config_name; self.dm = data_manager; self.timeit_runs = timeit_runs; self.timeit_repeats = timeit_repeats; self.frameworks_to_run = frameworks; self.kwargs = kwargs; self.results = []
        print(f"\n\n{bcolors.HEADER}{'='*80}\n {self.config_name.center(78)} \n {str(self.kwargs).center(78)} \n{'='*80}{bcolors.ENDC}")
    def run(self, patterns: List[Pattern], transforms: List[Transformation]):
        torch._dynamo.reset()
        for pattern in patterns:
            print(f"\n{bcolors.OKBLUE}{'-'*20} PATTERN: {pattern.name} {'-'*20}{bcolors.ENDC}")
            for trans in transforms:
                print(f"  {bcolors.OKCYAN}--> TRANSFORM: {trans.name}{bcolors.ENDC}")
                self._run_single_test(pattern, trans)
        return pd.DataFrame(self.results)
    def _run_single_test(self, pattern: Pattern, trans: Transformation):
        framework_funcs = {"Nabla": (trans.nabla_transform(pattern.nabla_func), tuple(self.dm.get_tensor(s, 'nabla') for s in pattern.arg_shapes)), "JAX": (trans.jax_transform(pattern.jax_func), tuple(self.dm.get_tensor(s, 'jax') for s in pattern.arg_shapes)), "PyTorch": (trans.torch_transform(pattern.torch_func), tuple(self.dm.get_tensor(s, 'torch') for s in pattern.arg_shapes))}
        for fw_name in self.frameworks_to_run:
            self._measure_and_store(fw_name, *framework_funcs[fw_name], pattern.name, trans.name)
    def _measure_and_store(self, fw_name, func, args, pattern_name, trans_name):
        try:
            def run_and_sync():
                res = func(*args)
                if fw_name == 'JAX': jax.block_until_ready(jax.tree_util.tree_leaves(res))
                elif fw_name == 'PyTorch': torch.cpu.synchronize()
            start_time = time.perf_counter(); run_and_sync(); first_run_ms = (time.perf_counter()-start_time)*1000
            timer = timeit.Timer(stmt="run_and_sync()", globals=locals()); times = timer.repeat(repeat=self.timeit_repeats, number=self.timeit_runs); steady_state_ms = (min(times)/self.timeit_runs)*1000
            print(f"      {fw_name:<8} First Run: {first_run_ms:8.3f} ms | Steady State: {steady_state_ms:8.3f} ms")
            self.results.append({"Benchmark": self.config_name, "Pattern": pattern_name, "Transform": trans_name, "Framework": fw_name, "Time (ms)": steady_state_ms})
        except Exception as e:
            print(f"      {bcolors.FAIL}{fw_name:<8} FAILED ({type(e).__name__}){bcolors.ENDC}"); self.results.append({"Benchmark": self.config_name, "Pattern": pattern_name, "Transform": trans_name, "Framework": fw_name, "Time (ms)": np.nan})

# ============================================================================
# 4. RESULTS ANALYSIS CLASS (Fully corrected pandas logic)
# ============================================================================
class ResultAnalyzer:
    def __init__(self, df: pd.DataFrame, frameworks: list):
        self.df = df.copy()
        self.frameworks = frameworks

    def process_and_display(self):
        if self.df.empty:
            print(f"{bcolors.WARNING}No results to analyze.{bcolors.ENDC}")
            return

        pivot_df = self.df.pivot_table(
            index=['Benchmark', 'Pattern'],
            columns=['Framework', 'Transform'],
            values='Time (ms)'
        )

        all_metrics_dfs = []
        jit_map = {'JIT': 'Eager', 'JIT(Grad)': 'Grad'}
        frameworks_in_data = pivot_df.columns.get_level_values('Framework').unique()

        # --- Calculate and collect metrics with robust DataFrame construction ---
        for fw in frameworks_in_data:
            fw_times = pivot_df[fw]
            # Time (ms)
            df_to_add = pd.concat([fw_times], keys=[(fw, 'Time (ms)')], axis=1)
            all_metrics_dfs.append(df_to_add)

            # JIT Speedup
            speedups = pd.DataFrame(index=fw_times.index)
            for jit_trans, eager_trans in jit_map.items():
                if jit_trans in fw_times.columns and eager_trans in fw_times.columns:
                    speedups[jit_trans] = fw_times[eager_trans] / fw_times[jit_trans]
            if not speedups.empty and not speedups.isnull().all().all():
                df_to_add = pd.concat([speedups], keys=[(fw, 'JIT Speedup')], axis=1)
                all_metrics_dfs.append(df_to_add)

        # Relative Speedup vs other frameworks
        if 'Nabla' in frameworks_in_data:
            for baseline in ['JAX', 'PyTorch']:
                if baseline in frameworks_in_data:
                    nabla_times = pivot_df['Nabla']
                    baseline_times = pivot_df[baseline]
                    common_transforms = nabla_times.columns.intersection(baseline_times.columns)
                    relative_speedup = baseline_times[common_transforms] / nabla_times[common_transforms]
                    if not relative_speedup.empty:
                        df_to_add = pd.concat([relative_speedup], keys=[('Nabla', f'vs {baseline}')], axis=1)
                        all_metrics_dfs.append(df_to_add)

        # --- Combine, Restructure, and Display ---
        if not all_metrics_dfs:
            print(f"{bcolors.WARNING}Could not compute any analysis metrics.{bcolors.ENDC}")
            return
            
        final_df = pd.concat(all_metrics_dfs, axis=1)
        final_df.columns.names = ['Framework', 'Metric', 'Transform']
        final_df = final_df.stack(level='Transform', future_stack=True).sort_index(axis=1)

        print(f"\n\n{bcolors.HEADER}{'='*150}\n{'FINAL BENCHMARK ANALYSIS'.center(150)}\n{'='*150}{bcolors.ENDC}")
        print(f"\n--- Performance Analysis Table ---\nJIT Speedup = Eager Time / JIT Time. 'vs FW' = FW Time / Nabla Time. Higher is better for all ratio metrics.\n")
        with pd.option_context('display.max_rows', None, 'display.width', 200, 'display.precision', 3):
            formatters = {(fw, m): "{:.2f}x".format for fw, m in final_df.columns if 'Speedup' in m or 'vs' in m}
            print(final_df.to_string(float_format="%.3f", formatters=formatters, na_rep="-"))

# ============================================================================
# 5. MAIN EXECUTION (Unchanged)
# ============================================================================
def print_environment_info():
    print(f"{bcolors.BOLD}Environment Information:{bcolors.ENDC}")
    print(f"  - Python Version: {sys.version.split(' ')[0]}")
    print(f"  - Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    try: import cpuinfo; print(f"  - CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    except ImportError: print("  - CPU: (Install 'py-cpuinfo' for details)")
    print(f"  - Library Versions:\n    - Nabla:   0.1.0\n    - JAX:     {jax.__version__}\n    - PyTorch: {torch.__version__}\n    - NumPy:   {np.__version__}\n    - Pandas:  {pd.__version__}")

def main():
    parser = argparse.ArgumentParser(description="Nabla/JAX/PyTorch Benchmarking Suite", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmarks", nargs='+', default=['micro', 'macro', 'transformer'], choices=['micro', 'macro', 'transformer'], help="Which benchmarks to run.")
    parser.add_argument("--frameworks", nargs='+', default=['Nabla', 'JAX', 'PyTorch'], choices=['Nabla', 'JAX', 'PyTorch'], help="Which frameworks to test.")
    parser.add_argument("--runs", type=int, default=TIMEIT_RUNS_DEFAULT, help="Number of runs per timing loop.")
    parser.add_argument("--repeats", type=int, default=TIMEIT_REPEATS_DEFAULT, help="Number of repeat timing loops.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save the raw timing results CSV file.")
    args = parser.parse_args()
    
    print_environment_info()
    dm = DataManager(SEED)
    all_results_df = pd.DataFrame()
    
    print(f"\n{bcolors.BOLD}Running benchmarks: {', '.join(args.benchmarks)}{bcolors.ENDC}")
    print(f"\n{bcolors.BOLD}Testing frameworks: {', '.join(args.frameworks)}{bcolors.ENDC}")
    print(f"\n{bcolors.BOLD}Timing config: {args.repeats} repeats, {args.runs} runs each.{bcolors.ENDC}")

    common_kwargs = {'timeit_runs': args.runs, 'timeit_repeats': args.repeats, 'frameworks': args.frameworks}

    if 'micro' in args.benchmarks:
        p,t,_,_ = get_patterns_and_transforms(N=4,D=8,B=0,S=0,E=0)
        all_results_df = pd.concat([all_results_df, BenchmarkRunner("MICRO-BENCHMARKS", dm, **common_kwargs, N=4, D=8).run(p,t)])
    if 'macro' in args.benchmarks:
        p,t,_,_ = get_patterns_and_transforms(N=256,D=512,B=0,S=0,E=0)
        all_results_df = pd.concat([all_results_df, BenchmarkRunner("MACRO-BENCHMARKS", dm, **common_kwargs, N=256, D=512).run(p,t)])
    if 'transformer' in args.benchmarks:
        _,_,p_trans,t_trans = get_patterns_and_transforms(N=0,D=0,B=8,S=128,E=256)
        all_results_df = pd.concat([all_results_df, BenchmarkRunner("TRANSFORMER-BENCHMARK", dm, **common_kwargs, B=8, S=128, E=256).run([p_trans], t_trans)])

    if args.output_csv:
        all_results_df.to_csv(args.output_csv, index=False)
        print(f"\n{bcolors.OKGREEN}Raw results saved to {args.output_csv}{bcolors.ENDC}")

    analyzer = ResultAnalyzer(all_results_df, args.frameworks)
    analyzer.process_and_display()

if __name__ == "__main__":
    main()