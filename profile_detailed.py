"""Detailed profiling of Nabla hot paths."""

import numpy as np
import time

import nabla as nb
from nabla import ops
from nabla.core.autograd import value_and_grad

num_samples = 5
x_np = np.linspace(0, 1, num_samples).reshape(-1, 1).astype(np.float32)
y_np = (np.sin(4 * np.pi * x_np) + 1) / 2.0

layers = [1, 64, 64, 1]
np.random.seed(42)

init_params = []
for i in range(len(layers) - 1):
    in_dim = layers[i]
    out_dim = layers[i+1]
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    w_np = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
    b_np = np.zeros((1, out_dim)).astype(np.float32)
    init_params.append((w_np, b_np))

lr = 0.01

def mlp(x, params):
    for i in range(0, len(params) - 2, 2):
        w = params[i]
        b = params[i+1]
        x = ops.relu(ops.matmul(x, w) + b)
    x = ops.matmul(x, params[-2]) + params[-1]
    return x

def loss_fn(params, x, y):
    preds = mlp(x, params)
    diff = preds - y
    return ops.mean(diff * diff)

vg_fn = value_and_grad(loss_fn, argnums=0, realize=False)

x = nb.Tensor.from_dlpack(x_np)
y = nb.Tensor.from_dlpack(y_np)

params = []
for w, b in init_params:
    params.extend([nb.Tensor.from_dlpack(w), nb.Tensor.from_dlpack(b.reshape(1, -1))])
for p in params:
    p.is_traced = True

# Warmup
for _ in range(3):
    loss, grads = vg_fn(params, x, y)
    new_params = [p - g * lr for p, g in zip(params, grads)]
    nb.realize_all(loss, *new_params)
    params = new_params

# Reset
params = []
for w, b in init_params:
    params.extend([nb.Tensor.from_dlpack(w), nb.Tensor.from_dlpack(b.reshape(1, -1))])
for p in params:
    p.is_traced = True

# === Instrument the hot paths ===
import functools

TIMINGS = {}

def timed(name):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            if name not in TIMINGS:
                TIMINGS[name] = {"count": 0, "total": 0.0}
            TIMINGS[name]["count"] += 1
            TIMINGS[name]["total"] += elapsed
            return result
        return wrapper
    return decorator

# Patch key functions
from nabla.ops import base
from nabla.core.common import pytree
from nabla.core.sharding import spmd
from nabla.core.autograd import utils as autograd_utils

original_op_call = base.Operation.__call__
original_tree_flatten = pytree.tree_flatten
original_tree_leaves = pytree.tree_leaves
original_tree_map = pytree.tree_map
original_execute_on_shards = spmd.execute_on_shards
original_get_tensor_hash = None  # Will extract from __call__

# Detailed breakdown of Operation.__call__
call_timings = {
    "collect_metadata": [],
    "get_mesh": [],
    "adapt_kwargs": [],
    "ensure_specs": [],
    "infer_output_sharding": [],
    "reshard_inputs": [],
    "get_tensor_hash": [],
    "compute_physical_shape": [],
    "create_sharded_output": [],
    "setup_output_refs": [],
    "total": [],
}

def instrumented_op_call(self, *args, **kwargs):
    t_total_start = time.perf_counter()
    
    # === New Path: Physical Execution ===
    if hasattr(self, "execute"):
        from nabla.core import GRAPH, Tensor, pytree as pt
        from nabla.core.sharding import spmd as sp

        # 1. Collect Metadata (optimized: inline iteration instead of tree_map)
        t0 = time.perf_counter()
        max_batch_dims = 0
        any_traced = False
        any_sharded = False
        any_has_tangent = False
        
        # Fast path: use stack-based iteration
        stack = list(args)
        while stack:
            x = stack.pop()
            if isinstance(x, Tensor):
                if x.batch_dims > max_batch_dims:
                    max_batch_dims = x.batch_dims
                if x.is_traced:
                    any_traced = True
                if x.is_sharded:
                    any_sharded = True
                if x.tangent is not None:
                    any_has_tangent = True
            elif isinstance(x, (list, tuple)):
                stack.extend(x)
            elif isinstance(x, dict):
                stack.extend(x.values())
        call_timings["collect_metadata"].append(time.perf_counter() - t0)

        # 2. Get mesh
        t0 = time.perf_counter()
        mesh = sp.get_mesh_from_args(args) if any_sharded else None
        if mesh is None and kwargs.get("mesh") is not None:
            mesh = kwargs["mesh"]
        call_timings["get_mesh"].append(time.perf_counter() - t0)

        # 3. Adapt kwargs
        t0 = time.perf_counter()
        adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)
        call_timings["adapt_kwargs"].append(time.perf_counter() - t0)
        
        # 4. Ensure specs
        t0 = time.perf_counter()
        args = sp.ensure_specs(args, mesh)
        call_timings["ensure_specs"].append(time.perf_counter() - t0)
        
        # 5. Infer output sharding
        t0 = time.perf_counter()
        predicted_output_spec, input_shardings, reduce_axes = (
            sp.infer_output_sharding(self, args, mesh, adapted_kwargs or {})
        )
        call_timings["infer_output_sharding"].append(time.perf_counter() - t0)

        # 6. Reshard inputs
        t0 = time.perf_counter()
        resharded_args = sp.reshard_inputs(args, input_shardings, mesh)
        call_timings["reshard_inputs"].append(time.perf_counter() - t0)

        # 7. Get tensor hash
        t0 = time.perf_counter()
        def get_tensor_hash(x):
            if isinstance(x, Tensor):
                buffers = x._impl._buffers
                has_output_refs = x._impl.output_refs is not None
                sharding_key = base._make_hashable(x.sharding) if x.sharding else None
                if buffers:
                    shape_tuple = tuple(int(d) for d in x.shape)
                    return ("realized", str(x.dtype), shape_tuple, sharding_key)
                elif has_output_refs and x._impl.output_refs._op_hash is not None:
                    return (x._impl.output_refs._op_hash, x._impl.output_index, sharding_key)
                else:
                    shape_tuple = tuple(int(d) for d in x.shape)
                    return ("leaf", str(x.dtype), shape_tuple, sharding_key)
            return base._make_hashable(x)
        
        arg_hashes = tuple(get_tensor_hash(x) for x in resharded_args)
        kwarg_hashes = tuple(sorted((k, get_tensor_hash(v)) for k, v in (adapted_kwargs or kwargs).items()))
        op_hash = (self.name, arg_hashes, kwarg_hashes)
        call_timings["get_tensor_hash"].append(time.perf_counter() - t0)

        # 8. Compute physical shape
        t0 = time.perf_counter()
        output_physical_shapes, output_shard_dtypes, output_shard_devices = (
            self.compute_physical_shape(
                resharded_args, adapted_kwargs, predicted_output_spec
            )
        )
        call_timings["compute_physical_shape"].append(time.perf_counter() - t0)

        # 9. Create sharded output
        t0 = time.perf_counter()
        output = sp.create_sharded_output(
            [],
            predicted_output_spec,
            any_traced,
            max_batch_dims,
            mesh=mesh,
            physical_shapes=output_physical_shapes,
            shard_dtypes=output_shard_dtypes,
            shard_devices=output_shard_devices,
        )
        output._impl.graph_values_epoch = -1
        GRAPH.add_unrealized(output._impl)
        call_timings["create_sharded_output"].append(time.perf_counter() - t0)

        # 10. Setup output refs
        t0 = time.perf_counter()
        self._setup_output_refs(output, resharded_args, kwargs, op_hash=op_hash)
        call_timings["setup_output_refs"].append(time.perf_counter() - t0)

        # Post-op collectives
        if reduce_axes and mesh:
            from nabla.ops.execution_utils import apply_auto_reduction
            output = apply_auto_reduction(output, reduce_axes, mesh)

        call_timings["total"].append(time.perf_counter() - t_total_start)
        return output
    
    # Fallback to original
    return original_op_call(self, *args, **kwargs)

# Apply patches
base.Operation.__call__ = instrumented_op_call
pytree.tree_flatten = timed("tree_flatten")(original_tree_flatten)
pytree.tree_leaves = timed("tree_leaves")(original_tree_leaves)
pytree.tree_map = timed("tree_map")(original_tree_map)
spmd.execute_on_shards = timed("execute_on_shards")(original_execute_on_shards)

# Run benchmark
print("Running 10 training steps with instrumentation...")
for _ in range(10):
    loss, grads = vg_fn(params, x, y)
    new_params = [p - g * lr for p, g in zip(params, grads)]
    nb.realize_all(loss, *new_params)
    params = new_params

# Report
print("\n" + "="*70)
print("Operation.__call__ breakdown (per call, microseconds):")
print("="*70)
for key, times in call_timings.items():
    if times:
        avg = np.mean(times) * 1e6
        total = np.sum(times) * 1000
        print(f"  {key:30s}: {avg:8.2f} µs avg, {total:8.2f} ms total ({len(times)} calls)")

print("\n" + "="*70)
print("Other hot functions:")
print("="*70)
for name, data in sorted(TIMINGS.items(), key=lambda x: -x[1]["total"]):
    avg = data["total"] / data["count"] * 1e6
    total = data["total"] * 1000
    print(f"  {name:30s}: {avg:8.2f} µs avg, {total:8.2f} ms total ({data['count']} calls)")
