"""Raw communication-op probes on real GPUs with Trace + MAX graph dumps.

Run as a plain script (not pytest):
  EAGER_MAX_GRAPH=1 python tests/unit/test_communication_on_gpus/test_basic_ops.py
"""

from __future__ import annotations

import os
import time
import traceback

import nabla as nb
import numpy as np
from max.dtype import DType
from nabla.core import GRAPH, trace
from nabla.core.sharding.spec import DeviceMesh, DimSpec
from nabla.ops.communication import (
    all_reduce,
    all_to_all,
    axis_index,
    distributed_broadcast,
    gather_all_axes,
    ppermute,
    pmean,
    reduce_scatter,
    reshard,
    transfer_to,
    cpu,
    gpu,
    accelerator,
)


def _print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _print_trace_and_graph(name: str, fn, arg: nb.Tensor) -> None:
    show_max_graph = os.environ.get("PRINT_MAX_GRAPH", "1") == "1"

    print(f"\n--- {name}: Nabla Trace ---")
    tr = trace(fn, arg)
    print(tr)

    if show_max_graph:
        print(f"\n--- {name}: MAX Graph (eager build state) ---")
        print(GRAPH.graph)


def _run_case(name: str, fn, input_data: np.ndarray) -> None:
    case_filter = os.environ.get("CASE_FILTER", "").strip()
    if case_filter and case_filter not in name:
        return

    _print_header(name)
    GRAPH.clear_all()
    timings = os.environ.get("CASE_TIMINGS", "0") == "1"

    def _step(msg: str) -> None:
        if timings:
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {msg}", flush=True)

    try:
        _step("build input")
        inp = nb.Tensor.constant(input_data, dtype=DType.float32)

        _step("trace + print")
        t0 = time.perf_counter()
        _print_trace_and_graph(name, fn, inp)
        if timings:
            print(
                f"[{time.strftime('%H:%M:%S')}] {name}: trace/print took {time.perf_counter()-t0:.3f}s",
                flush=True,
            )

        _step("build output")
        out = fn(inp)

        _step("out.realize")
        t0 = time.perf_counter()
        out.realize()
        if timings:
            print(
                f"[{time.strftime('%H:%M:%S')}] {name}: out.realize took {time.perf_counter()-t0:.3f}s",
                flush=True,
            )

        print(f"\n--- {name}: Result ---")
        print(f"is_sharded={out.is_sharded}, shape={tuple(int(d) for d in out.shape)}")
        _step("gather if needed")
        gathered = out.gather() if out.is_sharded else out

        _step("gathered.realize")
        gathered.realize()

        _step("to numpy")
        t0 = time.perf_counter()
        print(gathered.numpy())
        if timings:
            print(
                f"[{time.strftime('%H:%M:%S')}] {name}: to_numpy took {time.perf_counter()-t0:.3f}s",
                flush=True,
            )
    except Exception as exc:
        print(f"[CASE FAILED] {name}: {exc}")
        traceback.print_exc()
    finally:
        if timings:
            print(f"[{time.strftime('%H:%M:%S')}] {name}: case complete", flush=True)


def _make_mesh(n_devices: int) -> DeviceMesh:
    return DeviceMesh(f"mesh_{n_devices}", shape=(n_devices,), axis_names=("x",))


def _run_for_mesh(n_devices: int) -> None:
    _print_header(f"MESH n={n_devices}")
    mesh = _make_mesh(n_devices)
    print(f"mesh={mesh}")
    print(f"is_distributed={mesh.is_distributed}")
    print(f"devices={mesh.devices}")

    dimspec_x = [DimSpec(["x"], is_open=False)]

    # 1) shard + all_gather (via Tensor.gather())
    data = np.arange(max(8, n_devices * 4), dtype=np.float32)

    def case_all_gather(t: nb.Tensor) -> nb.Tensor:
        return t.shard(mesh, dimspec_x).gather()

    _run_case(f"all_gather_from_shard_n{n_devices}", case_all_gather, data)

    # 1b) gather_all_axes (explicit API)
    data = np.arange(max(8, n_devices * 4), dtype=np.float32)

    def case_gather_all_axes(t: nb.Tensor) -> nb.Tensor:
        sh = t.shard(mesh, dimspec_x)
        return gather_all_axes(sh)

    _run_case(f"gather_all_axes_n{n_devices}", case_gather_all_axes, data)

    # 2) all_reduce (sum)
    data = np.ones(max(8, n_devices * 4), dtype=np.float32)

    def case_all_reduce(t: nb.Tensor) -> nb.Tensor:
        sh = t.shard(mesh, dimspec_x)
        return all_reduce(sh)

    _run_case(f"all_reduce_n{n_devices}", case_all_reduce, data)

    # 2b) pmean
    data = np.ones(max(8, n_devices * 4), dtype=np.float32)

    def case_pmean(t: nb.Tensor) -> nb.Tensor:
        sh = t.shard(mesh, dimspec_x)
        return pmean(sh, axis_name="x")

    _run_case(f"pmean_n{n_devices}", case_pmean, data)

    # 3) reduce_scatter (input is explicitly sharded to force comm path)
    rs_len = max(16, n_devices * n_devices * 4)
    data = np.ones(rs_len, dtype=np.float32)

    def case_reduce_scatter(t: nb.Tensor) -> nb.Tensor:
        sh = t.shard(mesh, dimspec_x)
        return reduce_scatter(sh, axis=0)

    _run_case(f"reduce_scatter_n{n_devices}", case_reduce_scatter, data)

    # 4) all_to_all (split/concat on axis 0)
    # Local shard size must be divisible by n_devices => global divisible by n^2.
    a2a_len = max(16, n_devices * n_devices * 4)
    data = np.arange(a2a_len, dtype=np.float32)

    def case_all_to_all(t: nb.Tensor) -> nb.Tensor:
        sh = t.shard(mesh, dimspec_x)
        return all_to_all(sh, split_axis=0, concat_axis=0)

    _run_case(f"all_to_all_n{n_devices}", case_all_to_all, data)

    # 5) ppermute (ring shift)
    data = np.arange(max(8, n_devices * 2), dtype=np.float32)
    ring_perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]

    def case_ppermute(t: nb.Tensor) -> nb.Tensor:
        sh = t.shard(mesh, dimspec_x)
        return ppermute(sh, ring_perm)

    _run_case(f"ppermute_n{n_devices}", case_ppermute, data)

    # 5b) distributed_broadcast
    data = np.arange(max(8, n_devices * 2), dtype=np.float32)

    def case_distributed_broadcast(t: nb.Tensor) -> nb.Tensor:
        return distributed_broadcast(t, mesh=mesh)

    _run_case(f"distributed_broadcast_n{n_devices}", case_distributed_broadcast, data)

    # 5c) reshard (explicit API)
    data = np.arange(max(8, n_devices * 4), dtype=np.float32)

    def case_reshard(t: nb.Tensor) -> nb.Tensor:
        sh = t.shard(mesh, dimspec_x)
        rep = [DimSpec([], is_open=True)]
        return reshard(sh, mesh, rep)

    _run_case(f"reshard_to_replicated_n{n_devices}", case_reshard, data)

    # 6) axis_index
    _print_header(f"axis_index_n{n_devices}")
    GRAPH.clear_all()

    probe = nb.Tensor.constant(np.zeros((1,), dtype=np.float32), dtype=DType.float32)

    def case_axis_index(_: nb.Tensor) -> nb.Tensor:
        return axis_index(mesh, "x")

    try:
        _print_trace_and_graph(f"axis_index_n{n_devices}", case_axis_index, probe)

        idx = axis_index(mesh, "x")
        idx.realize()
        idx_g = idx.gather()
        idx_g.realize()
        print("\n--- axis_index result ---")
        print(idx_g.numpy())
    except Exception as exc:
        print(f"[CASE FAILED] axis_index_n{n_devices}: {exc}")
        traceback.print_exc()

    # 7) Transfer helpers (exported communication APIs)
    _print_header(f"transfer_helpers_n{n_devices}")
    GRAPH.clear_all()
    t = nb.Tensor.constant(np.arange(8, dtype=np.float32), dtype=DType.float32)
    try:
        print("\n--- transfer_to(gpu0) ---")
        t_gpu0 = transfer_to(t, mesh.device_refs[0])
        t_gpu0.realize()
        print(t_gpu0.numpy())

        print("\n--- gpu()/cpu()/accelerator() helpers ---")
        t_gpu = gpu(t)
        t_gpu.realize()
        t_cpu = cpu(t_gpu)
        t_cpu.realize()
        t_acc0 = accelerator(t, device_id=0)
        t_acc0.realize()
        print("cpu(gpu(t)):", t_cpu.numpy())
        print("accelerator(t,0):", t_acc0.numpy())
    except Exception as exc:
        print(f"[CASE FAILED] transfer_helpers_n{n_devices}: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Communication-op probe start (real unsimulated path priority)")
    print("Tip: run with EAGER_MAX_GRAPH=1 for MAX graph visibility.")

    mesh_sizes = os.environ.get("MESH_SIZES", "8")
    sizes = tuple(int(s.strip()) for s in mesh_sizes.split(",") if s.strip())

    for n in sizes:
        try:
            _run_for_mesh(n)
        except Exception as exc:
            print(f"\n[FAILED] mesh n={n}: {exc}")
            traceback.print_exc()

    print("\nProbe complete.")
