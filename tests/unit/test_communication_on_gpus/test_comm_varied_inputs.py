"""Rigorously probe communication ops on real meshes with varied inputs."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass

import nabla as nb
import numpy as np
from max.dtype import DType
from nabla.core import GRAPH, trace
from nabla.core.sharding.spec import DeviceMesh, DimSpec
from nabla.ops.communication import (
    all_reduce,
    all_to_all,
    distributed_broadcast,
    gather_all_axes,
    pmean,
    ppermute,
    reduce_scatter,
    reshard,
)


@dataclass
class Variant:
    name: str
    data: np.ndarray


def _print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _maybe_print_graphs(case: str, fn, x: nb.Tensor) -> None:
    if os.environ.get("PRINT_TRACE", "1") == "1":
        print(f"\n--- {case}: Nabla Trace ---")
        print(trace(fn, x))
    if os.environ.get("PRINT_MAX_GRAPH", "0") == "1":
        print(f"\n--- {case}: MAX Graph ---")
        print(GRAPH.graph)


def _run_case(case: str, fn, variant: Variant) -> tuple[bool, str]:
    case_filter = os.environ.get("CASE_FILTER", "").strip()
    if case_filter and case_filter not in case:
        return True, "skipped"

    GRAPH.clear_all()
    timings = os.environ.get("CASE_TIMINGS", "1") == "1"

    def _tick(msg: str, t0: float | None = None) -> float:
        now = time.perf_counter()
        if timings:
            if t0 is None:
                print(f"[{time.strftime('%H:%M:%S')}] {case}: {msg}", flush=True)
            else:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {case}: {msg} ({now - t0:.3f}s)",
                    flush=True,
                )
        return now

    _print_header(case)
    try:
        t0 = _tick("build input")
        x = nb.Tensor.constant(variant.data, dtype=DType.float32)

        t1 = _tick("trace/graph")
        _maybe_print_graphs(case, fn, x)
        _tick("trace/graph done", t1)

        _tick("build output")
        out = fn(x)

        t2 = _tick("realize")
        out.realize()
        _tick("realize done", t2)

        t3 = _tick("gather")
        gathered = out.gather() if out.is_sharded else out
        gathered.realize()
        res = gathered.numpy()
        _tick("gather done", t3)

        print(f"result shape={res.shape}, dtype={res.dtype}")
        print(res)
        print(f"[{case}] OK")
        _tick("case complete", t0)
        return True, "ok"
    except Exception as exc:
        print(f"[{case}] FAILED: {exc}")
        traceback.print_exc()
        return False, str(exc)


def _mesh(n_devices: int) -> DeviceMesh:
    return DeviceMesh(f"mesh_{n_devices}", shape=(n_devices,), axis_names=("x",))


def _build_variants(n: int) -> list[Variant]:
    base = max(16, n * 8)
    v1 = np.arange(base, dtype=np.float32)
    v2 = np.linspace(-3.0, 7.0, base, dtype=np.float32)
    v3 = (np.arange(base, dtype=np.float32) % 5) - 2.0
    return [
        Variant("arange", v1),
        Variant("linspace_signed", v2),
        Variant("periodic_signed", v3),
    ]


def _build_2d_variants(n: int) -> list[Variant]:
    rows = max(8, n * 2)
    cols = 8
    v1 = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    v2 = np.linspace(-1.0, 1.0, rows * cols, dtype=np.float32).reshape(rows, cols)
    return [Variant("matrix_arange", v1), Variant("matrix_linspace", v2)]


def run_mesh(n_devices: int) -> int:
    mesh = _mesh(n_devices)
    dimspec_x = [DimSpec(["x"], is_open=False)]
    failures = 0

    for v in _build_variants(n_devices):
        failures += not _run_case(
            f"all_gather_from_shard_{v.name}_n{n_devices}",
            lambda t, m=mesh, d=dimspec_x: t.shard(m, d).gather(),
            v,
        )[0]

    for v in _build_variants(n_devices):
        failures += not _run_case(
            f"gather_all_axes_{v.name}_n{n_devices}",
            lambda t, m=mesh, d=dimspec_x: gather_all_axes(t.shard(m, d)),
            v,
        )[0]

    for v in _build_variants(n_devices):
        failures += not _run_case(
            f"all_reduce_{v.name}_n{n_devices}",
            lambda t, m=mesh, d=dimspec_x: all_reduce(t.shard(m, d)),
            v,
        )[0]

    for v in _build_variants(n_devices):
        failures += not _run_case(
            f"pmean_{v.name}_n{n_devices}",
            lambda t, m=mesh, d=dimspec_x: pmean(t.shard(m, d), axis_name="x"),
            v,
        )[0]

    rs_len = max(16, n_devices * n_devices * 4)
    rs_data = np.linspace(1.0, 3.0, rs_len, dtype=np.float32)
    failures += not _run_case(
        f"reduce_scatter_weighted_n{n_devices}",
        lambda t, m=mesh, d=dimspec_x: reduce_scatter(t.shard(m, d), axis=0),
        Variant("reduce_scatter_weighted", rs_data),
    )[0]

    a2a_len = max(16, n_devices * n_devices * 4)
    a2a_data = np.arange(a2a_len, dtype=np.float32)
    failures += not _run_case(
        f"all_to_all_dense_n{n_devices}",
        lambda t, m=mesh, d=dimspec_x: all_to_all(t.shard(m, d), split_axis=0, concat_axis=0),
        Variant("all_to_all_dense", a2a_data),
    )[0]

    for v in _build_variants(n_devices):
        ring_perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        failures += not _run_case(
            f"ppermute_ring_{v.name}_n{n_devices}",
            lambda t, m=mesh, d=dimspec_x, p=ring_perm: ppermute(t.shard(m, d), p),
            v,
        )[0]

    for v in _build_variants(n_devices):
        failures += not _run_case(
            f"distributed_broadcast_{v.name}_n{n_devices}",
            lambda t, m=mesh: distributed_broadcast(t, mesh=m),
            v,
        )[0]

    for v in _build_2d_variants(n_devices):
        failures += not _run_case(
            f"reshard_to_replicated_{v.name}_n{n_devices}",
            lambda t, m=mesh, d=dimspec_x: reshard(
                t.shard(
                    m,
                    [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)],
                ),
                m,
                [DimSpec([], is_open=True), DimSpec([], is_open=True)],
            ),
            v,
        )[0]

    return failures


def main() -> int:
    mesh_sizes = os.environ.get("MESH_SIZES", "8")
    sizes = [int(s.strip()) for s in mesh_sizes.split(",") if s.strip()]
    print(f"Running varied-input communication probes for mesh sizes: {sizes}")

    failures = 0
    for n in sizes:
        print(f"\n=== MESH n={n} ===")
        failures += run_mesh(n)

    if failures:
        print(f"\nCompleted with failures={failures}")
        return 1

    print("\nAll varied-input communication probes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
