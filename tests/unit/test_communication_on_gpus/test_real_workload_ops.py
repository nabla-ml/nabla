"""Run real distributed workload-style probes on a mesh."""

from __future__ import annotations

import os
import time
import traceback

import numpy as np
from max.dtype import DType

import nabla as nb
from nabla.core import GRAPH, trace
from nabla.core.sharding.spec import DeviceMesh, DimSpec
from nabla.ops.communication import all_reduce


def _print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _run_case(name: str, fn) -> bool:
    case_filter = os.environ.get("CASE_FILTER", "").strip()
    if case_filter and case_filter not in name:
        return True

    _print_header(name)
    GRAPH.clear_all()
    try:
        t0 = time.perf_counter()
        out = fn()
        out.realize()
        gathered = out.gather() if out.is_sharded else out
        gathered.realize()
        print(f"[{name}] result shape={tuple(int(d) for d in gathered.shape)}")
        print(gathered.numpy())
        print(f"[{name}] elapsed={time.perf_counter() - t0:.3f}s")
        return True
    except Exception as exc:
        print(f"[{name}] FAILED: {exc}")
        traceback.print_exc()
        return False


def _mesh(n_devices: int) -> DeviceMesh:
    return DeviceMesh(f"mesh_{n_devices}", shape=(n_devices,), axis_names=("x",))


def _case_sharded_matmul(mesh: DeviceMesh) -> bool:
    m = max(32, len(mesh.devices) * 8)
    k = 16
    n = 8

    lhs_np = np.arange(m * k, dtype=np.float32).reshape(m, k) / 100.0
    rhs_np = np.arange(k * n, dtype=np.float32).reshape(k, n) / 50.0

    lhs = nb.Tensor.constant(lhs_np, dtype=DType.float32).shard(
        mesh, [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)]
    )
    rhs = nb.distributed_broadcast(
        nb.Tensor.constant(rhs_np, dtype=DType.float32), mesh=mesh
    )

    def work() -> nb.Tensor:
        out = nb.matmul(lhs, rhs)
        return out

    if os.environ.get("PRINT_TRACE", "1") == "1":
        print("\n--- sharded_matmul: Nabla Trace ---")
        print(trace(lambda a, b: nb.matmul(a, b), lhs, rhs))
    if os.environ.get("PRINT_MAX_GRAPH", "1") == "1":
        print("\n--- sharded_matmul: MAX Graph ---")
        print(GRAPH.graph)

    out = work()
    out.realize()
    got = out.gather().numpy()
    ref = lhs_np @ rhs_np
    if not np.allclose(got, ref, rtol=1e-4, atol=5e-2):
        print("max abs err:", float(np.max(np.abs(got - ref))))
        return False
    return True


def _case_reshard_then_regular_ops(mesh: DeviceMesh) -> bool:
    x_np = np.linspace(-2.0, 2.0, max(32, len(mesh.devices) * 8), dtype=np.float32)
    x = nb.Tensor.constant(x_np, dtype=DType.float32).shard(mesh, [DimSpec(["x"])])

    def workload() -> nb.Tensor:
        y = nb.reshard(x, mesh, [DimSpec([], is_open=True)])
        z = nb.relu(y)
        w = z * z
        return w

    if os.environ.get("PRINT_TRACE", "1") == "1":
        print("\n--- reshard_then_regular_ops: Nabla Trace ---")
        print(
            trace(
                lambda t: (lambda y: nb.relu(y) * nb.relu(y))(
                    nb.reshard(t, mesh, [DimSpec([], is_open=True)])
                ),
                x,
            )
        )
    if os.environ.get("PRINT_MAX_GRAPH", "1") == "1":
        print("\n--- reshard_then_regular_ops: MAX Graph ---")
        print(GRAPH.graph)

    out = workload()
    out.realize()
    got = out.gather().numpy()
    ref = np.maximum(x_np, 0.0)
    ref = ref * ref
    return np.allclose(got, ref, rtol=1e-5, atol=1e-5)


def _case_sharded_chain_ops(mesh: DeviceMesh) -> bool:
    rows = max(32, len(mesh.devices) * 8)
    cols = 8
    x_np = np.linspace(-1.0, 1.0, rows * cols, dtype=np.float32).reshape(rows, cols)
    w_np = np.linspace(0.1, 1.1, cols * cols, dtype=np.float32).reshape(cols, cols)

    x = nb.Tensor.constant(x_np, dtype=DType.float32).shard(
        mesh,
        [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)],
    )
    w = nb.distributed_broadcast(
        nb.Tensor.constant(w_np, dtype=DType.float32), mesh=mesh
    )

    def mapped(a: nb.Tensor, b: nb.Tensor) -> nb.Tensor:
        y = nb.tanh(nb.matmul(a, b))
        y = nb.relu(y)
        y = nb.reshard(y, mesh, [DimSpec([], is_open=True), DimSpec([], is_open=True)])
        y = y * y
        return y

    def collective_probe(a: nb.Tensor) -> nb.Tensor:
        v = nb.mean(a, axis=1)
        v = v.shard(mesh, [DimSpec(["x"], is_open=False)])
        return all_reduce(v)

    if os.environ.get("PRINT_TRACE", "1") == "1":
        print("\n--- sharded_chain_ops: Nabla Trace ---")
        print(trace(mapped, x, w))
        print("\n--- sharded_chain_collective: Nabla Trace ---")
        print(trace(collective_probe, x))
    if os.environ.get("PRINT_MAX_GRAPH", "1") == "1":
        print("\n--- sharded_chain_ops: MAX Graph ---")
        print(GRAPH.graph)

    out = mapped(x, w)
    out.realize()
    collective_out = collective_probe(x)
    collective_out.realize()

    got = out.gather().numpy()
    ref = np.tanh(x_np @ w_np)
    ref = np.maximum(ref, 0.0)
    ref = ref * ref

    coll_np = collective_out.gather().numpy()
    coll_ok = np.isfinite(coll_np).all() and coll_np.size > 0

    if not coll_ok:
        print("collective_probe produced invalid values")
        return False

    if not np.allclose(got, ref, rtol=1e-4, atol=5e-4):
        print("chain max abs err:", float(np.max(np.abs(got - ref))))
        return False

    return True


def main() -> int:
    n = int(os.environ.get("MESH_SIZE", "8"))
    mesh = _mesh(n)
    case_filter = os.environ.get("CASE_FILTER", "").strip()
    print(f"Running real workload probes on mesh n={n}")

    ok = True

    ok = ok and _run_case(
        "workload_sharded_matmul_n8",
        lambda: nb.matmul(
            nb.Tensor.constant(
                np.arange(
                    max(32, len(mesh.devices) * 8) * 16, dtype=np.float32
                ).reshape(max(32, len(mesh.devices) * 8), 16)
                / 100.0,
                dtype=DType.float32,
            ).shard(mesh, [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)]),
            nb.distributed_broadcast(
                nb.Tensor.constant(
                    np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 50.0,
                    dtype=DType.float32,
                ),
                mesh=mesh,
            ),
        ),
    )

    if (
        not case_filter or "workload_sharded_matmul" in case_filter
    ) and not _case_sharded_matmul(mesh):
        print("[workload_sharded_matmul_n8] numerical check failed")
        ok = False

    ok = ok and _run_case(
        "workload_reshard_then_regular_ops_n8",
        lambda: (
            (lambda y: (nb.relu(y) * nb.relu(y)))(
                nb.reshard(
                    nb.Tensor.constant(
                        np.linspace(
                            -2.0, 2.0, max(32, len(mesh.devices) * 8), dtype=np.float32
                        ),
                        dtype=DType.float32,
                    ).shard(mesh, [DimSpec(["x"], is_open=False)]),
                    mesh,
                    [DimSpec([], is_open=True)],
                )
            )
        ),
    )

    if (
        not case_filter or "workload_reshard_then_regular_ops" in case_filter
    ) and not _case_reshard_then_regular_ops(mesh):
        print("[workload_reshard_then_regular_ops_n8] numerical check failed")
        ok = False

    ok = ok and _run_case(
        "workload_sharded_chain_ops_n8",
        lambda: (
            lambda x, w: nb.relu(
                nb.reshard(
                    nb.tanh(nb.matmul(x, w)),
                    mesh,
                    [DimSpec([], is_open=True), DimSpec([], is_open=True)],
                )
            )
        )(
            nb.Tensor.constant(
                np.linspace(
                    -1.0, 1.0, max(32, len(mesh.devices) * 8) * 8, dtype=np.float32
                ).reshape(max(32, len(mesh.devices) * 8), 8),
                dtype=DType.float32,
            ).shard(mesh, [DimSpec(["x"], is_open=False), DimSpec([], is_open=True)]),
            nb.distributed_broadcast(
                nb.Tensor.constant(
                    np.linspace(0.1, 1.1, 8 * 8, dtype=np.float32).reshape(8, 8),
                    dtype=DType.float32,
                ),
                mesh=mesh,
            ),
        ),
    )

    if (
        not case_filter or "workload_sharded_chain_ops" in case_filter
    ) and not _case_sharded_chain_ops(mesh):
        print("[workload_sharded_chain_ops_n8] numerical check failed")
        ok = False

    if ok:
        print("All real workload probes passed")
        return 0

    print("Real workload probes had failures")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
