"""Run GPU communication probe cases in isolated subprocesses.

This avoids shared graph/runtime state and pinpoints hangs/failures per case.

Usage:
  source venv/bin/activate
  python tests/unit/test_communication_on_gpus/run_cases_isolated.py

Optional env vars:
  CASE_TIMEOUT_SEC=30
  MESH_SIZES=8
  EAGER_MAX_GRAPH=1
  PRINT_MAX_GRAPH=0
  CASE_TIMINGS=1
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class CaseResult:
    case: str
    status: str
    duration_s: float
    exit_code: int | None
    note: str


CASES = [
    "all_gather_from_shard_n8",
    "gather_all_axes_n8",
    "all_reduce_n8",
    "pmean_n8",
    "reduce_scatter_n8",
    "all_to_all_n8",
    "ppermute_n8",
    "distributed_broadcast_n8",
    "reshard_to_replicated_n8",
]


def run_case(case: str, timeout_sec: int) -> CaseResult:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("EAGER_MAX_GRAPH", "1")
    env.setdefault("PRINT_MAX_GRAPH", "0")
    env.setdefault("CASE_TIMINGS", "1")
    env.setdefault("MESH_SIZES", "8")
    env["CASE_FILTER"] = case

    cmd = [
        sys.executable,
        "tests/unit/test_communication_on_gpus/test_basic_ops.py",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        dt = time.perf_counter() - t0

        out = (proc.stdout or "") + "\n" + (proc.stderr or "")

        if "[CASE FAILED]" in out:
            return CaseResult(case, "failed", dt, proc.returncode, "[CASE FAILED] seen")
        if proc.returncode != 0:
            return CaseResult(case, "error", dt, proc.returncode, f"exit={proc.returncode}")
        if f"{case}: case complete" not in out:
            return CaseResult(case, "incomplete", dt, proc.returncode, "missing case-complete marker")

        return CaseResult(case, "ok", dt, proc.returncode, "")

    except subprocess.TimeoutExpired as exc:
        dt = time.perf_counter() - t0
        out_part = exc.stdout or ""
        err_part = exc.stderr or ""
        if isinstance(out_part, bytes):
            out_part = out_part.decode("utf-8", errors="replace")
        if isinstance(err_part, bytes):
            err_part = err_part.decode("utf-8", errors="replace")
        tail = (out_part + "\n" + err_part)[-500:]
        note = "timeout"
        if tail.strip():
            note += f"; tail={tail!r}"
        return CaseResult(case, "timeout", dt, None, note)


def main() -> int:
    timeout_sec = int(os.environ.get("CASE_TIMEOUT_SEC", "30"))
    case_filter = os.environ.get("CASE_FILTER", "").strip()

    cases = CASES
    if case_filter:
        cases = [c for c in CASES if case_filter in c]
        if not cases:
            print(f"No cases match CASE_FILTER={case_filter!r}")
            return 2

    print(f"Running {len(cases)} isolated cases with timeout={timeout_sec}s each")
    print("-" * 88)

    results: list[CaseResult] = []
    for case in cases:
        print(f"[RUN] {case}")
        res = run_case(case, timeout_sec)
        results.append(res)
        print(
            f"[{res.status.upper():8}] {case:30}  {res.duration_s:6.2f}s"
            + (f"  ({res.note})" if res.note else "")
        )

    print("\nSummary")
    print("-" * 88)
    for r in results:
        print(f"{r.status:10} {r.case:30} {r.duration_s:6.2f}s {r.note}")

    bad = [r for r in results if r.status != "ok"]
    if bad:
        print(f"\nNon-OK cases: {len(bad)}")
        for b in bad:
            print(f" - {b.case}: {b.status} ({b.note})")
        return 1

    print("\nAll isolated cases completed OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
