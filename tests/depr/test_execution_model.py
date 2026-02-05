# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import os
import sys

os.environ["NABLA_DEBUG"] = "1"

sys.path.append(os.getcwd())

from nabla.core import Tensor


def compute_logic(x):

    y = x * x
    z = x + x
    return z


def test_untraced():
    print("\n" + "=" * 30)
    print(" TEST 1: UNTRACED EXECUTION")
    print("=" * 30 + "\n")

    x = Tensor.constant(1.0)

    z = compute_logic(x)

    print(">> Calling z.realize()...")
    print("EXPECTATION: Only 'z' should be evaluated. 'y' should be fused/dead.")
    z.realize()


def test_traced():
    print("\n" + "=" * 30)
    print(" TEST 2: TRACED EXECUTION")
    print("=" * 30 + "\n")

    x = Tensor.constant(1.0)
    x.traced = True

    z = compute_logic(x)

    print(">> Calling z.realize()...")
    print(
        "EXPECTATION: Only 'z' should be evaluated. 'y' is alive via refs but should be skipped by current fix."
    )
    z.realize()


from nabla.core.graph.tracing import trace


def test_long_chain_trace():
    print("\n" + "=" * 30)
    print(" TEST 3: FULL TRACE & LONG CHAIN")
    print("=" * 30 + "\n")

    def complex_logic(x):

        v1 = x * 2.0
        v2 = v1 + 10.0
        v3 = v2 / 2.0
        v4 = v3 - x
        v5 = v4 * v4
        return v5

    x = Tensor.constant(5.0)

    print(">> Tracing complex_logic...")
    t = trace(complex_logic, x)

    print(">> Trace Result:")
    print(t)

    print(f"\nCaptured {len(t.nodes)} ops in trace.")

    assert len(t.nodes) == 8, f"Expected 8 ops, got {len(t.nodes)}"
    print("✅ Logic Verified")

    print("\n>> Realizing output to check MAX Graph...")
    print(
        "EXPECTATION: Graph should contain the full chain but ONLY 'v5' (and maybe 'x') as outputs."
    )
    print("             Intermediates (v1..v4) should NOT be outputs.")

    res = t.outputs
    if isinstance(res, Tensor):
        res.realize()

    else:
        print("Warning: Output is not a tensor, cannot realize directly.")


def test_alive_intermediate():
    print("\n" + "=" * 30)
    print(" TEST 4: ALIVE INTERMEDIATE")
    print("=" * 30 + "\n")

    x = Tensor.constant(10.0)

    v1 = x * 2.0
    v2 = v1 + 5.0

    print(">> Calling v2.realize()...")
    print(
        "EXPECTATION: Both 'v2' AND 'v1' should be evaluated because 'v1' is alive in Python scope."
    )
    print("             'x' (constant) should also be evaluated/kept.")

    v2.realize()

    print(">> Checking if v1 is realized...")
    if v1.real:
        print("✅ v1 is realized (Correct)")

    else:
        print("❌ v1 is NOT realized (Unexpected)")


def test_multi_return():
    print("\n" + "=" * 30)
    print(" TEST 5: MULTI-RETURN PARTIAL REALIZE")
    print("=" * 30 + "\n")

    def fork_logic(x):
        a = x * 2.0
        b = x + 10.0
        return a, b

    x = Tensor.constant(5.0)
    res_a, res_b = fork_logic(x)

    print(">> Calling res_a.realize() ONLY...")
    print(
        "EXPECTATION: 'res_b' should ALSO be evaluated because it is alive/unrealized in this scope."
    )

    res_a.realize()

    print(f"res_a.real: {res_a.real}")
    print(f"res_b.real: {res_b.real}")

    if res_a.real and res_b.real:
        print("✅ Both outputs realized (Correct)")
    else:
        print("❌ One output missing (Unexpected)")


if __name__ == "__main__":
    try:
        test_untraced()
        test_traced()
        test_long_chain_trace()
        test_alive_intermediate()
        test_multi_return()
    except Exception as e:
        print(f"FAILED: {e}")
