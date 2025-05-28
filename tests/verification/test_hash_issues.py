#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Test to demonstrate the hash collision issue."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_hash_determinism():
    """Test if the same graph produces the same hash."""
    import nabla
    from nabla.core.graph_execution import GraphTracer

    print("Testing hash determinism...")

    # Create the same operations multiple times
    for run in range(3):
        print(f"\nRun {run + 1}:")

        x = nabla.arange((2, 2))
        y = nabla.randn((2, 2), seed=42)

        z1 = nabla.add(x, y)
        z2 = nabla.mul(x, y)

        # Get traces and cache keys
        inputs1, trace1, cache_key1 = GraphTracer.get_trace([z1])
        inputs2, trace2, cache_key2 = GraphTracer.get_trace([z2])

        print(f"  Add operation cache key: {cache_key1}")
        print(f"  Mul operation cache key: {cache_key2}")

        # Print individual node hashes
        print("  Node hashes in add trace:")
        for i, node in enumerate(trace1):
            node_hash = GraphTracer.compute_node_hash(node)
            print(f"    {i}: {node.name} -> {node_hash}")

        print("  Node hashes in mul trace:")
        for i, node in enumerate(trace2):
            node_hash = GraphTracer.compute_node_hash(node)
            print(f"    {i}: {node.name} -> {node_hash}")


def test_hash_collision():
    """Test if different operations might get the same cache key."""
    import nabla
    from nabla.core.graph_execution import GraphTracer

    print("\nTesting potential hash collisions...")

    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    # Create multiple different operations with same inputs
    z_add = nabla.add(x, y)
    z_mul = nabla.mul(x, y)
    z_matmul = nabla.matmul(x, y)

    operations = [("add", z_add), ("mul", z_mul), ("matmul", z_matmul)]

    cache_keys = {}
    for op_name, z in operations:
        _, _, cache_key = GraphTracer.get_trace([z])
        cache_keys[op_name] = cache_key
        print(f"  {op_name}: cache_key = {cache_key}")

    # Check for collisions
    if len(set(cache_keys.values())) != len(cache_keys):
        print("  ❌ HASH COLLISION DETECTED!")
        return False
    else:
        print("  ✅ No hash collisions detected")
        return True


if __name__ == "__main__":
    test_hash_determinism()
    success = test_hash_collision()

    if not success:
        print("\n❌ Hash collision issue found!")
    else:
        print("\n✅ Hash system appears to work correctly")
