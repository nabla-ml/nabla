#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Test concurrent model execution to see if that causes segfaults."""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_concurrent_execution():
    """Test if concurrent execution of the same model causes issues."""
    import nabla
    from nabla.core.execution_context import global_execution_context

    print("Testing concurrent model execution...")

    # Clear cache
    global_execution_context.clear()

    # Create operations that should share the same cached model
    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    # Create multiple identical operations
    operations = []
    for _i in range(5):
        z = nabla.mul(x, y)  # These should all get the same cache key
        operations.append(z)

    print(f"Created {len(operations)} identical operations")
    print(f"Cache size before realization: {global_execution_context.size()}")

    # Realize the first one to populate the cache
    print("Realizing first operation to populate cache...")
    operations[0].realize()
    print(f"Cache size after first realization: {global_execution_context.size()}")

    # Now try to realize the rest concurrently
    print("Testing concurrent realization of cached operations...")

    results = {}
    errors = {}

    def realize_operation(op_id, operation):
        """Realize an operation in a separate thread."""
        try:
            print(f"  Thread {op_id}: Starting realization...")
            operation.realize()
            results[op_id] = f"Success: {operation.shape}"
            print(f"  Thread {op_id}: ✅ Success")
        except Exception as e:
            errors[op_id] = str(e)
            print(f"  Thread {op_id}: ❌ Error: {e}")

    # Create and start threads for remaining operations
    threads = []
    for i, operation in enumerate(operations[1:], 1):  # Skip first one already realized
        thread = threading.Thread(target=realize_operation, args=(i, operation))
        threads.append(thread)

    # Start all threads
    print(f"Starting {len(threads)} concurrent realization threads...")
    for thread in threads:
        thread.start()
        time.sleep(0.1)  # Small delay to stagger starts

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\nResults:")
    for op_id in sorted(results.keys()):
        print(f"  Operation {op_id}: {results[op_id]}")

    if errors:
        print("\nErrors:")
        for op_id in sorted(errors.keys()):
            print(f"  Operation {op_id}: {errors[op_id]}")
        return False

    print("\n✅ All concurrent operations completed successfully!")
    return True


def test_mixed_operations_sequential():
    """Test mixed operations sequentially (this should work)."""
    import nabla
    from nabla.core.execution_context import global_execution_context

    print("\n" + "=" * 50)
    print("Testing mixed operations sequentially...")

    global_execution_context.clear()

    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    operations = [
        ("add", nabla.add(x, y)),
        ("mul", nabla.mul(x, y)),
        ("add2", nabla.add(x, y)),
        ("mul2", nabla.mul(x, y)),
    ]

    print(f"Cache size before: {global_execution_context.size()}")

    for name, op in operations:
        print(f"Realizing {name}...")
        try:
            op.realize()
            print(
                f"  ✅ {name} succeeded, cache size: {global_execution_context.size()}"
            )
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            return False

    print("✅ All mixed operations completed sequentially!")
    return True


def test_mixed_operations_concurrent():
    """Test mixed operations concurrently (this might segfault)."""
    import nabla
    from nabla.core.execution_context import global_execution_context

    print("\n" + "=" * 50)
    print("Testing mixed operations concurrently...")

    global_execution_context.clear()

    x = nabla.arange((2, 2))
    y = nabla.randn((2, 2), seed=42)

    operations = [
        ("add1", nabla.add(x, y)),
        ("mul1", nabla.mul(x, y)),
        ("add2", nabla.add(x, y)),
        ("mul2", nabla.mul(x, y)),
    ]

    print(f"Cache size before: {global_execution_context.size()}")

    results = {}
    errors = {}

    def realize_mixed_operation(name, operation):
        """Realize a mixed operation in a separate thread."""
        try:
            print(f"  Thread {name}: Starting...")
            operation.realize()
            results[name] = f"Success: {operation.shape}"
            print(f"  Thread {name}: ✅ Success")
        except Exception as e:
            errors[name] = str(e)
            print(f"  Thread {name}: ❌ Error: {e}")

    # Create and start threads
    threads = []
    for name, operation in operations:
        thread = threading.Thread(
            target=realize_mixed_operation, args=(name, operation)
        )
        threads.append(thread)

    print(f"Starting {len(threads)} concurrent mixed operation threads...")
    for thread in threads:
        thread.start()
        time.sleep(0.05)  # Very small delay

    # Wait for completion
    for thread in threads:
        thread.join()

    print("\nResults:")
    for name in [op[0] for op in operations]:
        if name in results:
            print(f"  {name}: {results[name]}")
        elif name in errors:
            print(f"  {name}: ERROR - {errors[name]}")

    if errors:
        print("\n❌ Some concurrent mixed operations failed!")
        return False

    print("\n✅ All concurrent mixed operations completed!")
    return True


if __name__ == "__main__":
    print("Testing if concurrent model execution causes segfaults...\n")

    success1 = test_concurrent_execution()
    success2 = test_mixed_operations_sequential()
    success3 = test_mixed_operations_concurrent()

    if success1 and success2 and success3:
        print("\n🎉 All concurrency tests passed!")
    else:
        print("\n❌ Some concurrency tests failed!")
