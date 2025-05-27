#!/usr/bin/env python3
"""Test individual operations to identify segfault source."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🔍 Testing operations individually to find segfault...")

try:
    import nabla

    print("✅ Import successful")

    # Test array creation
    print("Testing array creation...")
    x = nabla.arange((2, 3))
    y = nabla.arange((2, 3))
    print(f"✅ Arrays created: x.shape={x.shape}, y.shape={y.shape}")

    # Test each operation individually
    operations_to_test = [
        ("addition", lambda: x + y),
        ("multiplication", lambda: x * y),
        ("negate", lambda: nabla.negate(x)),
        ("sin", lambda: nabla.sin(x)),
        ("cos", lambda: nabla.cos(x)),
        ("reduce_sum_all", lambda: nabla.reduce_sum(x)),
        ("reduce_sum_axis", lambda: nabla.reduce_sum(x, axes=0)),
        ("transpose", lambda: nabla.transpose(x)),
        ("reshape", lambda: nabla.reshape(x, (3, 2))),
    ]

    for name, operation in operations_to_test:
        try:
            print(f"Testing {name}...")
            result = operation()
            print(f"✅ {name}: shape {result.shape}")
        except Exception as e:
            print(f"❌ {name}: failed with {e}")
            import traceback

            traceback.print_exc()
            break

    print("🎉 Individual operation tests completed!")

except Exception as e:
    print(f"❌ Error in individual testing: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
