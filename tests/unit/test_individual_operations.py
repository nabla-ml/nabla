#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Test individual operations to identify segfault source."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

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
