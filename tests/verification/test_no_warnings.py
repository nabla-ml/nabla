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

"""Test that operations don't produce numpy warnings."""

import warnings
import nabla


def test_operations_without_warnings():
    """Test that our operations don't produce warnings."""
    print("=== Testing Operations for Warnings ===")

    # Capture all warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        device = nabla.device("cpu")

        # Test with safe positive values to avoid log/power issues
        x = nabla.array([[2.0, 3.0, 4.0]]).to(device)
        y = nabla.array([[1.0, 2.0, 3.0]]).to(device)

        print("Testing log operation...")
        log_result = nabla.log(x)
        log_result.realize()

        print("Testing power operation...")
        pow_result = nabla.pow(x, y)
        pow_result.realize()

        print("Testing division operation...")
        div_result = nabla.div(x, y)
        div_result.realize()

        # Test VJP computations
        print("Testing VJP with log...")

        def log_fn(inputs):
            return [nabla.log(inputs[0])]

        loss_values, vjp_fn = nabla.vjp(log_fn, [x])
        cotangent = [nabla.array([[1.0, 1.0, 1.0]]).to(device)]
        gradients = vjp_fn(cotangent)

        print("Testing VJP with power...")

        def pow_fn(inputs):
            return [nabla.pow(inputs[0], inputs[1])]

        loss_values, vjp_fn = nabla.vjp(pow_fn, [x, y])
        cotangent = [nabla.array([[1.0, 1.0, 1.0]]).to(device)]
        gradients = vjp_fn(cotangent)

        # Check for warnings
        if warning_list:
            print(f"\n⚠️  Found {len(warning_list)} warnings:")
            for warning in warning_list:
                print(f"  - {warning.category.__name__}: {warning.message}")
                print(f"    File: {warning.filename}:{warning.lineno}")
        else:
            print("\n✅ No warnings detected!")

        return len(warning_list) == 0


if __name__ == "__main__":
    success = test_operations_without_warnings()
    if success:
        print("\n🎉 All operations run cleanly without warnings!")
    else:
        print("\n⚠️  Some warnings were detected.")
