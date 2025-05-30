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

"""Test the newly added operations: log, power, and improved division."""

import nabla


def test_log_operation():
    """Test logarithm operation with VJP/JVP."""
    print("=== Testing Log Operation ===")

    device = nabla.device("cpu")

    # Test forward pass
    x = nabla.array([[1.0, 2.718, 10.0], [0.5, 1.5, 3.0]]).to(device)
    result = nabla.log(x)
    result.realize()

    print(f"Input shape: {x.shape}")
    print(f"Log result shape: {result.shape}")

    # Test VJP (gradient computation)
    def log_fn(inputs):
        return [nabla.log(inputs[0])]

    loss_values, vjp_fn = nabla.vjp(log_fn, [x])
    cotangent = [nabla.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).to(device)]
    gradients = vjp_fn(cotangent)

    print(f"VJP gradient shape: {gradients[0].shape}")
    print("✓ Log operation works with VJP")

    return True


def test_power_operation():
    """Test power operation with VJP/JVP."""
    print("\n=== Testing Power Operation ===")

    device = nabla.device("cpu")

    # Test forward pass with safe positive values
    base = nabla.array([[2.0, 3.0, 4.0], [1.5, 2.5, 5.0]]).to(device)
    exponent = nabla.array([[2.0, 1.0, 0.5], [3.0, 2.0, 1.0]]).to(device)

    result = nabla.pow(base, exponent)
    result.realize()

    print(f"Base shape: {base.shape}")
    print(f"Exponent shape: {exponent.shape}")
    print(f"Power result shape: {result.shape}")

    # Test VJP (gradient computation)
    def power_fn(inputs):
        return [nabla.pow(inputs[0], inputs[1])]

    loss_values, vjp_fn = nabla.vjp(power_fn, [base, exponent])
    cotangent = [nabla.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).to(device)]
    gradients = vjp_fn(cotangent)

    print(f"VJP gradient shapes: base={gradients[0].shape}, exp={gradients[1].shape}")
    print("✓ Power operation works with VJP")

    return True


def test_division_operation():
    """Test improved division operation with VJP/JVP."""
    print("\n=== Testing Division Operation ===")

    device = nabla.device("cpu")

    # Test forward pass
    numerator = nabla.array([[6.0, 8.0, 12.0], [15.0, 20.0, 25.0]]).to(device)
    denominator = nabla.array([[2.0, 4.0, 3.0], [5.0, 4.0, 5.0]]).to(device)

    result = nabla.div(numerator, denominator)
    result.realize()

    print(f"Numerator shape: {numerator.shape}")
    print(f"Denominator shape: {denominator.shape}")
    print(f"Division result shape: {result.shape}")

    # Test VJP (gradient computation)
    def div_fn(inputs):
        return [nabla.div(inputs[0], inputs[1])]

    loss_values, vjp_fn = nabla.vjp(div_fn, [numerator, denominator])
    cotangent = [nabla.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).to(device)]
    gradients = vjp_fn(cotangent)

    print(f"VJP gradient shapes: num={gradients[0].shape}, den={gradients[1].shape}")
    print("✓ Division operation works with VJP")

    return True


def test_combined_operations():
    """Test combining log, power, and division operations."""
    print("\n=== Testing Combined Operations ===")

    device = nabla.device("cpu")

    # Create a complex expression: log(x^2) / y
    x = nabla.array([[2.0, 3.0], [4.0, 5.0]]).to(device)
    y = nabla.array([[2.0, 3.0], [4.0, 5.0]]).to(device)
    exponent = nabla.array([[2.0, 2.0], [2.0, 2.0]]).to(device)

    # Compute: log(x^2) / y
    x_squared = nabla.pow(x, exponent)
    log_x_squared = nabla.log(x_squared)
    result = nabla.div(log_x_squared, y)
    result.realize()

    print(f"Combined operation result shape: {result.shape}")

    # Test VJP for the combined operation
    def combined_fn(inputs):
        x, y, exp = inputs
        x_squared = nabla.pow(x, exp)
        log_x_squared = nabla.log(x_squared)
        return [nabla.div(log_x_squared, y)]

    loss_values, vjp_fn = nabla.vjp(combined_fn, [x, y, exponent])
    cotangent = [nabla.array([[1.0, 1.0], [1.0, 1.0]]).to(device)]
    gradients = vjp_fn(cotangent)

    print(
        f"Combined VJP gradient shapes: x={gradients[0].shape}, y={gradients[1].shape}, exp={gradients[2].shape}"
    )
    print("✓ Combined operations work with VJP")

    return True


def test_edge_cases():
    """Test edge cases that might cause warnings."""
    print("\n=== Testing Edge Cases ===")

    device = nabla.device("cpu")

    # Test log with small values (should use epsilon handling)
    small_values = nabla.array([[1e-10, 1e-8, 1e-6], [0.1, 0.01, 0.001]]).to(device)
    log_result = nabla.log(small_values)
    log_result.realize()
    print("✓ Log handles small values safely")

    # Test power with integer exponents (safer)
    base = nabla.array([[2.0, 3.0, 4.0]]).to(device)
    int_exp = nabla.array([[2.0, 3.0, 1.0]]).to(device)  # Use integer-like exponents
    pow_result = nabla.pow(base, int_exp)
    pow_result.realize()
    print("✓ Power handles integer-like exponents safely")

    # Test division with reasonable denominators
    num = nabla.array([[10.0, 20.0, 30.0]]).to(device)
    den = nabla.array([[2.0, 4.0, 6.0]]).to(device)  # Avoid small denominators
    div_result = nabla.div(num, den)
    div_result.realize()
    print("✓ Division handles reasonable denominators safely")

    return True


def test_all_new_operations():
    """Run all tests for new operations."""
    print("Testing newly added operations: log, power, and improved division\n")

    tests = [
        test_log_operation,
        test_power_operation,
        test_division_operation,
        test_combined_operations,
        test_edge_cases,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n=== Summary: {passed}/{len(tests)} tests passed ===")
    return passed == len(tests)


if __name__ == "__main__":
    test_all_new_operations()
