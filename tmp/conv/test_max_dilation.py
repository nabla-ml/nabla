#!/usr/bin/env python
"""Test to isolate which MAX operation fails with dilation."""

import nabla as nb

print("Testing MAX dilation support...")
print("=" * 60)

# Test conv2d with different dilations
for dil in [(1, 1), (2, 2), (3, 3)]:
    print(f"\nTesting conv2d forward with dilation={dil}")
    try:
        @nb.compile
        def test_conv2d(x, w):
            y = nb.conv2d(x, w, stride=1, padding=1, dilation=dil)
            return y
        
        x = nb.randn((2, 3, 8, 8))
        w = nb.randn((16, 3, 3, 3))
        result = test_conv2d(x, w)
        print(f"   ✓ PASS - Output shape: {result.shape}")
    except Exception as e:
        print(f"   ✗ FAIL - {e}")

print("\n" + "-" * 60)

# Test conv2d_transpose with different dilations
for dil in [(1, 1), (2, 2), (3, 3)]:
    print(f"\nTesting conv2d_transpose forward with dilation={dil}")
    try:
        @nb.compile
        def test_conv2d_transpose(x, w):
            y = nb.conv2d_transpose(x, w, stride=1, padding=1, dilation=dil)
            return y
        
        x = nb.randn((2, 16, 8, 8))
        w = nb.randn((16, 3, 3, 3))  # (in_ch, out_ch, h, w)
        result = test_conv2d_transpose(x, w)
        print(f"   ✓ PASS - Output shape: {result.shape}")
    except Exception as e:
        print(f"   ✗ FAIL - {e}")

print("\n" + "=" * 60)
print("Test complete!")
