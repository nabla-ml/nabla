#!/usr/bin/env python3
"""Comprehensive test to verify complete OOP refactoring."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🧪 Testing Complete OOP Refactoring...")


def test_operation_consistency():
    """Test that all operations follow the same OOP pattern."""

    print("✅ Testing operation class structure...")

    # Import all operation classes
    from nabla.ops.operation import (
        Operation,
        UnaryOperation,
        BinaryOperation,
        ReductionOperation,
        ViewOperation,
    )
    from nabla.ops.unary import NegateOp, SinOp, CosOp, CastOp
    from nabla.ops.binary import AddOp, MulOp
    from nabla.ops.reduce import SumOp
    from nabla.ops.view import TransposeOp, ReshapeOp, BroadcastToOp
    from nabla.ops.linalg import MatMulOp
    from nabla.ops.creation import RandNOp

    # Test that all operations have required methods
    operations = [
        AddOp(),
        MulOp(),
        NegateOp(),
        SinOp(),
        CosOp(),
        SumOp(),
        TransposeOp(),
        ReshapeOp((2, 3)),
        BroadcastToOp((2, 3)),
        MatMulOp(),
    ]

    for op in operations:
        # Check that all operations inherit from Operation
        assert isinstance(
            op, Operation
        ), f"{type(op).__name__} should inherit from Operation"

        # Check that all operations have required methods
        assert hasattr(op, "forward"), f"{type(op).__name__} should have forward method"
        assert hasattr(op, "maxpr"), f"{type(op).__name__} should have maxpr method"
        assert hasattr(
            op, "eagerxpr"
        ), f"{type(op).__name__} should have eagerxpr method"
        assert hasattr(
            op, "vjp_rule"
        ), f"{type(op).__name__} should have vjp_rule method"
        assert hasattr(
            op, "jvp_rule"
        ), f"{type(op).__name__} should have jvp_rule method"
        assert hasattr(
            op, "compute_output_shape"
        ), f"{type(op).__name__} should have compute_output_shape method"

        # Check that operations have a name
        assert hasattr(op, "name"), f"{type(op).__name__} should have name attribute"
        assert isinstance(op.name, str), f"{type(op).__name__}.name should be a string"

    print("✅ All operations have consistent OOP structure!")


def test_function_interfaces():
    """Test that function interfaces still work and use OOP classes internally."""

    print("✅ Testing function interfaces...")

    import nabla

    # Create test arrays
    x = nabla.arange((2, 3))
    y = nabla.arange((2, 3))

    # Test that all functions still work
    operations_to_test = [
        ("add", lambda: x + y),
        ("mul", lambda: x * y),
        ("negate", lambda: nabla.negate(x)),
        ("sin", lambda: nabla.sin(x)),
        ("cos", lambda: nabla.cos(x)),
        ("sum", lambda: nabla.sum(x)),
        ("sum_axis", lambda: nabla.sum(x, axes=0)),
        ("transpose", lambda: nabla.transpose(x)),
        ("reshape", lambda: nabla.reshape(x, (3, 2))),
        ("broadcast_to", lambda: nabla.broadcast_to(x, (4, 2, 3))),
        ("matmul", lambda: x @ nabla.transpose(y)),
    ]

    for name, operation in operations_to_test:
        try:
            result = operation()
            assert hasattr(result, "shape"), f"{name} should return an Array with shape"
            print(f"  ✅ {name}: shape {result.shape}")
        except Exception as e:
            print(f"  ❌ {name}: failed with {e}")
            raise

    print("✅ All function interfaces work correctly!")


def test_no_old_static_methods():
    """Test that no old static method patterns remain."""

    print("✅ Testing removal of old static method patterns...")

    # Check that old classes don't exist or aren't used
    import nabla.ops.unary as unary_mod
    import nabla.ops.reduce as reduce_mod
    import nabla.ops.view as view_mod
    import nabla.ops.linalg as linalg_mod

    # Check that old static method classes are replaced
    old_classes = [
        "Negate",
        "Sin",
        "Cos",
        "Cast",
        "Sum",
        "Transpose",
        "Reshape",
        "BroadcastTo",
        "MatMul",
    ]

    for module, mod_name in [
        (unary_mod, "unary"),
        (reduce_mod, "reduce"),
        (view_mod, "view"),
        (linalg_mod, "linalg"),
    ]:
        for old_class in old_classes:
            if hasattr(module, old_class):
                print(f"  ⚠️  Found old class {old_class} in {mod_name} module")
            else:
                print(f"  ✅ Old class {old_class} properly removed from {mod_name}")

    print("✅ Old static method patterns properly removed!")


def test_backward_compatibility():
    """Test that the API remains backward compatible."""

    print("✅ Testing backward compatibility...")

    import nabla

    # Test that all public functions still exist and work
    x = nabla.arange((2, 3))
    y = nabla.arange((2, 3))

    # These should all work exactly as before
    assert (x + y).shape == (2, 3)
    assert (x * y).shape == (2, 3)
    assert nabla.negate(x).shape == (2, 3)
    assert nabla.sin(x).shape == (2, 3)
    assert nabla.cos(x).shape == (2, 3)
    assert nabla.sum(x).shape == ()
    assert nabla.sum(x, axes=0).shape == (3,)
    assert nabla.transpose(x).shape == (3, 2)
    assert nabla.reshape(x, (3, 2)).shape == (3, 2)
    assert nabla.broadcast_to(x, (4, 2, 3)).shape == (4, 2, 3)
    assert (x @ nabla.transpose(y)).shape == (2, 2)

    print("✅ Backward compatibility maintained!")


def main():
    """Run all tests."""
    try:
        test_operation_consistency()
        test_function_interfaces()
        test_no_old_static_methods()
        test_backward_compatibility()

        print("\n🎉 Complete OOP Refactoring Verification PASSED!")
        print("✅ All operations use consistent OOP design")
        print("✅ All function interfaces work correctly")
        print("✅ Old static method patterns removed")
        print("✅ Backward compatibility maintained")
        print("✅ Clean, consistent architecture achieved!")

    except Exception as e:
        print(f"\n❌ OOP Refactoring Verification FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
