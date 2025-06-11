#!/usr/bin/env python3
"""Plan and implement missing components for nabla.nn module."""

import nabla as nb


def test_missing_operations():
    """Test what operations are missing from nabla core."""
    print("=== Testing Missing Core Operations ===")
    
    x = nb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    missing_ops = []
    
    # Test core operations that might be missing
    ops_to_test = [
        ("softmax", "nb.softmax(x, axis=-1)"),
        ("logsumexp", "nb.logsumexp(x, axis=-1)"),
        ("where", "nb.where(x > 3, x, 0)"),
        ("expand_dims", "nb.expand_dims(x, axis=0)"),
        ("squeeze", "nb.squeeze(nb.expand_dims(x, axis=0))"),
        ("mean", "nb.mean(x, axis=-1)"),
        ("std", "nb.std(x, axis=-1)"),
        ("var", "nb.var(x, axis=-1)"),
    ]
    
    for op_name, test_code in ops_to_test:
        try:
            result = eval(test_code)
            print(f"✅ {op_name}: Available")
        except AttributeError:
            missing_ops.append(op_name)
            print(f"❌ {op_name}: Missing from nabla")
        except Exception as e:
            missing_ops.append(op_name)
            print(f"⚠️  {op_name}: Error - {e}")
    
    return missing_ops


def plan_implementation():
    """Plan the implementation order."""
    print("\n=== Implementation Plan ===")
    
    # Priority 1: Essential missing operations
    priority_1 = [
        "softmax", "logsumexp", "where", "mean", "expand_dims", "squeeze"
    ]
    
    # Priority 2: More optimizers
    priority_2 = [
        "SGD", "Adam", "RMSprop"
    ]
    
    # Priority 3: Classification losses
    priority_3 = [
        "cross_entropy", "binary_cross_entropy", "softmax_cross_entropy"
    ]
    
    # Priority 4: Activation functions
    priority_4 = [
        "leaky_relu", "gelu", "silu", "softmax_activation"
    ]
    
    # Priority 5: More layers
    priority_5 = [
        "dropout", "batch_norm", "layer_norm"
    ]
    
    print("Priority 1 (Core Ops):", priority_1)
    print("Priority 2 (Optimizers):", priority_2)
    print("Priority 3 (Losses):", priority_3)
    print("Priority 4 (Activations):", priority_4)
    print("Priority 5 (Layers):", priority_5)


if __name__ == "__main__":
    missing_ops = test_missing_operations()
    plan_implementation()
    
    if missing_ops:
        print(f"\n📝 Critical missing operations: {missing_ops}")
        print("These need to be implemented first!")
    else:
        print("\n🎉 All core operations available! Can proceed with higher-level components.")
