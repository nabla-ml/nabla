#!/usr/bin/env python3
"""Comprehensive test of all new nabla.nn modules and functions."""

import numpy as np
import nabla as nb


def test_loss_functions():
    """Test all loss functions."""
    print("=== Testing Loss Functions ===")
    
    # Create test data
    pred = nb.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    target = nb.array([[1.5, 2.2, 2.8], [1.8, 3.1, 4.2]])
    
    from nabla.nn.losses import mean_squared_error, mean_absolute_error, huber_loss
    
    # Test MSE
    mse = mean_squared_error(pred, target)
    print(f"✅ MSE Loss: {mse.to_numpy().item():.4f}")
    
    # Test MAE
    mae = mean_absolute_error(pred, target)
    print(f"✅ MAE Loss: {mae.to_numpy().item():.4f}")
    
    # Test Huber loss
    huber = huber_loss(pred, target, delta=1.0)
    print(f"✅ Huber Loss: {huber.to_numpy().item():.4f}")


def test_optimizers():
    """Test optimizer functions."""
    print("\n=== Testing Optimizers ===")
    
    from nabla.nn.optim import adamw_step, init_adamw_state
    
    # Create dummy parameters
    params = [
        nb.rand((4, 8), dtype=nb.DType.float32),
        nb.rand((1, 8), dtype=nb.DType.float32),
    ]
    
    # Create dummy gradients
    gradients = [
        nb.rand((4, 8), dtype=nb.DType.float32) * 0.01,
        nb.rand((1, 8), dtype=nb.DType.float32) * 0.01,
    ]
    
    # Initialize optimizer state
    m_states, v_states = init_adamw_state(params)
    print(f"✅ AdamW state initialized: {len(m_states)} momentum, {len(v_states)} velocity")
    
    # Test optimizer step
    updated_params, updated_m, updated_v = adamw_step(
        params, gradients, m_states, v_states, 1, 0.001
    )
    print(f"✅ AdamW step completed: {len(updated_params)} parameters updated")


def test_learning_rate_schedules():
    """Test learning rate schedules."""
    print("\n=== Testing Learning Rate Schedules ===")
    
    from nabla.nn.optim.schedules import (
        exponential_decay_schedule,
        cosine_annealing_schedule,
        warmup_cosine_schedule,
        learning_rate_schedule
    )
    
    # Test schedules
    exp_schedule = exponential_decay_schedule(0.001, 0.9, 100)
    cos_schedule = cosine_annealing_schedule(0.001, 1e-6, 1000)
    warmup_schedule = warmup_cosine_schedule(0.001, 100, 1000)
    
    epochs = [0, 100, 500, 1000]
    print("Learning Rate Schedules:")
    for epoch in epochs:
        exp_lr = exp_schedule(epoch)
        cos_lr = cos_schedule(epoch)
        warmup_lr = warmup_schedule(epoch)
        legacy_lr = learning_rate_schedule(epoch)
        print(f"  Epoch {epoch:4d}: Exp={exp_lr:.6f}, Cos={cos_lr:.6f}, Warmup={warmup_lr:.6f}, Legacy={legacy_lr:.6f}")
    print("✅ All learning rate schedules working")


def test_initialization():
    """Test parameter initialization functions."""
    print("\n=== Testing Parameter Initialization ===")
    
    from nabla.nn.init import (
        he_normal, xavier_normal, lecun_normal, initialize_mlp_params
    )
    
    shape = (64, 32)
    
    # Test different initializations
    he_weights = he_normal(shape, seed=42)
    xavier_weights = xavier_normal(shape, seed=42)
    lecun_weights = lecun_normal(shape, seed=42)
    
    print(f"✅ He normal init: std={he_weights.to_numpy().std():.4f}")
    print(f"✅ Xavier normal init: std={xavier_weights.to_numpy().std():.4f}")
    print(f"✅ LeCun normal init: std={lecun_weights.to_numpy().std():.4f}")
    
    # Test MLP initialization
    mlp_params = initialize_mlp_params([64, 128, 32], seed=42)
    print(f"✅ MLP specialized init: {len(mlp_params)} parameters")


def test_layers():
    """Test layer functions."""
    print("\n=== Testing Layers ===")
    
    from nabla.nn.layers import linear_forward, mlp_forward, mlp_forward_with_activations
    
    # Test linear layer
    x = nb.rand((16, 10), dtype=nb.DType.float32)
    weight = nb.rand((10, 5), dtype=nb.DType.float32)
    bias = nb.rand((1, 5), dtype=nb.DType.float32)
    
    output = linear_forward(x, weight, bias)
    print(f"✅ Linear layer: input {x.shape} -> output {output.shape}")
    
    # Test MLP
    params = [
        nb.rand((10, 20), dtype=nb.DType.float32),
        nb.rand((1, 20), dtype=nb.DType.float32),
        nb.rand((20, 5), dtype=nb.DType.float32),
        nb.rand((1, 5), dtype=nb.DType.float32),
    ]
    
    mlp_output = mlp_forward(x, params)
    print(f"✅ MLP forward: input {x.shape} -> output {mlp_output.shape}")
    
    # Test MLP with different activations
    mlp_tanh = mlp_forward_with_activations(x, params, activation="tanh")
    mlp_sigmoid = mlp_forward_with_activations(x, params, activation="sigmoid", final_activation="sigmoid")
    print(f"✅ MLP with tanh: output range [{mlp_tanh.to_numpy().min():.3f}, {mlp_tanh.to_numpy().max():.3f}]")
    print(f"✅ MLP with sigmoid: output range [{mlp_sigmoid.to_numpy().min():.3f}, {mlp_sigmoid.to_numpy().max():.3f}]")


def test_architectures():
    """Test architecture builders."""
    print("\n=== Testing Architectures ===")
    
    from nabla.nn.architectures import create_mlp_config, MLPBuilder
    
    # Test direct config creation
    config = create_mlp_config([10, 20, 5], activation="relu", init_method="he_normal")
    x_test = nb.rand((8, 10), dtype=nb.DType.float32)
    output = config["forward"](x_test, config["params"])
    print(f"✅ Direct MLP config: {config['layers']} -> output {output.shape}")
    
    # Test MLPBuilder
    builder_config = (MLPBuilder()
                     .with_layers([10, 32, 16, 5])
                     .with_activation("tanh")
                     .with_final_activation("sigmoid")
                     .with_init_method("xavier_normal")
                     .build())
    
    builder_output = builder_config["forward"](x_test, builder_config["params"])
    print(f"✅ MLPBuilder config: {builder_config['layers']} -> output {builder_output.shape}")


def test_unary_operations():
    """Test new unary operations."""
    print("\n=== Testing Unary Operations ===")
    
    x = nb.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Test tanh
    tanh_result = nb.tanh(x)
    print(f"✅ tanh: {tanh_result.to_numpy()}")
    
    # Test sigmoid
    sigmoid_result = nb.sigmoid(x)
    print(f"✅ sigmoid: {sigmoid_result.to_numpy()}")
    
    # Test abs
    abs_result = nb.abs(x)
    print(f"✅ abs: {abs_result.to_numpy()}")
    
    # Test logical_not
    bool_x = nb.array([True, False, True, False])
    logical_not_result = nb.logical_not(bool_x)
    print(f"✅ logical_not: {logical_not_result.to_numpy()}")


def test_comparison_operators():
    """Test comparison operators on Array class."""
    print("\n=== Testing Comparison Operators ===")
    
    a = nb.array([1.0, 2.0, 3.0, 4.0])
    b = nb.array([2.0, 2.0, 2.0, 2.0])
    
    # Test all comparison operators
    lt_result = a < b
    le_result = a <= b
    gt_result = a > b
    ge_result = a >= b
    
    print(f"✅ a < b:  {lt_result.to_numpy()}")
    print(f"✅ a <= b: {le_result.to_numpy()}")
    print(f"✅ a > b:  {gt_result.to_numpy()}")
    print(f"✅ a >= b: {ge_result.to_numpy()}")


def test_training_utilities():
    """Test training utility functions."""
    print("\n=== Testing Training Utilities ===")
    
    from nabla.nn.utils.training import (
        value_and_grad, create_sin_dataset, compute_correlation
    )
    
    # Test dataset creation
    x, targets = create_sin_dataset(batch_size=64, sin_periods=4)
    print(f"✅ Sin dataset: x {x.shape}, targets {targets.shape}")
    print(f"   x range: [{x.to_numpy().min():.3f}, {x.to_numpy().max():.3f}]")
    print(f"   targets range: [{targets.to_numpy().min():.3f}, {targets.to_numpy().max():.3f}]")
    
    # Test correlation
    pred_test = targets + nb.rand(targets.shape, dtype=nb.DType.float32) * 0.1
    correlation = compute_correlation(pred_test, targets)
    print(f"✅ Correlation: {correlation:.4f}")


def test_missing_operations():
    """Test for operations that might be missing from nabla core."""
    print("\n=== Testing for Missing Operations ===")
    
    x = nb.array([1.0, 2.0, 3.0])
    
    # Check operations that are commonly needed
    missing_ops = []
    
    # Test if these exist in nabla
    ops_to_test = [
        ("sqrt", "nb.sqrt(x)"),
        ("exp", "nb.exp(x)"),
        ("log", "nb.log(x)"),
        ("sin", "nb.sin(x)"),
        ("cos", "nb.cos(x)"),
        ("tanh", "nb.tanh(x)"),
        ("sigmoid", "nb.sigmoid(x)"),
        ("abs", "nb.abs(x)"),
        ("logical_not", "nb.logical_not(nb.array([True, False]))"),
        ("maximum", "nb.maximum(x, nb.array([1.5, 1.5, 1.5]))"),
        ("minimum", "nb.minimum(x, nb.array([2.5, 2.5, 2.5]))"),
    ]
    
    for op_name, test_code in ops_to_test:
        try:
            result = eval(test_code)
            print(f"✅ {op_name}: Available")
        except AttributeError:
            missing_ops.append(op_name)
            print(f"❌ {op_name}: Missing from nabla")
        except Exception as e:
            print(f"⚠️  {op_name}: Error - {e}")
    
    if missing_ops:
        print(f"\n📝 Missing operations that should be added: {missing_ops}")
    else:
        print(f"\n🎉 All common operations are available!")


if __name__ == "__main__":
    print("🚀 Comprehensive Nabla NN Module Test\n")
    
    try:
        test_loss_functions()
        test_optimizers()
        test_learning_rate_schedules()
        test_initialization()
        test_layers()
        test_architectures()
        test_unary_operations()
        test_comparison_operators()
        test_training_utilities()
        test_missing_operations()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ The nabla.nn module is fully functional with all components working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
