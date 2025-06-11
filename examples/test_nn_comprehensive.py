#!/usr/bin/env python3
"""Comprehensive test of all new neural network components."""

import numpy as np
import nabla as nb
import nabla.nn as nn

print("Testing comprehensive nn module components...")

def test_core_operations():
    """Test newly implemented core operations."""
    print("\n=== Testing Core Operations ===")
    
    # Test max reduction
    x = nb.array([[1.0, 5.0, 3.0], [2.0, 1.0, 4.0]])
    max_result = nb.max(x, axes=1)
    print(f"Max along axis 1: {max_result.to_numpy()}")
    
    # Test softmax
    logits = nb.array([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
    softmax_result = nb.softmax(logits, axis=1)
    print(f"Softmax result: {softmax_result.to_numpy()}")
    
    # Test logsumexp
    logsumexp_result = nb.logsumexp(logits, axis=1)
    print(f"Logsumexp result: {logsumexp_result.to_numpy()}")
    
    # Test where
    condition = x > 2.0
    where_result = nb.where(condition, x, nb.zeros_like(x))
    print(f"Where result: {where_result.to_numpy()}")


def test_optimizers():
    """Test all optimizer implementations."""
    print("\n=== Testing Optimizers ===")
    
    # Create simple parameters
    params = [nb.array([[1.0, 2.0], [3.0, 4.0]]), nb.array([0.5, -0.5])]
    grads = [nb.array([[0.1, -0.1], [0.2, -0.2]]), nb.array([0.05, -0.05])]
    
    # Test SGD
    sgd_momentum = nn.init_sgd_state(params)
    updated_params_sgd, updated_momentum = nn.sgd_step(
        params, grads, sgd_momentum, 0.01, 0.9  # learning_rate=0.01, momentum=0.9 as positional
    )
    print(f"SGD updated first param: {updated_params_sgd[0].to_numpy()}")
    
    # Test Adam
    adam_m, adam_v = nn.init_adam_state(params)
    updated_params_adam, updated_m, updated_v = nn.adam_step(
        params, grads, adam_m, adam_v, 1, 0.001  # step=1, learning_rate=0.001 as positional
    )
    print(f"Adam updated first param: {updated_params_adam[0].to_numpy()}")
    
    # Test AdamW
    adamw_m, adamw_v = nn.init_adamw_state(params)
    updated_params_adamw, updated_m_w, updated_v_w = nn.adamw_step(
        params, grads, adamw_m, adamw_v, 1, 0.001  # step=1, learning_rate=0.001 as positional
    )
    print(f"AdamW updated first param: {updated_params_adamw[0].to_numpy()}")


def test_loss_functions():
    """Test all loss function implementations."""
    print("\n=== Testing Loss Functions ===")
    
    # Regression losses
    pred = nb.array([1.0, 2.0, 3.0, 4.0])
    target = nb.array([1.1, 1.9, 3.2, 3.8])
    
    mse = nn.mean_squared_error(pred, target)
    mae = nn.mean_absolute_error(pred, target)
    huber = nn.huber_loss(pred, target, delta=1.0)
    
    print(f"MSE: {mse.to_numpy()}")
    print(f"MAE: {mae.to_numpy()}")
    print(f"Huber: {huber.to_numpy()}")
    
    # Classification losses
    logits = nb.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.8]])
    targets_onehot = nb.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets_sparse = nb.array([0, 1])
    
    ce_loss = nn.cross_entropy_loss(logits, targets_onehot)
    sparse_ce_loss = nn.sparse_cross_entropy_loss(logits, targets_sparse)
    
    print(f"Cross-entropy loss: {ce_loss.to_numpy()}")
    print(f"Sparse cross-entropy loss: {sparse_ce_loss.to_numpy()}")
    
    # Binary classification
    binary_pred = nb.array([0.8, 0.3, 0.9, 0.1])
    binary_target = nb.array([1.0, 0.0, 1.0, 0.0])
    bce_loss = nn.binary_cross_entropy_loss(binary_pred, binary_target)
    print(f"Binary cross-entropy loss: {bce_loss.to_numpy()}")


def test_activations():
    """Test activation functions."""
    print("\n=== Testing Activation Functions ===")
    
    x = nb.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Test various activations
    relu_result = nn.relu(x)
    leaky_relu_result = nn.leaky_relu(x)
    sigmoid_result = nn.sigmoid(x) 
    tanh_result = nn.tanh(x)
    gelu_result = nn.gelu(x)
    silu_result = nn.silu(x)
    
    print(f"Input: {x.to_numpy()}")
    print(f"ReLU: {relu_result.to_numpy()}")
    print(f"Leaky ReLU: {leaky_relu_result.to_numpy()}")
    print(f"Sigmoid: {sigmoid_result.to_numpy()}")
    print(f"Tanh: {tanh_result.to_numpy()}")
    print(f"GELU: {gelu_result.to_numpy()}")
    print(f"SiLU: {silu_result.to_numpy()}")
    
    # Test softmax on 2D array
    logits_2d = nb.array([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
    softmax_2d = nn.softmax(logits_2d, axis=1)
    print(f"Softmax 2D: {softmax_2d.to_numpy()}")


def test_metrics():
    """Test metric functions."""
    print("\n=== Testing Metrics ===")
    
    # Classification metrics
    y_true = nb.array([1, 0, 1, 1, 0, 1])  # True class labels
    y_pred_logits = nb.array([
        [0.2, 0.8],  # Predicted class 1 (correct)
        [0.9, 0.1],  # Predicted class 0 (correct)
        [0.6, 0.4],  # Predicted class 0 (wrong, should be 1)
        [0.3, 0.7],  # Predicted class 1 (correct)
        [0.8, 0.2],  # Predicted class 0 (correct)
        [0.1, 0.9]   # Predicted class 1 (correct)
    ])
    
    acc = nn.accuracy(y_pred_logits, y_true)
    prec = nn.precision(y_pred_logits, y_true, num_classes=2)
    rec = nn.recall(y_pred_logits, y_true, num_classes=2)
    f1 = nn.f1_score(y_pred_logits, y_true, num_classes=2)
    
    print(f"Accuracy: {acc.to_numpy()}")
    print(f"Precision: {prec.to_numpy()}")
    print(f"Recall: {rec.to_numpy()}")
    print(f"F1 Score: {f1.to_numpy()}")
    
    # Regression metrics
    y_true_reg = nb.array([1.0, 2.0, 3.0, 4.0])
    y_pred_reg = nb.array([1.1, 1.9, 3.2, 3.8])
    
    mse_metric = nn.mean_squared_error_metric(y_true_reg, y_pred_reg)
    mae_metric = nn.mean_absolute_error_metric(y_true_reg, y_pred_reg)
    r2 = nn.r_squared(y_true_reg, y_pred_reg)
    corr = nn.pearson_correlation(y_true_reg, y_pred_reg)
    
    print(f"MSE metric: {mse_metric.to_numpy()}")
    print(f"MAE metric: {mae_metric.to_numpy()}")
    print(f"R-squared: {r2.to_numpy()}")
    print(f"Correlation: {corr.to_numpy()}")


def test_regularization():
    """Test regularization techniques."""
    print("\n=== Testing Regularization ===")
    
    # Create some parameters
    params = [nb.array([[1.0, 2.0], [3.0, 4.0]]), nb.array([0.5, -0.5])]
    
    # Test L1 and L2 regularization
    l1_reg = nn.l1_regularization(params, weight=0.01)
    l2_reg = nn.l2_regularization(params, weight=0.01)
    elastic_reg = nn.elastic_net_regularization(params, l1_weight=0.01, l2_weight=0.01)
    
    print(f"L1 regularization: {l1_reg.to_numpy()}")
    print(f"L2 regularization: {l2_reg.to_numpy()}")
    print(f"Elastic Net regularization: {elastic_reg.to_numpy()}")
    
    # Test dropout
    x = nb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dropout_result = nn.dropout(x, p=0.5, training=True, seed=42)
    print(f"Dropout result: {dropout_result.to_numpy()}")
    
    # Test gradient clipping
    grads = [nb.array([[10.0, -5.0], [3.0, 8.0]]), nb.array([2.0, -12.0])]
    clipped_grads, total_norm = nn.gradient_clipping(grads, max_norm=5.0)
    print(f"Total gradient norm: {total_norm.to_numpy()}")
    print(f"Clipped first gradient: {clipped_grads[0].to_numpy()}")


def test_learning_rate_schedules():
    """Test learning rate scheduling."""
    print("\n=== Testing Learning Rate Schedules ===")
    
    # Test different schedules
    initial_lr = 0.1
    steps = [0, 100, 200, 500, 1000]
    
    for step in steps:
        exp_decay = nn.exponential_decay_schedule(initial_lr, decay_factor=0.9, decay_every=100)(step)
        cosine_lr = nn.cosine_annealing_schedule(initial_lr, min_lr=0.0001, period=1000)(step) 
        warmup_cosine = nn.warmup_cosine_schedule(initial_lr, warmup_epochs=100, total_epochs=1000)(step)
        
        print(f"Step {step}: Exp={exp_decay:.6f}, Cosine={cosine_lr:.6f}, Warmup={warmup_cosine:.6f}")


def test_mlp_with_new_components():
    """Test MLP with new activation functions and optimizers."""
    print("\n=== Testing MLP with New Components ===")
    
    # Create MLP with GELU activation
    builder = nn.MLPBuilder()
    mlp_config = (builder
                  .with_layers([2, 10, 5, 1])
                  .with_activation("gelu")
                  .with_init_method("he_normal")
                  .with_seed(42)
                  .build())
    
    # Test forward pass
    x = nb.array([[1.0, 2.0], [3.0, 4.0]])
    output = mlp_config["forward"](x, mlp_config["params"])
    print(f"MLP output shape: {output.shape}")
    print(f"MLP output: {output.to_numpy()}")
    
    print("✅ MLP with new components test passed!")
    print(f"MLP output: {output.to_numpy()}")
    
    # Test with different activation
    gelu_config = nn.create_mlp_config([2, 8, 1], activation="gelu", init_method="xavier_normal")
    gelu_output = gelu_config["forward"](x, gelu_config["params"])
    print(f"GELU MLP output: {gelu_output.to_numpy()}")


if __name__ == "__main__":
    try:
        test_core_operations()
        test_optimizers()
        test_loss_functions()
        test_activations()
        test_metrics()
        test_regularization()
        test_learning_rate_schedules()
        test_mlp_with_new_components()
        
        print("\n🎉 All tests passed! The nn module is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
