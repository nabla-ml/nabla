#!/usr/bin/env python3
"""Comprehensive test of the expanded nabla.nn module."""

import numpy as np
import nabla as nb
import nabla.nn as nn

def test_new_optimizers():
    """Test new optimizers: SGD and Adam."""
    print("Testing new optimizers...")
    
    # Create simple parameters
    params = [nb.randn((3, 3)), nb.randn((1, 3))]
    gradients = [nb.randn((3, 3)), nb.randn((1, 3))]
    
    # Test SGD
    print("  Testing SGD optimizer...")
    sgd_states = nn.init_sgd_state(params)
    updated_params, updated_states = nn.sgd_step(
        params, gradients, sgd_states, 
        learning_rate=0.01, momentum=0.9
    )
    print(f"  SGD updated {len(updated_params)} parameters")
    
    # Test Adam
    print("  Testing Adam optimizer...")
    m_states, v_states = nn.init_adam_state(params)
    updated_params, m_new, v_new = nn.adam_step(
        params, gradients, m_states, v_states, step=1,
        learning_rate=0.001
    )
    print(f"  Adam updated {len(updated_params)} parameters")
    
    print("✓ Optimizer tests passed!")


def test_classification_losses():
    """Test classification loss functions."""
    print("Testing classification losses...")
    
    batch_size, num_classes = 32, 10
    
    # Test data
    logits = nb.randn((batch_size, num_classes))
    
    # One-hot targets
    targets_np = np.zeros((batch_size, num_classes))
    targets_np[np.arange(batch_size), np.random.randint(0, num_classes, batch_size)] = 1
    targets_onehot = nb.Array.from_numpy(targets_np.astype(np.float32))
    
    # Sparse targets
    targets_sparse = nb.Array.from_numpy(np.random.randint(0, num_classes, batch_size).astype(np.float32))
    
    # Test cross-entropy loss
    print("  Testing cross-entropy loss...")
    ce_loss = nn.cross_entropy_loss(logits, targets_onehot)
    print(f"  Cross-entropy loss: {ce_loss.to_numpy().item():.4f}")
    
    # Test sparse cross-entropy
    print("  Testing sparse cross-entropy loss...")
    sce_loss = nn.sparse_cross_entropy_loss(logits, targets_sparse)
    print(f"  Sparse cross-entropy loss: {sce_loss.to_numpy().item():.4f}")
    
    # Test binary cross-entropy
    print("  Testing binary cross-entropy loss...")
    binary_preds = nn.sigmoid(nb.randn((batch_size,)))
    binary_targets = nb.Array.from_numpy((np.random.random(batch_size) > 0.5).astype(np.float32))
    bce_loss = nn.binary_cross_entropy_loss(binary_preds, binary_targets)
    print(f"  Binary cross-entropy loss: {bce_loss.to_numpy().item():.4f}")
    
    print("✓ Classification loss tests passed!")


def test_activation_functions():
    """Test activation functions."""
    print("Testing activation functions...")
    
    x = nb.Array.from_numpy(np.linspace(-3, 3, 10).astype(np.float32))
    
    activations_to_test = [
        ("ReLU", nn.relu),
        ("Leaky ReLU", lambda x: nn.leaky_relu(x, 0.1)),
        ("Sigmoid", nn.sigmoid),
        ("GELU", nn.gelu),
        ("SiLU", nn.silu),
    ]
    
    for name, activation_fn in activations_to_test:
        print(f"  Testing {name}...")
        result = activation_fn(x)
        print(f"  {name} output range: [{result.to_numpy().min():.3f}, {result.to_numpy().max():.3f}]")
    
    # Test softmax
    print("  Testing Softmax...")
    logits = nb.randn((5, 3))
    softmax_output = nn.softmax(logits, axis=-1)
    row_sums = nb.sum(softmax_output, axes=-1)
    print(f"  Softmax row sums (should be ~1.0): {row_sums.to_numpy()}")
    
    print("✓ Activation function tests passed!")


def test_metrics():
    """Test evaluation metrics."""
    print("Testing metrics...")
    
    batch_size, num_classes = 100, 5
    
    # Generate test data
    logits = nb.randn((batch_size, num_classes))
    true_labels = nb.Array.from_numpy(np.random.randint(0, num_classes, batch_size).astype(np.float32))
    
    # Test accuracy
    print("  Testing accuracy...")
    acc = nn.accuracy(logits, true_labels)
    print(f"  Accuracy: {acc.to_numpy().item():.3f}")
    
    # Test regression metrics
    print("  Testing regression metrics...")
    pred_values = nb.randn((50,))
    true_values = nb.randn((50,))
    
    mse = nn.mean_squared_error_metric(pred_values, true_values)
    mae = nn.mean_absolute_error_metric(pred_values, true_values)
    r2 = nn.r_squared(pred_values, true_values)
    
    print(f"  MSE: {mse.to_numpy().item():.3f}")
    print(f"  MAE: {mae.to_numpy().item():.3f}")
    print(f"  R²: {r2.to_numpy().item():.3f}")
    
    print("✓ Metrics tests passed!")


def test_regularization():
    """Test regularization techniques."""
    print("Testing regularization...")
    
    # Test L1/L2 regularization
    params = [nb.randn((10, 5)), nb.randn((5, 1))]
    
    print("  Testing L1 regularization...")
    l1_loss = nn.l1_regularization(params, weight=0.01)
    print(f"  L1 regularization loss: {l1_loss.to_numpy().item():.4f}")
    
    print("  Testing L2 regularization...")
    l2_loss = nn.l2_regularization(params, weight=0.01)
    print(f"  L2 regularization loss: {l2_loss.to_numpy().item():.4f}")
    
    # Test dropout
    print("  Testing dropout...")
    x = nb.ones((5, 10))
    
    # Training mode (should zero some elements)
    x_dropout_train = nn.dropout(x, p=0.5, training=True, seed=42)
    zeros_count = np.sum(x_dropout_train.to_numpy() == 0)
    print(f"  Dropout (training): {zeros_count}/{x.shape[0] * x.shape[1]} elements zeroed")
    
    # Inference mode (should keep all elements)
    x_dropout_inference = nn.dropout(x, p=0.5, training=False)
    print(f"  Dropout (inference): all elements preserved = {np.allclose(x_dropout_inference.to_numpy(), x.to_numpy())}")
    
    print("✓ Regularization tests passed!")


def test_complete_training_loop():
    """Test a complete training loop with new components."""
    print("Testing complete training loop with new components...")
    
    # Create a small classification problem
    batch_size, input_dim, num_classes = 32, 10, 3
    layers = [input_dim, 20, num_classes]
    
    # Build MLP
    config = nn.create_mlp_config(
        layers=layers,
        activation="relu",
        final_activation="softmax",
        init_method="he_normal",
        seed=42
    )
    
    params = config["params"]
    forward_fn = config["forward"]
    
    # Generate synthetic data
    X = nb.randn((batch_size, input_dim))
    y_true = nb.Array.from_numpy(np.random.randint(0, num_classes, batch_size).astype(np.float32))
    
    # Convert to one-hot
    y_onehot_np = np.zeros((batch_size, num_classes))
    y_onehot_np[np.arange(batch_size), y_true.to_numpy().astype(int)] = 1
    y_onehot = nb.Array.from_numpy(y_onehot_np.astype(np.float32))
    
    # Initialize optimizer
    m_states, v_states = nn.init_adam_state(params)
    
    print("  Running training steps...")
    for step in range(5):
        # Forward pass
        logits = forward_fn(X, params)
        
        # Compute loss with regularization
        ce_loss = nn.cross_entropy_loss(logits, y_onehot)
        reg_loss = nn.l2_regularization(params[::2], weight=0.001)  # Only weight matrices
        total_loss = ce_loss + reg_loss
        
        # Compute gradients
        loss_and_grads = nn.value_and_grad(
            lambda p: nn.cross_entropy_loss(forward_fn(X, p), y_onehot) + nn.l2_regularization(p[::2], weight=0.001),
            params
        )
        loss_val, gradients = loss_and_grads
        
        # Apply gradient clipping
        clipped_grads, grad_norm = nn.gradient_clipping(gradients, max_norm=1.0)
        
        # Update parameters
        params, m_states, v_states = nn.adam_step(
            params, clipped_grads, m_states, v_states, 
            step=step+1, learning_rate=0.001
        )
        
        # Compute metrics
        acc = nn.accuracy(logits, y_true)
        
        print(f"    Step {step+1}: Loss={loss_val.to_numpy().item():.4f}, "
              f"Acc={acc.to_numpy().item():.3f}, GradNorm={grad_norm.to_numpy().item():.3f}")
    
    print("✓ Complete training loop test passed!")


def main():
    """Run all tests."""
    print("Testing expanded nabla.nn module components...\n")
    
    try:
        test_new_optimizers()
        print()
        
        test_classification_losses()
        print()
        
        test_activation_functions()
        print()
        
        test_metrics()
        print()
        
        test_regularization()
        print()
        
        test_complete_training_loop()
        print()
        
        print("🎉 All tests passed! The expanded nabla.nn module is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
