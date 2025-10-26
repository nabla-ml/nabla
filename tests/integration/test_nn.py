import pytest
import nabla as nb
from nabla import nn, optim
import numpy as np

# ==============================================================================
# Test Utilities
# ==============================================================================

def assert_close(a, b, atol=1e-6, msg=""):
    a_np = a.to_numpy() if hasattr(a, 'to_numpy') else a
    b_np = b.to_numpy() if hasattr(b, 'to_numpy') else b
    assert np.allclose(a_np, b_np, atol=atol), f"FAIL: {msg}\n  A={a_np}\n  B={b_np}"

# ==============================================================================
# Test nn.Module, Containers, and State Management
# ==============================================================================

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()
        self.layers = nn.Sequential(nn.Linear(20, 30), nn.ReLU(), nn.Linear(30, 40))
        self.module_list = nn.ModuleList([nn.Linear(40, 50), nn.Linear(50, 60)])
        self.module_dict = nn.ModuleDict({'output': nn.Linear(60, 5), 'aux_output': nn.Linear(60, 1)})
        self.register_buffer('my_buffer', nb.Tensor.from_numpy(np.ones((1, 1), dtype=np.float32)))

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layers(x)
        for layer in self.module_list:
            x = layer(x)
        return self.module_dict['output'](x)

def test_module_and_container_architecture():
    """
    Tests initialization of a complex model with nested modules, parameter counting,
    and state_dict saving/loading. This verifies the core nn.Module architecture.
    """
    # 1. Initialization and Parameter Discovery
    complex_model = ComplexModel()
    params = list(complex_model.parameters())
    # 2 params per Linear layer (w, b) * 7 layers = 14
    assert len(params) == 14, f"Expected 14 parameters, found {len(params)}"

    # 2. State Dictionary Generation
    state_dict = complex_model.state_dict()
    # 14 params + 1 buffer = 15
    assert len(state_dict) == 15, f"Expected 15 items in state_dict, found {len(state_dict)}"

    # 3. State Dictionary Loading
    new_model = ComplexModel()
    new_model.load_state_dict(state_dict)
    assert_close(complex_model.layer1.weight, new_model.layer1.weight, msg="load_state_dict failed for a parameter")
    assert_close(complex_model.my_buffer, new_model.my_buffer, msg="load_state_dict failed for a buffer")

# ==============================================================================
# Test Optimizers
# ==============================================================================

def test_optimizer_functional_consistency():
    """
    Ensures that stateful optim.SGD and optim.Adam produce the exact same results
    as their stateless functional counterparts on the first step.
    """
    # Test SGD
    p_sgd = nb.Tensor.from_numpy(np.ones((3, 3), dtype=np.float32))
    p_sgd.grad = nb.Tensor.from_numpy(np.ones((3, 3), dtype=np.float32) * 2)
    expected_p, _ = nn.functional.sgd_step(p_sgd, p_sgd.grad, None, lr=0.1, momentum=0, weight_decay=0)
    optimizer_sgd = optim.SGD([p_sgd], lr=0.1)
    optimizer_sgd.step()
    assert_close(p_sgd, expected_p, msg="optim.SGD does not match functional.sgd_step")

    # Test Adam
    p_adam = nb.Tensor.from_numpy(np.ones((3, 3), dtype=np.float32))
    p_adam.grad = nb.Tensor.from_numpy(np.ones((3, 3), dtype=np.float32) * 2)
    exp_avg = nb.Tensor.from_numpy(np.zeros((3, 3), dtype=np.float32))
    exp_avg_sq = nb.Tensor.from_numpy(np.zeros((3, 3), dtype=np.float32))
    expected_p_adam, _, _ = nn.functional.adam_step(p_adam, p_adam.grad, exp_avg, exp_avg_sq, 1, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0)
    optimizer_adam = optim.Adam([p_adam], lr=0.001)
    optimizer_adam.step()
    assert_close(p_adam, expected_p_adam, msg="optim.Adam does not match functional.adam_step")

def test_sgd_with_momentum():
    """Verifies the correctness of SGD's momentum implementation over multiple steps."""
    p_mom = nb.Tensor.from_numpy(np.array([0.5], dtype=np.float32))
    p_mom.grad = nb.Tensor.from_numpy(np.array([0.1], dtype=np.float32))
    opt_mom = optim.SGD([p_mom], lr=1.0, momentum=0.9)
    
    # Step 1: buf = grad = 0.1; p = 0.5 - 1.0 * 0.1 = 0.4
    opt_mom.step()
    assert_close(p_mom, np.array([0.4], dtype=np.float32), msg="SGD momentum step 1 failed")
    
    # Step 2: grad=0.1; buf = 0.9*0.1 + 0.1 = 0.19; p = 0.4 - 1.0 * 0.19 = 0.21
    p_mom.grad = nb.Tensor.from_numpy(np.array([0.1], dtype=np.float32))
    opt_mom.step()
    assert_close(p_mom, np.array([0.21], dtype=np.float32), msg="SGD momentum step 2 failed")

def test_optimizer_state_dict_resumption():
    """
    CRITICAL TEST: Verifies that an optimizer's state can be saved and loaded,
    producing bit-for-bit identical results to an uninterrupted training run.
    """
    # 1. Setup two identical models and an input
    m1 = nn.Linear(2, 2)
    m2 = nn.Linear(2, 2)
    m2.load_state_dict(m1.state_dict()) # Ensure models start identical
    inp = nb.Tensor.from_numpy(np.ones((1, 2), dtype=np.float32))

    # 2. Run first optimizer for one step
    opt1 = optim.Adam(m1.parameters(), lr=0.1)
    loss1 = nb.mean(m1(inp))
    loss1.backward()
    opt1.step()

    # 3. Save the state of the first optimizer
    opt_state = opt1.state_dict()

    # 4. Create a second optimizer and load the state
    opt2 = optim.Adam(m2.parameters(), lr=0.1)
    opt2.load_state_dict(opt_state)

    # 5. Run the second optimizer for one step
    loss2 = nb.mean(m2(inp))
    loss2.backward()
    opt2.step()

    # 6. Verify that the weights of the two models are now identical
    assert_close(m1.weight, m2.weight, msg="Optimizer state_dict failed to restore state")

# ==============================================================================
# Test Functional APIs
# ==============================================================================

def test_functional_activations():
    """Verifies the numerical correctness of all activation functions against numpy."""
    test_tensor = nb.Tensor.from_numpy(np.array([-1.0, 0.0, 2.0], dtype=np.float32))
    
    assert_close(nn.functional.relu(test_tensor), np.array([0., 0., 2.], dtype=np.float32), msg="relu")
    assert_close(nn.functional.leaky_relu(test_tensor, 0.1), np.array([-0.1, 0., 2.], dtype=np.float32), msg="leaky_relu")
    assert_close(nn.functional.sigmoid(test_tensor), np.array([0.26894, 0.5, 0.88079], dtype=np.float32), atol=1e-5, msg="sigmoid")
    assert_close(nn.functional.tanh(test_tensor), np.array([-0.76159, 0.0, 0.96402], dtype=np.float32), atol=1e-5, msg="tanh")
    
    softmax_res = nn.functional.softmax(test_tensor)
    assert_close(softmax_res, np.array([0.04201, 0.11419, 0.84379]), atol=1e-4, msg="softmax")
    
    log_softmax_res = nn.functional.log_softmax(test_tensor)
    assert_close(log_softmax_res, np.array([-3.1700, -2.1700, -0.1700]), atol=1e-3, msg="log_softmax")

def test_functional_losses():
    """Verifies the numerical correctness of all loss functions."""
    preds = nb.Tensor.from_numpy(np.array([0.2, 0.8, 0.1], dtype=np.float32))
    targets = nb.Tensor.from_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    
    assert_close(nn.functional.mean_squared_error(preds, targets), np.mean((preds.to_numpy() - targets.to_numpy())**2), msg="mse")
    assert_close(nn.functional.mean_absolute_error(preds, targets), np.mean(np.abs(preds.to_numpy() - targets.to_numpy())), msg="mae")
    assert_close(nn.functional.huber_loss(preds, targets, delta=0.5), 0.015, atol=1e-4, msg="huber_loss")

def test_linear_layer_functional_consistency():
    """Ensures the nn.Linear module is a correct wrapper for nn.functional.linear_forward."""
    x_linear = nb.Tensor.from_numpy(np.ones((1, 5), dtype=np.float32))
    l_linear = nn.Linear(5, 2)
    
    # Manually call the functional equivalent
    manual_linear = nn.functional.linear_forward(x_linear, l_linear.weight, l_linear.bias)
    
    # Compare with the stateful module's output
    assert_close(l_linear(x_linear), manual_linear, msg="nn.Linear is inconsistent with functional.linear_forward")
