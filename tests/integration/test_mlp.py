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

import nabla


def test_mlp_training():
    """Test a complete MLP training setup using only VJP."""
    print("=== Testing MLP Training ===")

    device = nabla.device("cpu")  # Change to "gpu:0" for GPU testing

    def mlp_loss(inputs: list[nabla.Array]) -> list[nabla.Array]:
        """MLP forward pass + loss computation.
        inputs = [x, w1, b1, w2, b2, targets]
        returns = [loss]
        """
        x, targets, w1, b1, w2, b2 = inputs
        
        # Hidden layer: x @ w1 + b1, then ReLU-like activation
        hidden = nabla.relu(nabla.matmul(x, w1) + b1)  # Using sin as activation
        
        # Output layer: hidden @ w2 + b2
        output = nabla.matmul(hidden, w2) + b2
        
        # Mean squared error loss
        diff = output - targets  # Fixed: should be minus, not plus
        loss = nabla.reduce_sum(diff * diff)
        
        return [loss]

    # Create simple training data (2 samples, 3 input features, 2 output features)
    x = nabla.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(device)  # (2, 3)
    targets = nabla.array([[0.5, 1.5], [2.5, 3.5]]).to(device)      # (2, 2)
    
    # # Initialize MLP parameters
    params = [
        nabla.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).to(device),  # w1 (3, 2)
        nabla.array([[0.1, 0.2]]).to(device),                            # b1 (1, 2)
        nabla.array([[0.7, 0.8], [0.9, 1.0]]).to(device),               # w2 (2, 2)
        nabla.array([[0.1, 0.2]]).to(device),                            # b2 (1, 2)
    ]
    learning_rate = -0.01  # Negative to simulate subtraction in gradient descent
    
    # Training loop
    for epoch in range(3):
        # Forward pass + VJP for gradients
        all_inputs = [x, targets] + params
        loss_values, vjp_fn = nabla.vjp(mlp_loss, all_inputs)
        
        # Backward pass
        cotangent = [nabla.array([1.0]).to(device)]  # Simple scalar cotangent
        outputs = vjp_fn(cotangent)
        
        # Update parameters using gradients
        for i in range(len(params)):
            param_idx = i + 2
            params[i] = params[i] + nabla.array([learning_rate]).to(device) * outputs[param_idx]
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Loss: {loss_values[0]}")


if __name__ == "__main__":
    test_mlp_training()
