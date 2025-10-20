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

"""
PyTorch-like imperative MLP training example using Nabla's .backward() API.

This demonstrates:
- Module base class (PyTorch-like nn.Module)
- Automatic parameter and submodule registration
- Modular layer and model design with classes
- Parameters marked with .requires_grad_()
- Forward pass through the model
- Loss computation and .backward() for gradients
- SGD optimizer for parameter updates
"""

import nabla as nb
from nabla.nn import Module, Linear, SGD


class MLP(Module):
    """Multi-Layer Perceptron with ReLU activations"""
    
    def __init__(self, layer_sizes: list[int]):
        """
        Initialize MLP with specified layer sizes.
        
        Args:
            layer_sizes: List of layer sizes, e.g., [1, 32, 64, 32, 1]
        """
        super().__init__()  # Initialize Module base class
        
        # Create layers - they get auto-registered as submodules!
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            # Manually register in _modules dict since we're using a list
            self._modules[f'layer_{i}'] = layer
    
    def forward(self, x: nb.Tensor) -> nb.Tensor:
        """
        Forward pass through all layers with ReLU activations.
        No activation on the final layer.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Can now call layer directly instead of layer.forward(x)!
            # Apply ReLU to all layers except the last
            if i < len(self.layers) - 1:
                x = nb.relu(x)
        return x


def mse_loss(predictions: nb.Tensor, targets: nb.Tensor) -> nb.Tensor:
    """Mean Squared Error loss"""
    diff = predictions - targets
    return nb.mean(diff * diff)


def main():
    """Main training loop showcasing the Module API"""
    print("=" * 60)
    print("Imperative MLP Training with Nabla (.backward() style)")
    print("=" * 60)
    print("\n🎯 Features demonstrated:")
    print("  • Module base class from nabla.nn")
    print("  • Linear layer from nabla.nn")
    print("  • SGD optimizer from nabla.nn")
    print("  • Automatic parameter registration")
    print("  • Recursive parameter access via .parameters()")
    print("  • Callable models: model(x)")
    print("  • Optimizer.step() for parameter updates")
    print()
    
    # Hyperparameters
    LAYER_SIZES = [1, 32, 64, 32, 1]
    BATCH_SIZE = 256
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 1000
    PRINT_INTERVAL = 100
    
    print(f"\nConfiguration:")
    print(f"  Architecture: {LAYER_SIZES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print()
    
    # Create model
    model = MLP(LAYER_SIZES)
    num_params = len(list(model.parameters()))
    print(f"Model created with {num_params} parameters")
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimizer: {optimizer}")
    
    # Generate random training data
    x_train = nb.rand((BATCH_SIZE, 1))
    y_train = nb.rand((BATCH_SIZE, 1))
    
    print(f"Training data: X{x_train.shape}, Y{y_train.shape}")
    print("\nStarting training...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(NUM_EPOCHS + 1):
        # Forward pass - now we can just call the model!
        predictions = model(x_train)  # Uses __call__ -> forward()
        
        # Compute loss
        loss = mse_loss(predictions, y_train)
        
        # Backward pass (compute gradients)
        loss.backward()
        
        # Update parameters using optimizer
        optimizer.step()
        
        # Print progress
        if epoch % PRINT_INTERVAL == 0:
            loss_val = loss.to_numpy()
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.6f}")
    
    print("-" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
