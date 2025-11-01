import nabla as nb
from nabla.nn import Linear, ReLU, Sequential
from nabla.optim import SGD

# 1. Define a simple MLP model with explicit layers
class SimpleModel(nb.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Sequential(
            Linear(1, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 1)
        )

    def forward(self, x):
        return self.mlp(x)

# 2. Instantiate model and optimizer
model = SimpleModel()
optimizer = SGD(model.parameters(), lr=0.01)

# 3. Define a training step
def train(inputs, targets):
    optimizer.zero_grad()
    predictions = model.forward(inputs)
    loss = nb.sum((predictions - targets) ** 2)
    loss.backward()
    optimizer.step()
    return loss

# 4. Compile the training step with dynamic JIT (static JIT does not work here)
compiled_train = nb.compile(train)

# 5. Training loop with synthetic data
for i in range(1000):
    inputs = nb.randn((10, 1))
    targets = nb.sin(5 * inputs)
    loss = compiled_train(inputs, targets)

    if (i + 1) % 100 == 0:
        print(f"Epoch {i+1}, Loss: {loss}")