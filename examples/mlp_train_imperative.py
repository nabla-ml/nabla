import nabla as nb
from nabla.nn import Linear, ReLU
from nabla.optim import SGD

# 1. Define a simple MLP model with explicit layers
class SimpleModel(nb.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(1, 64)
        self.relu = ReLU()
        self.linear2 = Linear(64, 64)
        self.relu = ReLU()
        self.linear3 = Linear(64, 1)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

# 2. Instantiate model and optimizer
model = SimpleModel()
optimizer = SGD(model.parameters(), lr=0.01)

# 3. Define a training step
def train(inputs, targets):
    predictions = model.forward(inputs)
    loss = nb.sum((predictions - targets) ** 2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
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