# PyTorch nn.Module: Core Functionality Analysis

## What does PyTorch's `nn.Module` provide at its core?

### 1. **Automatic Parameter Management**
```python
# PyTorch automatically tracks parameters
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))  # Auto-registered
        self.linear = nn.Linear(10, 5)  # Its params auto-registered too
    
# Access all parameters recursively
model.parameters()  # Returns iterator over ALL params (including nested)
model.named_parameters()  # Returns (name, param) pairs
```

**Core benefit**: No manual tracking of parameters across nested modules!

### 2. **Module Registration & Hierarchical Structure**
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)  # Auto-registered as submodule
        self.layer2 = nn.Linear(20, 10)
    
# Access all submodules recursively
model.modules()  # Returns ALL modules including self
model.named_modules()  # Returns (name, module) pairs
model.children()  # Returns only direct children
```

**Core benefit**: Tree structure for complex architectures!

### 3. **Training/Eval Mode Management**
```python
model.train()  # Sets module and all children to training mode
model.eval()   # Sets module and all children to evaluation mode

# Affects layers like:
# - Dropout (active in train, disabled in eval)
# - BatchNorm (updates stats in train, uses fixed stats in eval)
```

**Core benefit**: Consistent behavior control across entire model!

### 4. **Device Management**
```python
model.to('cuda')  # Moves ALL parameters to GPU
model.cpu()       # Moves ALL parameters to CPU
model.to(dtype=torch.float16)  # Converts ALL parameters to FP16
```

**Core benefit**: One call moves everything!

### 5. **State Dict for Serialization**
```python
# Save model
torch.save(model.state_dict(), 'model.pt')

# Load model
model.load_state_dict(torch.load('model.pt'))

# State dict is just a dictionary: {'layer1.weight': tensor, 'layer1.bias': tensor, ...}
```

**Core benefit**: Easy checkpointing and model sharing!

### 6. **Hooks System**
```python
# Register hooks for debugging/visualization
def hook_fn(module, input, output):
    print(f"Output shape: {output.shape}")

model.layer1.register_forward_hook(hook_fn)
```

**Core benefit**: Debugging and introspection without modifying code!

### 7. **Gradient Management**
```python
model.zero_grad()  # Zeros gradients for ALL parameters
model.requires_grad_(False)  # Freezes ALL parameters
```

**Core benefit**: Batch operations on all parameters!

---

## What's ESSENTIAL for Nabla (Minimal Viable Module)?

### Must-Have:
1. ✅ **Parameter registration** - `self.add_parameter()` or automatic detection
2. ✅ **Recursive parameter access** - `model.parameters()` gets ALL params
3. ✅ **Submodule registration** - Nested modules work automatically
4. ✅ **`__call__` method** - Calls `forward()` with hooks support (future)

### Nice-to-Have (Phase 2):
5. ⏭️ Training/eval mode (`.train()`, `.eval()`)
6. ⏭️ Device movement (`.to(device)`)
7. ⏭️ State dict (`.state_dict()`, `.load_state_dict()`)
8. ⏭️ `zero_grad()` helper
9. ⏭️ Named parameters/modules

### Future (Phase 3):
10. ⏸️ Hooks system
11. ⏸️ `apply()` function for recursive operations
12. ⏸️ Buffer registration (non-trainable state)

---

## Proposed Nabla Module Design

```python
import nabla as nb
from typing import Iterator, Callable

class Module:
    """Base class for all neural network modules (PyTorch-like)."""
    
    def __init__(self):
        # Storage for parameters and submodules
        self._parameters: dict[str, nb.Tensor] = {}
        self._modules: dict[str, Module] = {}
    
    def __setattr__(self, name: str, value) -> None:
        """Intercept attribute setting to register params/modules."""
        if isinstance(value, nb.Tensor) and getattr(value, 'requires_grad', False):
            # Auto-register as parameter
            self._parameters[name] = value
        elif isinstance(value, Module):
            # Auto-register as submodule
            self._modules[name] = value
        # Also set as normal attribute
        object.__setattr__(self, name, value)
    
    def parameters(self) -> Iterator[nb.Tensor]:
        """Recursively yield all parameters."""
        # Yield own parameters
        for param in self._parameters.values():
            yield param
        # Yield submodule parameters
        for module in self._modules.values():
            yield from module.parameters()
    
    def forward(self, *args, **kwargs):
        """Define forward pass (must be overridden)."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        """Call forward method."""
        return self.forward(*args, **kwargs)
    
    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for param in self.parameters():
            param.grad = None
    
    def update_parameters_(self, new_params: list[nb.Tensor]) -> None:
        """Update parameters in-place (for imperative training)."""
        param_list = list(self.parameters())
        if len(new_params) != len(param_list):
            raise ValueError(f"Expected {len(param_list)} params, got {len(new_params)}")
        
        # This is tricky - need to update the actual stored tensors
        # May need a different approach...


# Usage example:
class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nb.glorot_uniform((in_features, out_features))
        self.bias = nb.zeros((1, out_features))
        self.weight.requires_grad_(True)  # Auto-registered!
        self.bias.requires_grad_(True)    # Auto-registered!
    
    def forward(self, x: nb.Tensor) -> nb.Tensor:
        return nb.matmul(x, self.weight) + self.bias

class MLP(Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        self.layers = [
            Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ]  # These get auto-registered as submodules!
    
    def forward(self, x: nb.Tensor) -> nb.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)  # Can call layer directly!
            if i < len(self.layers) - 1:
                x = nb.relu(x)
        return x

# Now training becomes cleaner:
model = MLP([1, 32, 64, 32, 1])
for epoch in range(1000):
    pred = model(x)  # Just call the model!
    loss = mse_loss(pred, y)
    loss.backward()
    new_params = sgd_step(list(model.parameters()), lr)
    # ... update model ...
    model.zero_grad()  # One call zeros all gradients!
```

---

## Key Challenge: Parameter Updates in Imperative Mode

The tricky part is updating parameters after SGD. PyTorch solves this with:
```python
optimizer.step()  # Updates param.data in-place
```

For Nabla imperative mode, we need a strategy:

### Option 1: In-place update (PyTorch style)
```python
# Update the underlying buffer without breaking the graph
param._impl = new_param._impl
```

### Option 2: Reassignment pattern (current approach)
```python
# Return new params and manually reassign
new_params = sgd_step(model.parameters(), lr)
model.update_parameters_(new_params)
```

### Option 3: Optimizer integration
```python
class SGD:
    def __init__(self, params):
        self.params = list(params)
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                # Create new tensor and update in-place
                new_val = param - self.lr * param.grad
                param._impl = new_val._impl
                param.grad = None
```

---

## Recommendation

**YES, implement a Module base class!** It will:

1. ✅ Make code cleaner and more intuitive
2. ✅ Provide foundation for future features (state dict, device movement, etc.)
3. ✅ Match user expectations (PyTorch users know this pattern)
4. ✅ Enable composable, reusable components

**Start minimal:**
- Parameter/module registration via `__setattr__`
- Recursive `.parameters()` method
- `__call__` → `forward()` dispatch
- `.zero_grad()` helper

**Save for later:**
- State dict / serialization
- Training/eval mode
- Device movement
- Hooks system

This gives immediate value without over-engineering!
