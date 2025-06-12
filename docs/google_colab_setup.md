# Google Colab Setup Guide for Nabla

## Quick Setup

Run this in a Google Colab cell to install and verify Nabla:

```python
import sys
print(f"Python Version: {sys.version}")

# Install nabla-ml
!pip install nabla-ml --upgrade

# Test import
try:
    import nabla as nb
    print("✅ SUCCESS: Nabla is installed and ready!")
    print(f"📍 Nabla package location: {nb.__file__}")
except ImportError as e:
    print(f"❌ ERROR: {e}")
    print("💡 Try restarting your runtime: Runtime → Restart Runtime")
```

## Python Version Compatibility

- **Nabla-ML requires Python 3.10+**
- **Google Colab typically runs Python 3.10**
- If you see version errors, the package has been updated to support Colab!

## Troubleshooting

### "No matching distribution found"
This error occurs when:
1. Your Python version is too old (< 3.10)
2. There's a temporary PyPI sync issue

**Solution:**
```python
# Check your Python version
import sys
print(f"Python version: {sys.version_info}")

# If Python < 3.10, try updating Colab (rarely needed)
# Or use a specific version:
!pip install nabla-ml==25.06121330
```

### Import Errors After Installation
**Solution:**
1. Restart your runtime: `Runtime → Restart Runtime`
2. Re-run the installation cell
3. Try importing again

### Want the Latest Features?
```python
# Install the very latest version
!pip install nabla-ml --upgrade --no-cache-dir
```

## Ready to Go!

Once installed, you can use Nabla just like in the tutorials:

```python
import nabla as nb
import numpy as np

# Create arrays
x = nb.Array.from_numpy(np.array([1.0, 2.0, 3.0]))
y = nb.sin(x)
print(f"sin([1, 2, 3]) = {y.to_numpy()}")
```
