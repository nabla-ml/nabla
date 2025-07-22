# Documentation Guide

This guide explains how to generate and maintain the Nabla documentation.

## Quick Start

```bash
# Generate structured API documentation
python docs/scripts/generate_api_docs.py

# Build the documentation
cd docs && bash build.sh

# View the documentation
open docs/_build/html/index.html
```

## Documentation Scripts

Documentation maintenance scripts are located in `docs/scripts/`:

- `generate_api_docs.py` - Generate API documentation from code
- `generate_sitemap.py` - Create sitemap.xml for SEO
- `check_indexing.py` - Verify Google indexing status
- `seo_audit.py` - Run SEO analysis on documentation
- `fix_seo_issues.py` - Fix common SEO problems
- `final_seo_report.py` - Generate comprehensive SEO report

## Documentation Structure

The documentation automatically mirrors the Nabla library structure:

```
docs/api/
├── core/          # nabla.core.*
├── ops/           # nabla.ops.*
├── transforms/    # nabla.transforms.*
├── nn/            # nabla.nn.*
│   ├── layers/    # nabla.nn.layers.*
│   ├── losses/    # nabla.nn.losses.*
│   └── ...
└── utils/         # nabla.utils.*
```

## Excluding Functions/Classes from Documentation

### Method 1: Using `@nodoc` Decorator (Recommended)

```python
from nabla.utils.docs import nodoc

# Exclude a function
@nodoc
def internal_helper():
    """This won't appear in docs."""
    pass

# Exclude a class
@nodoc
class InternalUtility:
    """This class won't appear in docs."""
    pass

# Exclude methods within a class
class PublicClass:
    def public_method(self):
        """This will appear in docs."""
        pass
    
    @nodoc
    def _internal_method(self):
        """This won't appear in docs."""
        pass
```

### Method 2: Using `__all__` Lists

Define what should be documented in each module:

```python
# In your module
__all__ = ['PublicClass', 'public_function']  # Only these will be documented

class PublicClass:
    """This will be documented."""
    pass

class InternalClass:
    """This won't be documented (not in __all__)."""
    pass
```

### Method 3: Naming Convention

Functions/classes starting with `_` are automatically excluded:

```python
def public_function():
    """This will be documented."""
    pass

def _private_function():
    """This won't be documented (starts with _)."""
    pass
```

## Available Commands

### Generate Documentation Structure
```bash
python scripts/generate_structured_docs.py
```
Creates the complete API documentation structure that mirrors your library organization.

### Build Documentation
```bash
cd docs
make html          # Build HTML documentation
make clean         # Clean build artifacts
make clean && make html  # Clean rebuild
```

### Fix Documentation Warnings
```bash
python scripts/fix_doc_warnings.py
```
Automatically fixes common documentation warnings and formatting issues.

### Development Server
```bash
cd docs/_build/html
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

## Configuration

### Sphinx Settings
Key settings in `docs/conf.py`:
- **autodoc**: Automatically generates docs from docstrings
- **autosummary**: Creates module summaries
- **napoleon**: Supports Google/NumPy style docstrings
- **nodoc integration**: Respects `@nodoc` decorator

### Custom Skip Function
The documentation system automatically excludes:
- Items marked with `@nodoc`
- Private items (starting with `_`)
- Items not in `__all__` lists
- Common internal attributes

## Best Practices

### 1. Use Clear Docstrings
```python
def my_function(x, y):
    """
    Brief description of what the function does.
    
    Args:
        x: Description of parameter x
        y: Description of parameter y
        
    Returns:
        Description of return value
        
    Example:
        >>> result = my_function(1, 2)
        >>> print(result)
        3
    """
    return x + y
```

### 2. Organize with `@nodoc`
```python
# Public API
def create_array(shape):
    """Create a new array with given shape."""
    return _internal_create_array(shape)

# Internal implementation
@nodoc
def _internal_create_array(shape):
    """Internal array creation logic."""
    # Implementation details
    pass
```

### 3. Use `__all__` for Module Organization
```python
# At the top of your module
__all__ = [
    'PublicClass',
    'public_function',
    'CONSTANT'
]

# Only items in __all__ will be documented
```

## Troubleshooting

### Common Issues

**ImportError during build:**
- Check that all imports work correctly
- Ensure circular imports are resolved
- Verify `@nodoc` imports are correct

**Missing documentation:**
- Check if item is in `__all__` list
- Verify item doesn't start with `_`
- Ensure `@nodoc` decorator isn't applied

**Docstring formatting warnings:**
- Use consistent indentation (4 spaces)
- Follow Google/NumPy docstring format
- Run `python scripts/fix_doc_warnings.py`

### Build Errors
```bash
# Clean rebuild to fix caching issues
cd docs && make clean && make html

# Check specific errors
cd docs && sphinx-build -b html . _build/html
```

## Files Overview

| File | Purpose |
|------|---------|
| `scripts/generate_structured_docs.py` | Creates API documentation structure |
| `scripts/fix_doc_warnings.py` | Fixes common documentation warnings |
| `docs/conf.py` | Sphinx configuration |
| `nabla/utils/docs.py` | Contains `@nodoc` decorator |
| `docs/Makefile` | Build commands |

## Example Workflow

1. **Mark internal code:**
   ```python
   @nodoc
   def internal_function():
       pass
   ```

2. **Generate documentation:**
   ```bash
   python scripts/generate_structured_docs.py
   ```

3. **Build and view:**
   ```bash
   cd docs && make html && open _build/html/index.html
   ```

4. **Fix any warnings:**
   ```bash
   python scripts/fix_doc_warnings.py
   ```

That's it! Your documentation will automatically stay in sync with your code structure and respect your public/private API boundaries.

## SEO & Discoverability

The documentation includes **comprehensive SEO optimization**:

### Automatic SEO Features

- ✅ Page-specific meta tags and descriptions
- ✅ Open Graph tags for social sharing  
- ✅ Twitter Cards support
- ✅ Automatic sitemap generation
- ✅ Robots.txt configuration
- ✅ Mobile-responsive design
- ✅ Clean URL structure

### SEO Best Practices for API Docs

```python
def your_function(param):
    """
    Brief, descriptive summary for search engines.
    
    Detailed explanation with keywords like 'GPU-accelerated',
    'NumPy-compatible', 'machine learning', etc.
    
    Args:
        param: Clear parameter description
        
    Returns:
        Clear return value description
        
    Example:
        >>> import nabla as nb
        >>> result = your_function(value)
    """
```

### Module-Level SEO

Add SEO-friendly module docstrings:

```python
# At the top of each module
"""
Core array operations for GPU-accelerated computation.

This module provides NumPy-compatible array operations optimized
for machine learning and scientific computing workloads.
"""
```

## Performance & Analytics

### Build Performance

```bash
# Monitor build time
time make html

# Check for SEO issues
python scripts/fix_doc_warnings.py
```

### SEO Analysis

```bash
# View current SEO configuration
cat docs/conf.py | grep -A 10 "html_meta"

# Check sitemap generation
ls docs/_build/html/sitemap.xml
```
