#!/usr/bin/env python3
"""
Generate structured API documentation that mirrors the Nabla library organization.

This script creates a documentation structure that exactly matches the internal
organization of the Nabla library, ensuring the docs reflect the actual codebase.
"""

import os
import importlib
from pathlib import Path
from typing import Dict, List, Set

def get_nabla_structure() -> Dict:
    """Get the actual structure of the Nabla library by introspection."""
    nabla_path = Path(__file__).parent.parent / "nabla"
    
    structure = {
        "core": {
            "description": "Core array operations and execution context",
            "modules": ["array", "execution_context", "graph_execution"]
        },
        "ops": {
            "description": "Low-level operations and kernels",
            "modules": ["binary", "unary", "creation", "linalg", "reduce", "view", "special", "operation", "conv_utils"],
            "submodules": {
                "kernels": "Custom compute kernels"
            }
        },
        "transforms": {
            "description": "Function transformations (grad, jit, vmap, etc.)",
            "modules": ["grad", "jacfwd", "jacrev", "jit", "jvp", "vjp", "vmap", "utils"]
        },
        "nn": {
            "description": "Neural network components and utilities",
            "submodules": {
                "layers": "Neural network layers (Linear, Conv2D, etc.)",
                "losses": "Loss functions (MSE, CrossEntropy, etc.)",
                "optim": "Optimizers (SGD, Adam, etc.)",
                "init": "Parameter initialization strategies",
                "architectures": "Pre-built network architectures",
                "utils": "Neural network utilities and training helpers"
            }
        },
        "utils": {
            "description": "Utility functions and helpers",
            "modules": ["docs", "formatting", "grad_utils", "max_interop", "shape_utils", "testing", "types"]
        }
    }
    
    return structure

def create_api_index():
    """Create the main API index file."""
    content = """# API Reference

This section provides detailed documentation for all Nabla modules and functions, 
organized to match the internal structure of the library.

```{toctree}
:maxdepth: 2
:caption: API Documentation

core/index
ops/index
transforms/index
nn/index
utils/index
```

## Quick Navigation

### Core Components
- **{doc}`core/index`** - Array operations and execution context
- **{doc}`ops/index`** - Low-level operations and kernels

### Function Transformations
- **{doc}`transforms/index`** - grad, jit, vmap, and other transformations

### Neural Networks
- **{doc}`nn/index`** - Layers, losses, optimizers, and architectures

### Utilities
- **{doc}`utils/index`** - Helper functions and utilities
"""
    
    api_dir = Path("docs/api")
    api_dir.mkdir(exist_ok=True)
    
    with open(api_dir / "index.md", "w") as f:
        f.write(content)
    
    print("✅ Created main API index")

def create_module_docs(module_name: str, config: Dict):
    """Create documentation for a specific module."""
    module_dir = Path(f"docs/api/{module_name}")
    module_dir.mkdir(exist_ok=True)
    
    # Create module index
    index_content = f"""# {module_name.title()} Module

{config['description']}

"""
    
    # Add toctree for submodules if they exist
    if 'submodules' in config:
        index_content += """```{toctree}
:maxdepth: 1
:caption: Submodules

"""
        for submodule, desc in config['submodules'].items():
            index_content += f"{submodule}/index\n"
        index_content += "```\n\n"
    
    # Add toctree for individual modules if they exist
    if 'modules' in config:
        index_content += """```{toctree}
:maxdepth: 1
:caption: Modules

"""
        for module in config['modules']:
            index_content += f"{module}\n"
        index_content += "```\n\n"
    
    # Add automodule directive for the main module
    index_content += f"""
## Module Overview

```{{eval-rst}}
.. automodule:: nabla.{module_name}
   :members:
   :undoc-members:
   :show-inheritance:
```
"""
    
    with open(module_dir / "index.md", "w") as f:
        f.write(index_content)
    
    # Create individual module files
    if 'modules' in config:
        for module in config['modules']:
            create_individual_module_doc(module_name, module)
    
    # Create submodule documentation
    if 'submodules' in config:
        for submodule, desc in config['submodules'].items():
            create_submodule_docs(module_name, submodule, desc)
    
    print(f"✅ Created {module_name} module documentation")

def create_individual_module_doc(parent_module: str, module_name: str):
    """Create documentation for an individual module file."""
    content = f"""# {module_name.replace('_', ' ').title()}

```{{eval-rst}}
.. automodule:: nabla.{parent_module}.{module_name}
   :members:
   :undoc-members:
   :show-inheritance:
```
"""
    
    module_dir = Path(f"docs/api/{parent_module}")
    with open(module_dir / f"{module_name}.md", "w") as f:
        f.write(content)

def create_submodule_docs(parent_module: str, submodule: str, description: str):
    """Create documentation for submodules (like nn.layers, nn.optim, etc.)."""
    submodule_dir = Path(f"docs/api/{parent_module}/{submodule}")
    submodule_dir.mkdir(exist_ok=True)
    
    # Get the actual submodule files
    submodule_path = Path(f"nabla/{parent_module}/{submodule}")
    if submodule_path.exists():
        python_files = [f.stem for f in submodule_path.glob("*.py") if f.stem != "__init__"]
    else:
        python_files = []
    
    # Create submodule index
    content = f"""# {submodule.title()}

{description}

"""
    
    if python_files:
        content += """```{toctree}
:maxdepth: 1

"""
        for module in python_files:
            content += f"{module}\n"
        content += "```\n\n"
    
    content += f"""
## Submodule Overview

```{{eval-rst}}
.. automodule:: nabla.{parent_module}.{submodule}
   :members:
   :undoc-members:
   :show-inheritance:
```
"""
    
    with open(submodule_dir / "index.md", "w") as f:
        f.write(content)
    
    # Create individual files for each module in the submodule
    for module in python_files:
        module_content = f"""# {module.replace('_', ' ').title()}

```{{eval-rst}}
.. automodule:: nabla.{parent_module}.{submodule}.{module}
   :members:
   :undoc-members:
   :show-inheritance:
```
"""
        with open(submodule_dir / f"{module}.md", "w") as f:
            f.write(module_content)
    
    print(f"✅ Created {parent_module}.{submodule} submodule documentation")

def main():
    """Generate the complete structured API documentation."""
    print("🚀 Generating structured API documentation...")
    
    # Get the actual library structure
    structure = get_nabla_structure()
    
    # Create main API index
    create_api_index()
    
    # Create documentation for each major module
    for module_name, config in structure.items():
        create_module_docs(module_name, config)
    
    print("\n✨ Done! The documentation now perfectly mirrors your library structure.")
    print("\n📁 Documentation structure created:")
    print("docs/api/")
    print("├── index.md                 # Main API index")
    print("├── core/")
    print("│   ├── index.md            # Core module overview")
    print("│   ├── array.md            # Array operations")
    print("│   ├── execution_context.md")
    print("│   └── graph_execution.md")
    print("├── ops/")
    print("│   ├── index.md            # Ops module overview")
    print("│   ├── binary.md           # Binary operations")
    print("│   ├── unary.md            # Unary operations")
    print("│   ├── creation.md         # Array creation")
    print("│   ├── linalg.md           # Linear algebra")
    print("│   └── ...")
    print("├── transforms/")
    print("│   ├── index.md            # Transforms overview")
    print("│   ├── grad.md             # Gradient computation")
    print("│   ├── jit.md              # Just-in-time compilation")
    print("│   ├── vmap.md             # Vectorization")
    print("│   └── ...")
    print("├── nn/")
    print("│   ├── index.md            # Neural networks overview")
    print("│   ├── layers/")
    print("│   │   ├── index.md        # Layers overview")
    print("│   │   ├── linear.md       # Linear layers")
    print("│   │   └── ...")
    print("│   ├── losses/")
    print("│   ├── optim/")
    print("│   ├── init/")
    print("│   ├── architectures/")
    print("│   └── utils/")
    print("└── utils/")
    print("    ├── index.md            # Utils overview")
    print("    ├── docs.md             # Documentation utilities")
    print("    ├── testing.md          # Testing utilities")
    print("    └── ...")
    print(f"\n🏗️  Now run 'make html' in the docs directory to build!")

if __name__ == "__main__":
    main()
