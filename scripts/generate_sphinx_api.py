#!/usr/bin/env python3
"""
Generate API documentation using Sphinx autodoc and autosummary.

This script creates .rst files that use Sphinx's autodoc directives to automatically
generate documentation from docstrings, eliminating the need for manual .md creation.
"""

import os
import sys
from pathlib import Path

def create_api_rst_files():
    """Create .rst files for automatic API documentation."""
    
    # Get the nabla package directory
    nabla_root = Path(__file__).parent.parent
    docs_api_dir = nabla_root / "docs" / "api"
    
    # Create api directory if it doesn't exist
    docs_api_dir.mkdir(exist_ok=True)
    
    # Main API index file
    api_index_content = """
API Reference 
=============

This section contains the complete API reference for Nabla, automatically generated from docstrings.

.. currentmodule:: nabla

Core Modules
------------

.. autosummary::
   :toctree: generated/
   :recursive:
   
   nabla.core
   nabla.ops
   nabla.transforms
   nabla.nn
   nabla.utils

Array Operations
---------------

.. automodule:: nabla.core.array
   :members:
   :undoc-members:
   :show-inheritance:

Binary Operations  
-----------------

.. automodule:: nabla.ops.binary
   :members:
   :undoc-members:
   :show-inheritance:

Neural Network Layers
---------------------

.. automodule:: nabla.nn.layers
   :members:
   :undoc-members:
   :show-inheritance:

Function Transformations
------------------------

.. automodule:: nabla.transforms
   :members:
   :undoc-members:
   :show-inheritance:

Creation Functions
------------------

.. automodule:: nabla.ops.creation
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: nabla.utils
   :members:
   :undoc-members:
   :show-inheritance:
"""
    
    # Write the main API index
    with open(docs_api_dir / "index.rst", "w") as f:
        f.write(api_index_content.strip())
    
    print(f"✅ Created API documentation index at {docs_api_dir / 'index.rst'}")
    
    # Create individual module files for better organization
    modules = [
        ("nabla.core", "Core Array Operations"),
        ("nabla.ops", "Operations"),
        ("nabla.transforms", "Function Transformations"),
        ("nabla.nn", "Neural Network Components"),
        ("nabla.utils", "Utilities"),
    ]
    
    for module_name, title in modules:
        module_content = f"""
{title}
{'=' * len(title)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :recursive:
"""
        
        filename = module_name.replace("nabla.", "") + ".rst"
        with open(docs_api_dir / filename, "w") as f:
            f.write(module_content.strip())
        
        print(f"✅ Created {filename}")

def update_main_index():
    """Update the main docs index to include API reference."""
    
    docs_dir = Path(__file__).parent.parent / "docs"
    index_file = docs_dir / "index.md"
    
    # Check if API reference is already in the index
    if index_file.exists():
        content = index_file.read_text()
        if "api/index" not in content:
            # Add API reference to toctree
            if "```{toctree}" in content:
                content = content.replace(
                    "```{toctree}",
                    "```{toctree}\n:maxdepth: 2\n\napi/index"
                )
                index_file.write_text(content)
                print("✅ Added API reference to main index")
            else:
                print("⚠️  Could not find toctree in index.md - you may need to add api/index manually")

if __name__ == "__main__":
    print("🚀 Generating Sphinx API documentation...")
    create_api_rst_files()
    update_main_index()
    print("\n✨ Done! Now run 'make html' in the docs directory to build the documentation.")
    print("📖 The API docs will be automatically generated from your docstrings!")
