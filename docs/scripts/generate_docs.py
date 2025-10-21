
import inspect
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Add nabla to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nabla
import nabla.nn


def generate_markdown(name, obj, module_path, is_nn=False):
    """Generates markdown documentation for a function or class."""
    md = [f"# {name}\n"]
    
    # Get docstring
    docstring = inspect.getdoc(obj)
    if not docstring:
        docstring = f"No documentation available for `{name}`."
    
    # Generate signature with proper prefix
    prefix = "nabla.nn." if is_nn else "nabla."
    
    md.append("## Signature\n")
    if inspect.isclass(obj):
        md.append(f"```python\n{prefix}{name}\n```\n")
    elif inspect.isfunction(obj):
        try:
            sig = inspect.signature(obj)
            md.append(f"```python\n{prefix}{name}{sig}\n```\n")
        except:
            md.append(f"```python\n{prefix}{name}(...)\n```\n")
    else:
        md.append(f"```python\n{prefix}{name}\n```\n")
    
    # Module path
    md.append(f"**Source**: `{module_path}`\n")
    
    # Add the entire docstring as-is (assumes docstring is already in markdown format)
    md.append(docstring)
    md.append("\n")
    
    return '\n'.join(md)


def main():
    """Generate documentation for all public API items."""
    project_root = Path(__file__).parent.parent.parent
    api_dir = project_root / 'docs' / 'api'
    
    print("Discovering public API from nabla.__all__ and nabla.nn.__all__...")
    
    # Clean up old docs
    if api_dir.exists():
        for item in api_dir.iterdir():
            if item.name not in ['index.rst']:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    
    api_dir.mkdir(exist_ok=True)
    
    # Organize items by category
    categories = defaultdict(list)
    
    # Get all public items from nabla (main module)
    all_items = nabla.__all__
    print(f"Found {len(all_items)} public items in nabla.__all__")
    
    for item_name in sorted(all_items):
        try:
            # Get the actual object
            obj = getattr(nabla, item_name)
            
            # Determine the source module
            module_name = getattr(obj, '__module__', 'unknown')
            
            # Simplify module path for categorization
            # Only use the first level after 'nabla.' (e.g., 'core', 'transforms', 'ops')
            if module_name.startswith('nabla.'):
                parts = module_name.replace('nabla.', '').split('.')
                category = parts[0] if parts else 'other'
            else:
                category = 'other'
            
            # Skip 'other' category - not interesting
            if category == 'other':
                continue
            
            categories[category].append((item_name, obj, module_name))
            
        except Exception as e:
            print(f"  - WARNING: Could not process '{item_name}': {e}")
            continue
    
    # Get all public items from nabla.nn
    nn_items = nabla.nn.__all__
    print(f"Found {len(nn_items)} public items in nabla.nn.__all__")
    
    # Skip submodule names - they're not documentable items
    submodule_names = {'losses', 'optim', 'init', 'layers', 'architectures', 'utils'}
    
    for item_name in sorted(nn_items):
        # Skip submodule names
        if item_name in submodule_names:
            print(f"  - Skipping submodule name: {item_name}")
            continue
            
        try:
            # Get the actual object from nabla.nn
            obj = getattr(nabla.nn, item_name)
            
            # Determine the source module - this tells us where the item is ACTUALLY defined
            module_name = getattr(obj, '__module__', 'unknown')
            
            # Map the source module to documentation category
            # The category should match the actual file structure in nabla/nn/
            
            if module_name.startswith('nabla.nn.'):
                # Remove 'nabla.nn.' prefix to get relative path
                relative_path = module_name.replace('nabla.nn.', '')
                parts = relative_path.split('.')
                
                if parts[0] == 'functional':
                    # Items from nabla/nn/functional/* subdirectories
                    # e.g., nabla.nn.functional.losses.regression -> nn/functional/losses
                    if len(parts) >= 2:
                        category = f'nn/functional/{parts[1]}'
                    else:
                        category = 'nn/functional'
                elif parts[0] == 'modules':
                    # Items from nabla/nn/modules/*
                    category = 'nn/modules'
                elif parts[0] == 'module':
                    # Items from module.py -> nn/module
                    category = 'nn/module'
                elif parts[0] == 'containers':
                    # Items from containers.py -> nn/containers
                    category = 'nn/containers'
                elif parts[0] == 'optim':
                    # Items from optim.py -> nn/optim
                    category = 'nn/optim'
                else:
                    # Default fallback
                    category = 'nn/other'
            else:
                # For JIT-wrapped functions (show as nabla.transforms.jit), 
                # we need to infer the category from context
                # Check if it's imported from functional submodules based on name patterns
                if any(x in item_name.lower() for x in ['loss', 'cross_entropy', 'mse', 'mae']):
                    category = 'nn/functional/losses'
                elif any(x in item_name.lower() for x in ['accuracy', 'precision', 'recall', 'f1', 'metric', 'dropout', 'regularization', 'dataset', 'gradient_clipping']):
                    category = 'nn/functional/utils'
                elif any(x in item_name.lower() for x in ['step', 'schedule', 'init_adam', 'init_sgd']):
                    category = 'nn/functional/optim'
                elif any(x in item_name.lower() for x in ['relu', 'sigmoid', 'tanh', 'gelu', 'softmax', 'forward', 'activation']):
                    category = 'nn/functional/layers'
                elif any(x in item_name.lower() for x in ['he_', 'xavier_', 'lecun_', 'initialize']):
                    category = 'nn/functional/init'
                elif any(x in item_name.lower() for x in ['mlp', 'builder', 'config']):
                    category = 'nn/functional/architectures'
                else:
                    # Default fallback - put in modules
                    category = 'nn/modules'
            
            categories[category].append((item_name, obj, module_name))
            
        except Exception as e:
            print(f"  - WARNING: Could not process 'nn.{item_name}': {e}")
            continue
    
    print(f"\nGenerating documentation for {len(categories)} categories...")
    
    # Generate docs for each category
    for category, items in sorted(categories.items()):
        category_dir = api_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {category} ({len(items)} items)")
        
        is_nn_category = category.startswith('nn')
        
        for item_name, obj, module_name in items:
            try:
                markdown_content = generate_markdown(item_name, obj, module_name, is_nn=is_nn_category)
                output_file = category_dir / f"{item_name}.md"
                output_file.write_text(markdown_content, encoding='utf-8')
                print(f"  ✓ {item_name}")
            except Exception as e:
                print(f"  ✗ {item_name}: {e}")
        
        # Generate category index
        category_index_rst = [category.replace('/', ' / ').title()]
        category_index_rst.append("=" * len(category_index_rst[0]))
        category_index_rst.append("\n.. toctree::")
        category_index_rst.append("   :maxdepth: 1\n")
        for item_name, _, _ in sorted(items):
            category_index_rst.append(f"   {item_name}.md")
        
        category_index_file = category_dir / "index.rst"
        category_index_file.write_text('\n'.join(category_index_rst), encoding='utf-8')
    
    # Generate main index
    print("\nGenerating main API index...")
    main_index_rst = ["API Reference"]
    main_index_rst.append("=" * len(main_index_rst[0]))
    main_index_rst.append("\nComplete API reference for Nabla.\n")
    main_index_rst.append(".. toctree::")
    main_index_rst.append("   :maxdepth: 2\n")
    for category in sorted(categories.keys()):
        main_index_rst.append(f"   {category}/index")
    
    main_index_file = api_dir / "index.rst"
    main_index_file.write_text('\n'.join(main_index_rst), encoding='utf-8')
    
    print(f"\n✅ Done! Generated docs for {sum(len(items) for items in categories.values())} items")
    print(f"   Output: {api_dir}")


if __name__ == "__main__":
    main()
