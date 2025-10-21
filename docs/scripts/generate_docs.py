
import inspect
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Add nabla to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nabla
import nabla.nn


def generate_markdown(name, obj, module_path, prefix='nabla.'):
    """Generates markdown documentation for a function or class."""
    md = [f"# {name}\n"]
    
    # Get docstring
    docstring = inspect.getdoc(obj)
    if not docstring:
        docstring = f"No documentation available for `{name}`."
    
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
    
    # Process docstring to properly format code examples
    # Convert doctest-style examples to proper code blocks
    docstring_lines = docstring.split('\n')
    processed_lines = []
    in_example = False
    example_buffer = []
    
    for line in docstring_lines:
        # Detect example sections (both "Examples" and "Example")
        if line.strip().startswith('Examples') or line.strip().startswith('Example'):
            if line.strip() in ['Examples', 'Example', 'Examples:', 'Example:', 'Examples\n-------', 'Example\n-------']:
                processed_lines.append(line)
                in_example = True
                continue
        
        # Check if we're at a new section (exits example mode)
        if in_example and line.strip() and not line.startswith(' ') and line.strip().endswith('---'):
            # This is a new section header
            if example_buffer:
                processed_lines.append('```python')
                processed_lines.extend(example_buffer)
                processed_lines.append('```')
                example_buffer = []
            in_example = False
            processed_lines.append(line)
            continue
        
        if in_example:
            stripped = line.strip()
            # Start collecting code when we see >>> or ...
            if stripped.startswith('>>>') or (stripped.startswith('...') and example_buffer):
                # Remove the >>> or ... prefix but keep indentation structure
                if stripped.startswith('>>>'):
                    code_line = stripped[4:]  # Remove '>>> '
                elif stripped.startswith('...'):
                    code_line = stripped[4:]  # Remove '... '
                example_buffer.append(code_line)
            elif example_buffer and stripped and not stripped.startswith('>>>'):
                # This might be output or a continuation, end the current code block
                processed_lines.append('```python')
                processed_lines.extend(example_buffer)
                processed_lines.append('```')
                processed_lines.append(line)
                example_buffer = []
            else:
                # Regular line in examples section (like "Usage as decorator:")
                if example_buffer:
                    processed_lines.append('```python')
                    processed_lines.extend(example_buffer)
                    processed_lines.append('```')
                    example_buffer = []
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    # Don't forget remaining buffered code
    if example_buffer:
        processed_lines.append('```python')
        processed_lines.extend(example_buffer)
        processed_lines.append('```')
    
    md.append('\n'.join(processed_lines))
    md.append("\n")
    
    return '\n'.join(md)


def main():
    """Generate documentation for all public API items."""
    project_root = Path(__file__).parent.parent.parent
    api_dir = project_root / 'docs' / 'api'
    
    print("Discovering public API from nabla modules...")
    
    # Clean up old docs
    if api_dir.exists():
        for item in api_dir.iterdir():
            if item.name not in ['index.rst']:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    
    api_dir.mkdir(exist_ok=True)
    
    # Organize items by their actual module path to preserve structure
    # Key: module path (e.g., 'nn/functional/losses'), Value: list of (item_name, obj, module_name, prefix)
    categories = defaultdict(list)
    
    # Define modules to process with their prefixes
    modules_to_process = [
        (nabla, nabla.__all__, 'nabla.'),     # Main nabla module
        (nabla.nn, nabla.nn.__all__, 'nabla.nn.'),  # Neural network module
    ]
    
    for module, all_items, default_prefix in modules_to_process:
        module_name_str = module.__name__
        print(f"Found {len(all_items)} public items in {module_name_str}.__all__")
        
        # Skip submodule names that are not actual objects
        submodule_names = {'losses', 'optim', 'init', 'layers', 'architectures', 'utils'}
        
        for item_name in sorted(all_items):
            # Skip submodule names
            if item_name in submodule_names:
                print(f"  - Skipping submodule name: {item_name}")
                continue
            
            try:
                # Get the actual object
                obj = getattr(module, item_name)
                
                # Determine the source module
                obj_module_name = getattr(obj, '__module__', 'unknown')
                
                # Build the category path based on the actual module structure
                # Example: nabla.nn.functional.losses.regression -> nn/functional/losses
                # Example: nabla.ops.binary -> ops
                # Example: nabla.transforms.grad -> transforms
                if obj_module_name.startswith('nabla.'):
                    # Remove 'nabla.' prefix
                    relative_path = obj_module_name.replace('nabla.', '')
                    parts = relative_path.split('.')
                    
                    # Build category based on actual module structure
                    if len(parts) == 1:
                        # Top-level module file (e.g., nabla.core.tensor)
                        category = parts[0]
                    elif len(parts) == 2:
                        # Module file in subdirectory (e.g., nabla.nn.module, nabla.ops.binary)
                        # Just use top-level (nn, ops, etc.)
                        category = parts[0]
                    else:
                        # Deeper structure (e.g., nabla.nn.functional.losses.regression)
                        # Preserve the structure: nn/functional/losses
                        category = '/'.join(parts[:-1])  # Remove the file name, keep directory structure
                else:
                    # Unknown module, skip
                    continue
                
                # Use the prefix from the module we're processing
                prefix = default_prefix
                
                # Store the item with its category (preserving module structure)
                categories[category].append((item_name, obj, obj_module_name, prefix))
                
            except Exception as e:
                print(f"  - WARNING: Could not process '{item_name}': {e}")
                continue
    
    print(f"\nGenerating documentation for {len(categories)} categories...")
    
    # Generate docs for each category
    for category, items in sorted(categories.items()):
        category_dir = api_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {category} ({len(items)} items)")
        
        for item_name, obj, module_name, prefix in items:
            try:
                markdown_content = generate_markdown(item_name, obj, module_name, prefix=prefix)
                output_file = category_dir / f"{item_name}.md"
                output_file.write_text(markdown_content, encoding='utf-8')
                print(f"  ✓ {item_name}")
            except Exception as e:
                print(f"  ✗ {item_name}: {e}")
        
        # Generate category index with a nice title
        category_title = category.replace('/', ' / ').replace('_', ' ').title()
        category_index_rst = [category_title]
        category_index_rst.append("=" * len(category_index_rst[0]))
        category_index_rst.append("\n.. toctree::")
        category_index_rst.append("   :maxdepth: 1\n")
        for item_name, _, _, _ in sorted(items):
            category_index_rst.append(f"   {item_name}.md")
        
        category_index_file = category_dir / "index.rst"
        category_index_file.write_text('\n'.join(category_index_rst), encoding='utf-8')
    
    # Build hierarchical index structure
    # Group categories by their parent directories to create proper hierarchy
    print("\nBuilding hierarchical index structure...")
    
    # Collect all unique directory paths
    all_dirs = set()
    for category in categories.keys():
        parts = category.split('/')
        for i in range(len(parts)):
            all_dirs.add('/'.join(parts[:i+1]))
    
    # Create index files for intermediate directories
    for dir_path in sorted(all_dirs):
        dir_parts = dir_path.split('/')
        dir_full_path = api_dir / dir_path
        
        # Skip if this is a leaf category (already has an index)
        if dir_path in categories:
            continue
        
        # Find all immediate children (both subdirs and leaf categories)
        children = []
        for other_dir in sorted(all_dirs):
            other_parts = other_dir.split('/')
            # Check if other_dir is an immediate child
            if len(other_parts) == len(dir_parts) + 1 and other_dir.startswith(dir_path + '/'):
                children.append(other_parts[-1])
        
        if children:
            dir_full_path.mkdir(parents=True, exist_ok=True)
            dir_title = dir_parts[-1].replace('_', ' ').title()
            index_rst = [dir_title]
            index_rst.append("=" * len(index_rst[0]))
            index_rst.append("\n.. toctree::")
            index_rst.append("   :maxdepth: 1\n")
            for child in sorted(children):
                index_rst.append(f"   {child}/index")
            
            index_file = dir_full_path / "index.rst"
            index_file.write_text('\n'.join(index_rst), encoding='utf-8')
            print(f"  Created index for {dir_path}/")
    
    # Generate main API index with top-level categories
    print("\nGenerating main API index...")
    top_level_categories = sorted(set(cat.split('/')[0] for cat in categories.keys()))
    
    main_index_rst = ["API Reference"]
    main_index_rst.append("=" * len(main_index_rst[0]))
    main_index_rst.append("\nComplete API reference for Nabla.\n")
    main_index_rst.append(".. toctree::")
    main_index_rst.append("   :maxdepth: 2\n")
    
    for top_category in top_level_categories:
        main_index_rst.append(f"   {top_category}/index")
    
    main_index_file = api_dir / "index.rst"
    main_index_file.write_text('\n'.join(main_index_rst), encoding='utf-8')
    
    print(f"\n✅ Done! Generated docs for {sum(len(items) for items in categories.values())} items")
    print(f"   Output: {api_dir}")


if __name__ == "__main__":
    main()
