
import ast
import inspect
import re
import shutil
import importlib
import sys
from pathlib import Path
from collections import defaultdict

# Add nabla to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nabla

def parse_numpydoc(docstring):
    """Parses a numpydoc-style docstring into a structured dictionary."""
    if not docstring:
        return {}

    docstring = inspect.cleandoc(docstring)
    
    sections = re.split(r'\n\s*(Parameters|Returns|Examples)\s*\n\s*-+\s*\n', docstring)
    
    parsed = {'description': sections[0].strip()}
    
    for i in range(1, len(sections), 2):
        section_name = sections[i].lower()
        section_content = sections[i+1].strip()
        
        if section_name == 'examples':
            parsed['examples'] = section_content
            parsed['examples'] = re.sub(r'<BLANKLINE>\n', '\n', parsed['examples'])
            parsed['examples'] = re.sub(r'^\.\.\. ', '', parsed['examples'], flags=re.MULTILINE)
            continue
            
        items = []
        item_blocks = re.split(r'\n(?=\S)', section_content)
        
        for block in item_blocks:
            if not block.strip():
                continue
            
            lines = block.strip().split('\n')
            first_line = lines[0]
            description = ' '.join(l.strip() for l in lines[1:])
            
            name, type_hint = '', ''
            parts = first_line.split(':', 1)
            if len(parts) == 2:
                name = parts[0].strip()
                type_hint = parts[1].strip()
            else:
                type_hint = parts[0].strip()
            
            items.append({'name': name, 'type': type_hint, 'description': description})
        
        parsed[section_name] = items

    return parsed


def generate_markdown(name, obj, module_path):
    """Generates markdown documentation for a function or class."""
    md = [f"# {name}\n"]
    
    # Get docstring
    docstring = inspect.getdoc(obj)
    if not docstring:
        docstring = f"No documentation available for `{name}`."
    
    parsed_docstring = parse_numpydoc(docstring)
    
    # Generate signature
    md.append("## Signature\n")
    if inspect.isclass(obj):
        md.append(f"```python\nnabla.{name}\n```\n")
    elif inspect.isfunction(obj):
        try:
            sig = inspect.signature(obj)
            md.append(f"```python\nnabla.{name}{sig}\n```\n")
        except:
            md.append(f"```python\nnabla.{name}(...)\n```\n")
    else:
        md.append(f"```python\nnabla.{name}\n```\n")
    
    # Module path
    md.append(f"**Source**: `{module_path}`\n")
    
    # Description
    if parsed_docstring.get('description'):
        md.append("## Description\n")
        md.append(f"{parsed_docstring['description']}\n")
    
    # Parameters
    if parsed_docstring.get('parameters'):
        md.append("## Parameters\n")
        for param in parsed_docstring['parameters']:
            md.append(f"- **`{param['name']}`** (`{param['type']}`): {param['description']}\n")
    
    # Returns
    if parsed_docstring.get('returns'):
        md.append("## Returns\n")
        for ret in parsed_docstring['returns']:
            md.append(f"- `{ret['type']}`: {ret['description']}\n")
    
    # Examples
    if parsed_docstring.get('examples'):
        md.append("## Examples\n")
        md.append(f"```pycon\n{parsed_docstring['examples']}\n```\n")
    
    return '\n'.join(md)


def main():
    """Generate documentation for all public API items."""
    project_root = Path(__file__).parent.parent.parent
    api_dir = project_root / 'docs' / 'api'
    
    print("Discovering public API from nabla.__all__...")
    
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
    
    # Get all public items from nabla
    all_items = nabla.__all__
    print(f"Found {len(all_items)} public items in nabla.__all__")
    
    for item_name in sorted(all_items):
        try:
            # Get the actual object
            obj = getattr(nabla, item_name)
            
            # Determine the source module
            module_name = getattr(obj, '__module__', 'unknown')
            
            # Simplify module path for categorization
            if module_name.startswith('nabla.'):
                category = module_name.replace('nabla.', '').replace('.', '/')
            else:
                category = 'other'
            
            categories[category].append((item_name, obj, module_name))
            
        except Exception as e:
            print(f"  - WARNING: Could not process '{item_name}': {e}")
            continue
    
    print(f"\nGenerating documentation for {len(categories)} categories...")
    
    # Generate docs for each category
    for category, items in sorted(categories.items()):
        category_dir = api_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {category} ({len(items)} items)")
        
        for item_name, obj, module_name in items:
            try:
                markdown_content = generate_markdown(item_name, obj, module_name)
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
