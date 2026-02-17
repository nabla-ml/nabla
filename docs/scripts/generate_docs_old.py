
import ast
import inspect
import re
import shutil
from pathlib import Path
from collections import defaultdict

def parse_numpydoc(docstring):
    """
    Parses a numpydoc-style docstring into a structured dictionary.
    """
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

def get_function_signature(node, function_name):
    """
    Extracts a formatted function signature from an AST node.
    """
    signature = f"nabla.{function_name}"
    
    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            annotation_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else 'Any'
            arg_str += f": '{annotation_str}'"
        args.append(arg_str)
    
    signature += f"({', '.join(args)})"
    
    if node.returns:
        return_str = ast.unparse(node.returns) if hasattr(ast, 'unparse') else 'Any'
        signature += f" -> '{return_str}'"
            
    return signature

def generate_markdown(function_name, signature, parsed_docstring):
    """
    Generates a markdown string for a single function.
    """
    md = [f"# {function_name}\n"]
    
    md.append("## Signature\n")
    md.append(f"```python\n{signature}\n```\n")

    if parsed_docstring.get('description'):
        md.append("## Description\n")
        md.append(f"{parsed_docstring['description']}\n")

    if parsed_docstring.get('parameters'):
        md.append("## Parameters\n")
        for param in parsed_docstring['parameters']:
            md.append(f"- **`{param['name']}`** (`{param['type']}`): {param['description']}\n")

    if parsed_docstring.get('returns'):
        md.append("## Returns\n")
        for ret in parsed_docstring['returns']:
            md.append(f"- `{ret['type']}`: {ret['description']}\n")

    if parsed_docstring.get('examples'):
        md.append("## Examples\n")
        md.append(f"```pycon\n{parsed_docstring['examples']}\n```\n")

    return '\n'.join(md)

def discover_public_api(module_dir, prefix=''):
    """
    Recursively parses __all__ from modules to find the public API.
    For modules without __all__, extracts public imports from __init__.py.
    Returns a dictionary mapping function names to their source module path.
    """
    public_api_map = {}
    
    # First check if __init__.py has __all__
    init_file = module_dir / '__init__.py'
    has_all = False
    
    if init_file.exists():
        with open(init_file, 'r', encoding='utf-8') as f:
            init_content = f.read()
        
        try:
            tree = ast.parse(init_content)
            # Check for __all__
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            has_all = True
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        # Map to the prefix (the module itself)
                                        public_api_map[elt.value] = prefix if prefix else 'root'
                            break
            
            # If no __all__, extract from imports (for transforms, core, etc.)
            if not has_all and prefix:  # Only for submodules
                for node in tree.body:
                    # Handle: from .module import name
                    if isinstance(node, ast.ImportFrom):
                        if node.module and not node.module.startswith('_'):
                            for alias in node.names:
                                if not alias.name.startswith('_'):
                                    # Map to source module
                                    source_module = f"{prefix}/{node.module.lstrip('.')}"
                                    public_api_map[alias.asname or alias.name] = source_module
        except SyntaxError:
            pass
    
    # Process individual .py files for __all__
    for py_file in module_dir.glob('*.py'):
        if py_file.name.startswith('__'):
            continue
        
        module_name = py_file.stem
        full_module_path = f"{prefix}/{module_name}" if prefix else module_name
        
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        public_api_map[elt.value] = full_module_path
                            break
        except SyntaxError:
            continue
    
    # Recursively process subdirectories
    for subdir in module_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('__') and not subdir.name.startswith('.'):
            sub_prefix = f"{prefix}/{subdir.name}" if prefix else subdir.name
            sub_api = discover_public_api(subdir, sub_prefix)
            public_api_map.update(sub_api)
            
    return public_api_map

def main():
    """
    Main function to generate hierarchical API docs based on public API.
    """
    project_root = Path(__file__).parent.parent.parent
    nabla_dir = project_root / 'nabla'
    api_dir = project_root / 'docs' / 'api'
    
    # Discover all modules to document
    modules_to_document = ['ops', 'nn', 'transforms', 'core', 'utils']
    
    print("Discovering public API from `__all__` variables...")
    all_public_api = {}
    
    for module_name in modules_to_document:
        module_path = nabla_dir / module_name
        if not module_path.exists():
            print(f"  - Skipping {module_name} (not found)")
            continue
        print(f"  - Scanning {module_name}/")
        module_api = discover_public_api(module_path, module_name)
        all_public_api.update(module_api)
    
    print(f"\nFound {len(all_public_api)} public functions/classes.")

    functions_by_category = defaultdict(list)
    for func, cat in all_public_api.items():
        functions_by_category[cat].append(func)

    print(f"Cleaning up old API files in: {api_dir}")
    if api_dir.exists():
        for item in api_dir.iterdir():
            if item.name not in ['index.rst', 'index.md', 'scripts']:
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.suffix != '.rst':
                    item.unlink()
    
    api_dir.mkdir(exist_ok=True)

    print("Generating documentation...")
    for category_name, functions in functions_by_category.items():
        # Keep the exact folder structure (don't replace / with _)
        category_dir = api_dir / category_name
        category_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing category: {category_name}")

        # Determine source file path - check both .py file and __init__.py
        source_file = nabla_dir / f"{category_name}.py"
        if not source_file.exists():
            # Try __init__.py for package directories
            source_file_init = nabla_dir / category_name / "__init__.py"
            if source_file_init.exists():
                source_file = source_file_init
            else:
                print(f"  - WARNING: Source file for '{category_name}' not found. Skipping category.")
                continue
            
        with open(source_file, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        function_nodes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        class_nodes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        all_nodes = {**function_nodes, **class_nodes}

        for function_name in sorted(functions):
            if function_name not in all_nodes:
                print(f"  - WARNING: Public item '{function_name}' not found in '{source_file.name}'. Skipping.")
                continue

            node = all_nodes[function_name]
            docstring = ast.get_docstring(node)
            
            if not docstring:
                print(f"  - WARNING: Public item '{function_name}' has no docstring. Skipping.")
                continue

            print(f"  - Generating docs for: {function_name}")
            
            parsed_docstring = parse_numpydoc(docstring)
            
            # Use different signature format for classes vs functions
            if isinstance(node, ast.ClassDef):
                signature = f"nabla.{function_name}"
            else:
                signature = get_function_signature(node, function_name)
            
            markdown_content = generate_markdown(function_name, signature, parsed_docstring)
            
            output_file = category_dir / f"{function_name}.md"
            output_file.write_text(markdown_content, encoding='utf-8')

        # Generate index for the category as an .rst file
        category_index_rst = [f"{category_name.replace('_', ' ').title()}"]
        category_index_rst.append("=" * len(category_index_rst[0]))
        category_index_rst.append("\n.. toctree::")
        category_index_rst.append("   :maxdepth: 1")
        category_index_rst.append("   :caption: Functions\n")
        for name in sorted(functions):
            category_index_rst.append(f"   {name}.md")
        
        category_index_file = category_dir / "index.rst"
        category_index_file.write_text('\n'.join(category_index_rst), encoding='utf-8')

    # Generate main API index as an .rst file
    print("\nGenerating main API index...")
    main_index_rst = ["API Reference"]
    main_index_rst.append("=============\n")
    main_index_rst.append("This page contains the complete API reference for Nabla, organized by functionality.\n")
    main_index_rst.append(".. toctree::")
    main_index_rst.append("   :maxdepth: 2")
    main_index_rst.append("   :caption: API Documentation\n")
    for category in sorted(functions_by_category.keys()):
        # Use the exact path structure
        main_index_rst.append(f"   {category}/index")
    
    main_index_file = api_dir / "index.rst"
    main_index_file.write_text('\n'.join(main_index_rst), encoding='utf-8')

    print("\nDone. Hierarchical documentation generated in:", api_dir)

if __name__ == "__main__":
    main()
