
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
    
    # First, handle malformed docstrings where Returns/Examples are indented under Parameters
    # Extract these nested sections and move them to top level, removing extra indentation
    def dedent_section(match):
        section_name = match.group(2)
        # Find the full section content until next top-level section or end
        return f'\n{section_name}:'
    
    # Move nested sections to top-level
    docstring = re.sub(
        r'(\n    (Returns|Examples):)',
        dedent_section,
        docstring
    )
    
    # Also dedent the content of these extracted sections (remove 8 spaces of indent)
    lines = docstring.split('\n')
    new_lines = []
    in_extracted_section = False
    
    for i, line in enumerate(lines):
        # Check if this is a top-level section
        if re.match(r'^(Parameters|Returns|Examples):', line):
            in_extracted_section = line.startswith('Returns:') or line.startswith('Examples:')
            new_lines.append(line)
        elif in_extracted_section and line.startswith('        '):
            # Remove 8 spaces of indentation from extracted section content
            new_lines.append(line[8:])
        else:
            new_lines.append(line)
            # Stop dedenting if we hit a new section or unindented line
            if not line.startswith(' ') and line.strip():
                in_extracted_section = False
    
    docstring = '\n'.join(new_lines)
    
    # Split by section headers - handle both styles:
    # 1. Google-style: "Parameters:" with colon
    # 2. NumPy-style: "Parameters" followed by dashes on next line
    # Try NumPy-style first (with underline)
    numpy_sections = re.split(r'\n(Parameters|Returns|Examples)\s*\n-+\s*\n', docstring)
    
    # Try Google-style (with colon)
    google_sections = re.split(r'\n(Parameters|Returns|Examples):\s*\n', docstring)
    
    # Use whichever split found more sections (>1 means it found something)
    if len(numpy_sections) > len(google_sections):
        sections = numpy_sections
        style = 'numpy'
    else:
        sections = google_sections
        style = 'google'
    
    parsed = {'description': sections[0].strip()}
    
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
        section_name = sections[i].lower()
        section_content = sections[i+1].strip()
        
        if section_name == 'examples':
            # Process examples section to convert RST-style examples to markdown
            examples_content = section_content
            
            # Remove <BLANKLINE> markers
            examples_content = re.sub(r'<BLANKLINE>\n', '\n', examples_content)
            
            # Remove doctest continuation markers
            examples_content = re.sub(r'^\.\.\. ', '', examples_content, flags=re.MULTILINE)
            
            # Parse examples into text and code blocks
            parsed['examples'] = []
            
            # Detect if this is NumPy-style (with >>>) or Google-style (with ::)
            has_doctest = '>>>' in examples_content
            has_rst_blocks = '::' in examples_content
            
            if has_doctest and not has_rst_blocks:
                # NumPy-style: Parse doctest format
                lines = examples_content.split('\n')
                i = 0
                current_description = []
                current_code = []
                
                while i < len(lines):
                    line = lines[i]
                    
                    # Check if this is a doctest line (starts with >>>)
                    if line.lstrip().startswith('>>>'):
                        # Collect all consecutive doctest lines
                        while i < len(lines) and (lines[i].lstrip().startswith('>>>') or 
                                                  (current_code and lines[i].strip() and not lines[i].lstrip().startswith('>>>'))):
                            code_line = lines[i].lstrip()
                            if code_line.startswith('>>>'):
                                # Remove >>> and space
                                current_code.append(code_line[4:] if len(code_line) > 4 else '')
                            else:
                                # This is output or continuation, skip for now
                                pass
                            i += 1
                        
                        # Save the example
                        if current_code:
                            desc = ' '.join(current_description).strip() if current_description else ''
                            parsed['examples'].append({
                                'description': desc,
                                'code': '\n'.join(current_code)
                            })
                            current_description = []
                            current_code = []
                    else:
                        # This is description text
                        if line.strip():
                            current_description.append(line.strip())
                        i += 1
                
                # Save any remaining example
                if current_code:
                    desc = ' '.join(current_description).strip() if current_description else ''
                    parsed['examples'].append({
                        'description': desc,
                        'code': '\n'.join(current_code)
                    })
                    
            else:
                # Google-style: Parse :: format
                lines = examples_content.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i]
                    
                    # Skip empty lines between examples
                    if not line.strip():
                        i += 1
                        continue
                    
                    # Check if this line ends with :: (RST code block marker)
                    if line.rstrip().endswith('::'):
                        description = line.strip()[:-2]  # Remove :: 
                        i += 1
                        
                        # Skip empty line after ::
                        if i < len(lines) and not lines[i].strip():
                            i += 1
                        
                        # Detect the indentation level of the first code line
                        base_indent = 0
                        if i < len(lines) and lines[i].strip():
                            # Count leading spaces on first code line
                            base_indent = len(lines[i]) - len(lines[i].lstrip())
                        
                        # Collect the following indented lines as code
                        code_lines = []
                        while i < len(lines):
                            curr_line = lines[i]
                            
                            # Empty line - include it in code if we're in the middle of code
                            if not curr_line.strip():
                                if code_lines:  # Only add empty lines within code blocks
                                    code_lines.append('')
                                i += 1
                                continue
                            
                            # Check if this line has at least the base indentation
                            curr_indent = len(curr_line) - len(curr_line.lstrip())
                            
                            # If indentation is less than base_indent and line is not empty, stop
                            if curr_indent < base_indent:
                                break
                            
                            # Remove base indentation
                            code_lines.append(curr_line[base_indent:] if len(curr_line) >= base_indent else curr_line.strip())
                            i += 1
                        
                        if code_lines:
                            # Remove trailing empty lines
                            while code_lines and not code_lines[-1].strip():
                                code_lines.pop()
                            
                            parsed['examples'].append({
                                'description': description,
                                'code': '\n'.join(code_lines)
                            })
                        # Don't increment i here - we stopped at the next description line
                    else:
                        # Regular text line (not a code block marker)
                        if line.strip():
                            parsed['examples'].append({
                                'description': line.strip(),
                                'code': None
                            })
                        i += 1
            
            continue
        
        # Handle Returns section differently - it's usually free-form prose, not structured
        if section_name == 'returns':
            # Just store the content as-is for prose rendering
            parsed['returns'] = section_content
            continue
            
        # Parse Parameters section
        # Handle both Google-style (name: description) and NumPy-style (name : type\n    description)
        items = []
        
        # For NumPy-style, parameters are separated by blank lines or new parameter lines
        # For Google-style, they're on single lines with "name: description"
        
        # Try to detect NumPy-style (has " : " with type info on same line)
        is_numpy_style = re.search(r'^\w+\s+:\s+\w+', section_content, re.MULTILINE)
        
        if is_numpy_style:
            # NumPy-style parsing
            # Split by lines that start with a parameter name (non-whitespace at column 0 or after exactly leading whitespace)
            lines = section_content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Check if this is a parameter line (e.g., "x : Tensor | float | int")
                if re.match(r'^(\w+)\s+:\s+(.+)$', line.strip()):
                    match = re.match(r'^(\w+)\s+:\s+(.+)$', line.strip())
                    name = match.group(1)
                    type_hint = match.group(2)
                    i += 1
                    
                    # Collect description lines (indented lines following the parameter)
                    desc_lines = []
                    while i < len(lines) and (lines[i].startswith('    ') or not lines[i].strip()):
                        if lines[i].strip():
                            desc_lines.append(lines[i].strip())
                        i += 1
                    
                    description = ' '.join(desc_lines)
                    items.append({'name': name, 'type': type_hint, 'description': description})
                else:
                    i += 1
        else:
            # Google-style parsing (original logic)
            item_blocks = re.split(r'\n(?=\s{4}\S)', section_content)  # Split on lines with exactly 4 spaces indent
            
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
        returns = parsed_docstring['returns']
        
        # Check if it's a string (prose) or list (structured)
        if isinstance(returns, str):
            # Prose format - just render as markdown
            md.append(f"{returns}\n")
        else:
            # Structured format (legacy)
            for ret in returns:
                md.append(f"- `{ret['type']}`: {ret['description']}\n")
    
    # Examples
    if parsed_docstring.get('examples'):
        md.append("## Examples\n")
        examples = parsed_docstring['examples']
        
        # Check if examples is a list of dicts (new format) or string (old format)
        if isinstance(examples, list):
            for example in examples:
                if example.get('description'):
                    md.append(f"{example['description']}\n")
                if example.get('code'):
                    md.append(f"```python\n{example['code']}\n```\n")
        else:
            # Fallback for old format
            md.append(f"```python\n{examples}\n```\n")
    
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
            # Only use the first level after 'nabla.' (e.g., 'core', 'transforms', 'ops')
            if module_name.startswith('nabla.'):
                parts = module_name.replace('nabla.', '').split('.')
                category = parts[0] if parts else 'other'
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
